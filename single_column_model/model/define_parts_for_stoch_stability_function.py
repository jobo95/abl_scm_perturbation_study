import numpy as np


def make_stochastic_grid(params, z):
    H_s_det_idx, H_s, N_s, delta_s = calculate_stochastic_grid_parameter(params, z)
    stoch_grid = define_grid_for_stoch_stab_function(params.z0, params.z0 + H_s, N_s)

    return stoch_grid, H_s_det_idx, H_s, N_s, delta_s


def calculate_stochastic_grid_parameter(model_params, det_grid):
    # Step 1: Define distance between grid points
    # The distance between grid points on the stochastic grid can be at most the smallest grid point distance of the
    # deterministic grid
    delta_s = det_grid[1] - det_grid[0]

    # Step 2: Height of stochastic grid
    # Calculate height of stochastic grid
    H_s_total = model_params.z_l * model_params.stoch_domain_ext
    # Find corresponding grid point of the deterministic grid
    H_s_total_det_idx = np.abs(det_grid - H_s_total).argmin()
    H_s_total_det = det_grid[H_s_total_det_idx]
    # The stochastic grid starts at s=0 not z0
    H_s_total = H_s_total_det - model_params.z0

    # Step 3: Number of grid points
    N_s = int(np.ceil(H_s_total / delta_s))

    # Step 4: Recalculate distance between grid points to account for rounding in step 3
    delta_s = H_s_total / N_s

    return H_s_total_det_idx, H_s_total, N_s, delta_s


def define_grid_for_stoch_stab_function(lowest_grid_point, height, num_grid_points):
    return np.linspace(lowest_grid_point, height, num_grid_points)


def get_stoch_stab_function_parameter(richardson_num, perturbation_strength):
    return (
        define_stoch_stab_function_param_Lambda(richardson_num),
        define_stoch_stab_function_param_Upsilon(richardson_num),
        define_stoch_stab_function_param_Sigma(
            richardson_num, sigma_s=perturbation_strength
        ),
    )


def define_stoch_stab_function_param_Lambda(Ri):
    lambda_1 = 9.3212
    lambda_2 = 0.9088
    lambda_3 = 0.0738
    lambda_4 = 8.3220
    return lambda_1 * np.tanh(lambda_2 * np.log10(Ri) - lambda_3) + lambda_4


def define_stoch_stab_function_param_Upsilon(Ri):
    upsilon_1 = 0.4294
    upsilon_2 = 0.1749
    return 10 ** (upsilon_1 * np.log10(Ri) + upsilon_2)


def define_stoch_stab_function_param_Sigma(Ri, sigma_s=0.0):
    sigma_1 = 0.8069
    sigma_2 = 0.6044
    sigma_3 = -0.8368 # wrong documentation in Boyko and Vercauteren 2023!
    return 10 ** (sigma_1 * np.tanh(sigma_2 * np.log10(Ri) - sigma_3) + sigma_s)


def initialize_SDEsolver(params):
    M = 1
    q = 15

    stoch_solver = SDEsolver(params.Ns_n, M, params.dz_s, q)

    return stoch_solver, params


class SDEsolver:
    def __init__(self, dof, ext_dof, dz, q):
        self.__dof = dof
        self.__ext_dof = ext_dof
        self.__dz = dz
        self.__q = q
        self.__state = np.zeros(self.__dof)

    def __sample_gaussian_with_circulant_covariance(self, vector):
        """See page 245 and following of Lord et al., 2014, for a detailed description of the method. Same notation is
        used."""
        # mirror the vector without first! and last element
        vector_mir = vector[-2:0:-1]

        # extend vector to make it conjugate symmetric
        vector_circ = np.concatenate((vector, vector_mir), axis=0)

        N_circ = np.size(vector_circ)

        # Compute the eigenvalues
        # Note: fft.ifft computes the one-dimensional inverse discrete Fourier Transform
        d = np.real(np.fft.ifft(vector_circ)) * N_circ

        # Split the eigenvalues into positive and negative ones
        d_min = np.copy(d)
        d_min[d_min > 0] = 0
        d_pos = np.copy(d)
        d_pos[d_pos < 0] = 0

        # Inform that the matrix is non-negative
        if np.max(-d_min) > 1e-9:
            print(
                "Covariance matrix is not non-negative.",
                "Max negative value: ",
                np.max(-d_min),
            )

        # Generate random variable with complex gaussian distribution
        xi = np.dot(np.random.randn(N_circ, 2), np.array([1, 1j]))

        # Calculate y
        y = np.multiply(np.power(d_pos, 0.5), xi)

        # Calculate Z
        Z = np.fft.fft(y) / np.sqrt(N_circ)

        # select the sample paths
        N = np.size(vector)

        return np.real(Z[0:N])

    def __gauss_sampling(self, N, M, dt, q):
        """Sample spatially correlated noise.
        params:
            N: is length of the sampled space
            M: is extensions to better approx the non-negative matrix
            dt: space increment
            q: parameter for matern covariance. i.e. length of correlation
        """
        N_dash = N + M - 1
        t = dt * np.arange(N_dash)

        # Define exponential covariance matrix
        c = np.exp(-((t / q) ** 2))

        # Sample random variable with X~N(0,c)
        X = self.__sample_gaussian_with_circulant_covariance(c)

        return X[0:N]

    def evolve(self, phi_k, dt, Lambda, Upsilon, Sigma, v_sqrt, lz):
        """Solve phi SDE for next step, i.e. k+1."""
        # Sample spatially correlated noise, i.e. noise vector for current time step and full height of stochastic grid
        dW = self.__gauss_sampling(self.__dof, self.__ext_dof, self.__dz, lz)

        # Define all parts of dphi
        # Note: the data-driven identification of the parameters was done based on hourly time units. The constant tau_h
        # transforms the units of the equation into seconds for the numerical implementation. Consider that due to
        # E(dWt)^2 = dt, the process dWt has the units of sqrt(time), and hence the transformation of units for the
        # noise (stochastic) term is different than in the drift (deterministic) term.
        tau_h = 3600
        dphi_deterministic_part = (
            1.0 + Lambda * phi_k - Upsilon * np.power(phi_k, 2)
        ) / tau_h
        dphi_stochastic_part = Sigma * phi_k / np.sqrt(tau_h)

        derivative_dphi_stochastic_part = Sigma / np.sqrt(tau_h)

        # Calculate the value of phi at the next iteration step with the Milstein method
        phi_kp1 = (
            phi_k
            + dphi_deterministic_part * dt
            + dphi_stochastic_part * v_sqrt * dW
            + 0.5
            * dphi_stochastic_part
            * derivative_dphi_stochastic_part
            * (np.power(v_sqrt * dW, 2) - dt)
        )

        self.__state = phi_kp1

    def getState(self):
        return self.__state
