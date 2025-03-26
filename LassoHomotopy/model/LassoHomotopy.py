import numpy as np

class LassoHomotopyModel():
    def __init__(self, reg_param, max_iterations=1000, verbose=False, init_coef=None, scale_mu=1.0):
        self.mu = reg_param
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.init_coef = init_coef
        self.scale_mu = scale_mu
        self.coefficients = None
        self.active_set = None

    def fit(self, X, y):
        n, m = X.shape
        mu = self.mu * n * self.scale_mu
        max_iterations = self.max_iterations

        if self.init_coef is not None:
            theta = self.init_coef.copy()
            active_indices = set(np.nonzero(theta)[0])
            residuals = y - X @ theta
            if self.verbose:
                print("Using provided init_coef. Initial active set:", active_indices)
        else:
            theta = np.zeros(m)
            active_indices = set()
            residuals = y.copy()

        iteration_counter = 0

        while iteration_counter < max_iterations:
            iteration_counter += 1
            corr = X.T @ residuals
            max_corr = np.max(np.abs(corr))
            
            if self.verbose:
                print(f"Iteration {iteration_counter}: max_corr = {max_corr}")

            if max_corr < mu or len(active_indices) == m:
                if self.verbose:
                    print("Optimality condition met or all features active.")
                break

            new_active = np.argmax(np.abs(corr))
            active_indices.add(new_active)
            if self.verbose:
                print(f"Adding feature {new_active} to active set.")

            active_list = list(active_indices)

            X_active = X[:, active_list]
            theta_active = np.linalg.inv(X_active.T @ X_active) @ (X_active.T @ y - mu * np.sign(corr[active_list]))

            while True:
                theta_old = theta_active.copy()
                residuals = y - X_active @ theta_active
                corr = X.T @ residuals
                inactive = np.array([j for j in range(m) if j not in active_indices])

                if len(inactive) == 0:
                    break

                max_corr = np.max(np.abs(corr[inactive]))
                if max_corr > mu:
                    new_idx = inactive[np.argmax(np.abs(corr[inactive]))]
                    active_indices.add(new_idx)
                    if self.verbose:
                        print(f"Adding feature {new_idx} from inactive set (max_corr = {max_corr}).")
                    active_list = list(active_indices)
                    X_active = X[:, active_list]
                    theta_active = np.linalg.inv(X_active.T @ X_active) @ (X_active.T @ y - mu * np.sign(corr[active_list]))
                    continue

                zero_crossings = theta_active * theta_old < 0
                if np.any(zero_crossings):
                    idx_zero_cross = np.where(zero_crossings)[0][0]
                    idx_remove = active_list[idx_zero_cross]
                    active_indices.remove(idx_remove)
                    if self.verbose:
                        print(f"Removing feature {idx_remove} due to sign change.")
                    active_list = list(active_indices)
                    if len(active_list) == 0:
                        theta_active = np.array([])
                        break
                    X_active = X[:, active_list]
                    theta_active = np.linalg.inv(X_active.T @ X_active) @ (X_active.T @ y - mu * np.sign(corr[active_list]))
                    continue

                break

        coefficients = np.zeros(m)
        for idx, coef in zip(active_list, theta_active):
            coefficients[idx] = coef
        return LassoHomotopyResults(coefficients)


class LassoHomotopyResults():
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def predict(self, x):
        return x @ self.coefficients
