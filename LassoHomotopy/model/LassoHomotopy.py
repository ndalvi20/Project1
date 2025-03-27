import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import math

# This is the main model class for LASSO regression using the Homotopy method
class LassoHomotopyModel():
    def __init__(self, reg_param, max_iterations=1000, verbose=False, init_coef=None, scale_mu=1.0):
        self.mu = reg_param                      # Regularization strength
        self.max_iterations = max_iterations     # Max iterations to avoid infinite loop
        self.verbose = verbose                   # Whether to print debug info
        self.init_coef = init_coef               # Optional: start from custom coefficients
        self.scale_mu = scale_mu                 # Scaling factor for mu
        self.coefficients = None                 # Placeholder for learned coefficients
        self.active_set = None                   # Tracks which features are currently used

    def fit(self, X, y):
        # Convert inputs to float if necessary
        if X.dtype.kind not in 'fc':
            X = X.astype(np.float64)
        if y.dtype.kind not in 'fc':
            y = y.astype(np.float64)
        if y.ndim > 1:
            y = y.flatten()

        # Get dimensions
        n, m = X.shape
        mu = self.mu * n * self.scale_mu
        max_iterations = self.max_iterations

        # Initialize coefficients and residuals
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

        # Main loop — this gradually builds up the set of active features
        while iteration_counter < max_iterations:
            iteration_counter += 1
            corr = X.T @ residuals                           # Correlation with residuals
            max_corr = np.max(np.abs(corr))                  # Largest correlation

            if self.verbose:
                print(f"Iteration {iteration_counter}: max_corr = {max_corr}")

            # If max correlation is low or all features are active, stop
            if max_corr < mu or len(active_indices) == m:
                if self.verbose:
                    print("Optimality condition met or all features active.")
                break

            # Add most correlated feature to active set
            new_active = np.argmax(np.abs(corr))
            active_indices.add(new_active)
            if self.verbose:
                print(f"Adding feature {new_active} to active set.")

            active_list = list(active_indices)

            # Compute solution for current active set
            X_active = X[:, active_list]
            theta_active = np.linalg.inv(X_active.T @ X_active) @ (X_active.T @ y - mu * np.sign(corr[active_list]))

            while True:
                # Update residuals and correlations
                theta_old = theta_active.copy()
                residuals = y - X_active @ theta_active
                corr = X.T @ residuals

                # Find inactive features (not in current solution)
                inactive = np.array([j for j in range(m) if j not in active_indices])
                if len(inactive) == 0:
                    break

                max_corr = np.max(np.abs(corr[inactive]))

                # If a new inactive feature is strongly correlated, add it
                if max_corr > mu:
                    new_idx = inactive[np.argmax(np.abs(corr[inactive]))]
                    active_indices.add(new_idx)
                    if self.verbose:
                        print(f"Adding feature {new_idx} from inactive set (max_corr = {max_corr}).")
                    active_list = list(active_indices)
                    X_active = X[:, active_list]
                    theta_active = np.linalg.inv(X_active.T @ X_active) @ (X_active.T @ y - mu * np.sign(corr[active_list]))
                    continue

                # Remove features whose sign flipped (crossed zero) — Homotopy method condition
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

        # Create full coefficient vector (zeros for inactive features)
        coefficients = np.zeros(m)
        for idx, coef in zip(active_list, theta_active):
            coefficients[idx] = coef

        return LassoHomotopyResults(coefficients)


# This class holds the output of the trained model and provides prediction
class LassoHomotopyResults():
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def predict(self, x):
        # Ensure input is float
        if x.dtype.kind not in 'fc':
            x = x.astype(np.float64)
        preds = x @ self.coefficients
        return preds.reshape(-1, 1)  # Ensure 2D output
    
    # Root Mean Square Error: how far are our predictions from actual values on average
    def rmse(self, y_true, y_pred):
        return math.sqrt(mean_squared_error(y_true, y_pred))

    # R-squared Score: how much variance in target is explained by the model
    def r2_score(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    # Mean Absolute Error: average of absolute errors
    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    # Pearson Correlation: how strongly predictions align linearly with actual values
    def correlation(self, y_true, y_pred):
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return float('nan')
        return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]

    # Nicely print all evaluation metrics
    def summary(self, x=None, y_true=None):
        print("Coefficients:", self.coefficients)
        if x is not None and y_true is not None:
            y_pred = self.predict(x)
            print("RMSE: ", round(self.rmse(y_true, y_pred), 4))
            print("R2 Score: ", round(self.r2_score(y_true, y_pred), 4))
            print("MAE: ", round(self.mae(y_true, y_pred), 4))
            print("Correlation (r): ",round(self.correlation(y_true, y_pred), 4))
        else:
            print("Provide X and y to compute evaluation metrics.")