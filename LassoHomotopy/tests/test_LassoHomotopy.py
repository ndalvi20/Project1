import csv
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from model.LassoHomotopy import LassoHomotopyModel

# TEST 1: Basic sanity check using professor's small test dataset
# ---------------------------------------------------------------
# This test ensures that the Lasso model:
# - Trains successfully on a small dataset
# - Produces predictions of the correct shape
# - Outputs values within a reasonable numerical range
def test_predict():
    # Initialize the Lasso model with a regularization parameter
    model = LassoHomotopyModel(reg_param=0.1)

    # Load the small test dataset provided by the professor
    data = []
    with open("LassoHomotopy/tests/small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract feature matrix (X) and target variable (y) from the CSV
    X = numpy.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[float(v) for k, v in datum.items() if k == 'y'] for datum in data])
    
    # Train the model on the data
    results = model.fit(X, y)
    preds = results.predict(X)

    print("\n" + "="*30)
    print("TEST 1: Predict Sanity Check")
    print("="*30)

    # Print first few predictions and targets for manual inspection
    print("Predictions:", preds[:5])
    print("Targets:", y[:5])
    print("Prediction Shape:", preds.shape)
    print("Target Shape:", y.shape)

    # Show performance summary using built-in metrics
    results.summary(X, y)

    # Check that prediction dimensions match the target's
    assert preds.shape == y.shape

    # Ensure predictions are within a plausible numerical range
    assert numpy.all(preds > -1e6) and numpy.all(preds < 1e6)

# TEST 2: Generalization on Small Dataset
# ---------------------------------------
# This test checks whether the Lasso model can generalize well to unseen data.
# It uses a custom small dataset and splits it into training and testing sets.
# After training, we evaluate the model on the test set using RMSE, R2, MAE, and correlation.
def test_generalization_on_small_dataset():
    # Initialize the model with a small regularization parameter
    model = LassoHomotopyModel(reg_param=0.1)

    # Load custom small dataset for generalization test
    data = []
    with open("LassoHomotopy/datasets/small_dataset.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract feature matrix (X) and target variable (y) from the CSV
    X = numpy.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[float(v) for k, v in datum.items() if k == 'y'] for datum in data])

    # Split dataset into training and test portions (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the model and make predictions on the test set
    results = model.fit(X_train, y_train)
    preds = results.predict(X_test)
    
    print("\n" + "="*30)
    print("TEST 2: Generalization on Small Dataset")
    print("="*30)

    # Visual summary of results using model-side evaluation methods
    results.summary(X_test, y_test)

    # Assert evaluation metrics are within acceptable range for a "good" model
    assert results.rmse(y_test, preds) < 10.0
    assert results.r2_score(y_test, preds) > 0.5
    assert results.mae(y_test, preds) < 5.0
    assert results.correlation(y_test, preds) > 0.85


# TEST 3: Sparsity on Professor's Collinear Data
# ----------------------------------------------
# This test checks if the model enforces sparsity when given highly collinear features.
# LASSO should automatically zero out redundant (correlated) features.
def test_sparsity_on_professor_collinear_data():
    # Initialize the model with a small regularization parameter
    model = LassoHomotopyModel(reg_param=0.1)

    # Load the professor-provided collinear dataset
    data = []
    with open("LassoHomotopy/tests/collinear_data.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract feature matrix (X) and target variable (y) from the CSV
    X = numpy.array([[float(v) for k, v in datum.items() if k.lower().startswith('x_')] for datum in data])
    y = numpy.array([[float(v) for k, v in datum.items() if k.lower() == 'target'] for datum in data])

    # Train the model and collect results
    results = model.fit(X, y)
    coef = results.coefficients
    print("\n" + "="*30)
    print("TEST 3: Sparsity on Professor's Collinear Data")
    print("="*30)
    print("Professor Coefficients:", coef)
    results.summary(X, y)

    # At least one coefficient should be close to zero - showing sparsity
    assert numpy.any(numpy.abs(coef) < 1e-3), "Model failed to enforce sparsity on collinear data"

# TEST 4: Sparsity on Custom Collinear Data
# -----------------------------------------
# This test validates that the model works correctly on custom generated collinear dataset.
# LASSO should ignore redundant features and return a sparse solution by driving some coefficients to zero.
def test_sparsity_on_custom_collinear_data():
    # Initialize the model with a small regularization parameter
    model = LassoHomotopyModel(reg_param=0.1)

    # Load custom-generated collinear dataset
    data = []
    with open("LassoHomotopy/datasets/collinear_dataset.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract features (X) and target (y)
    X = numpy.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[float(v) for k, v in datum.items() if k == 'y'] for datum in data])

    # Fit the model and evaluate
    results = model.fit(X, y)
    coef = results.coefficients
    print("\n" + "="*30)
    print("TEST 4: Sparsity on Custom Collinear Data")
    print("="*30)
    print("Custom Coefficients:", coef)
    results.summary(X, y)

    # Assert that at least one coefficient is effectively zero, confirming sparsity
    assert numpy.any(numpy.abs(coef) < 1e-3), "Model did not produce a sparse solution"

# TEST 5: Robustness on Noisy Data
# --------------------------------
# This test checks if the model can still perform well when the data has noise.
# A robust LASSO model should still generalize decently, even if the data isn't perfect.
def test_robustness_on_noisy_data():
    # Initialize the model with a small regularization parameter
    model = LassoHomotopyModel(reg_param=0.1)

    # Load noisy dataset
    data = []
    with open("LassoHomotopy/datasets/noisy_dataset.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract features (X) and target (y)
    X = numpy.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[float(v) for k, v in datum.items() if k == 'y'] for datum in data])

    # Split into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit the model and predict
    results = model.fit(X_train, y_train)
    preds = results.predict(X_test)
    print("\n" + "="*30)
    print("TEST 5: Robustness on Noisy Data")
    print("="*30)
    results.summary(X_test, y_test)

    # Ensure model doesn't totally break under noise
    assert results.rmse(y_test, preds) < 20.0
    assert results.r2_score(y_test, preds) > 0.0
    assert results.mae(y_test, preds) < 10.0
    assert results.correlation(y_test, preds) > 0.6

# TEST 6: All-Zero Features Edge Case
# -----------------------------------
# This test checks how the model behaves when all input features are zero.
# Since there's no signal in the data, we expect the model to exit early
# and return all-zero coefficients without throwing an error.
def test_all_zero_features():
    # Initialize the model with a small regularization parameter
    model = LassoHomotopyModel(reg_param=0.1)

    # Create dummy data where all features and targets are zero
    X = numpy.zeros((50, 10)) # 50 samples, 10 features
    y = numpy.zeros((50, 1)) # Target is also all zeros
    print("\n" + "="*30)
    print("TEST 6: All-Zero Features Edge Case")
    print("="*30)
    try:
        # Fit and evaluate
        results = model.fit(X, y)
        preds = results.predict(X)
        results.summary(X, y)
        # Since everything is zero, predictions should be exactly zero
        assert numpy.all(preds == 0), "Prediction should be zero for all-zero input"
        print("PASS: Model exited early with zero input (no active features)")
    except Exception as e:
        assert False, f"Model failed on all-zero input: {e}"


# TEST 7: Constant Target Behavior
# --------------------------------
# This test evaluates how the model handles a dataset where the target output is the same for all samples.
# Ideally, a good regression model should predict close to the constant value.
# But since LASSO adds regularization, it may underfit in this scenario - which is expected behavior.
def test_constant_target_output():
    # Initialize the model with a small regularization parameter
    model = LassoHomotopyModel(reg_param=0.1)

    # Generate data: 100 samples, 5 random features, constant target value
    rng = numpy.random.default_rng(seed=42)
    X = rng.normal(0, 1, size=(100, 5))
    y = numpy.full((100, 1), 7.0)  # Constant output

    # Split into train and test (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model and predict
    results = model.fit(X_train, y_train)
    preds = results.predict(X_test)
    print("\n" + "="*30)
    print("TEST 7: Constant Target Behavior")
    print("="*30)
    # Evaluate using model-side methods
    results.summary(X_test, y_test)

    # Since regularization may prevent perfect fit, allow some tolerance in assertions
    assert results.rmse(y_test, preds) < 10.0
    assert results.r2_score(y_test, preds) < 0.1 # R2 will likely be near 0
    assert results.mae(y_test, preds) < 10.0
    r = results.correlation(y_test, preds)
    assert numpy.isnan(r) or (-1.0 <= r <= 1.0)
    print("NOTE: LASSO underfits constant target - expected behavior due to regularization.")


# TEST 8: OLS vs LASSO - Collinearity
# -----------------------------------
# This test compares LASSO model to ordinary least squares (OLS) when faced with highly collinear data.
# LASSO is expected to zero out redundant features (i.e., produce sparse coefficients),
# whereas OLS will assign non-zero weights even to redundant features.
def test_compare_lasso_vs_ols_collinearity():

    # Load professor-provided collinear dataset
    data = []
    with open("LassoHomotopy/tests/collinear_data.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract features and target
    X = numpy.array([[float(v) for k, v in datum.items() if k.lower().startswith('x_')] for datum in data])
    y = numpy.array([[float(v) for k, v in datum.items() if k.lower() == 'target'] for datum in data])

    # Initialize the model with a small regularization parameter
    lasso = LassoHomotopyModel(reg_param=0.1)

    # Fit LASSO
    lasso_results = lasso.fit(X, y)
    lasso_coefs = lasso_results.coefficients

    # Fit OLS (no regularization)
    ols = LinearRegression()
    ols.fit(X, y)
    ols_coefs = ols.coef_.flatten()

    print("\n" + "="*30)
    print("TEST 8: OLS vs LASSO - Collinearity")
    print("="*30)

    # Compare coefficients
    print("OLS Coefficients:   ", ols_coefs)
    print("LASSO Coefficients: ", lasso_coefs)

    # Count how many coefficients are zero (sparsity check)
    lasso_zero = numpy.sum(numpy.abs(lasso_coefs) < 1e-3)
    ols_zero = numpy.sum(numpy.abs(ols_coefs) < 1e-3)

    print(f"Zero Coefficients -> LASSO: {lasso_zero}, OLS: {ols_zero}")
    # Basic assertion to show LASSO is sparser than OLS
    assert lasso_zero > ols_zero, "LASSO should have more zeros (sparser) than OLS"

# TEST 9: OLS vs LASSO - Noise
# ----------------------------
# This test compares how LASSO and OLS perform on noisy data.
# While both models might fit the data well, LASSO may generalize better by avoiding overfitting.
# We compare them using RMSE to see if LASSO stays competitive even in the presence of noise.
def test_compare_lasso_vs_ols_noise():

    # Load noisy dataset
    data = []
    with open("LassoHomotopy/datasets/noisy_dataset.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

    # Extract features and targets
    X = numpy.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = numpy.array([[float(v) for k, v in datum.items() if k == 'y'] for datum in data])

    # Split into train and test (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the model with a small regularization parameter
    # Fit LASSO
    lasso = LassoHomotopyModel(reg_param=0.1)
    lasso_results = lasso.fit(X_train, y_train)
    lasso_preds = lasso_results.predict(X_test)

    # Calculate RMSE
    lasso_rmse = lasso_results.rmse(y_test, lasso_preds)

    # Fit OLS
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_preds = ols.predict(X_test)

    # Calculate RMSE
    ols_rmse = numpy.sqrt(numpy.mean((ols_preds - y_test)**2))
    print("\n" + "="*30)
    print("TEST 9: OLS vs LASSO - Noise")
    print("="*30)

    # Print comparison
    print("LASSO RMSE:", lasso_rmse)
    print("OLS RMSE:  ", ols_rmse)

    # Check that LASSO does not perform significantly worse
    assert lasso_rmse < ols_rmse + 3, "LASSO should be close to or better than OLS with noisy data"


# TEST 10: Compare OLS and LASSO - Constant Target Case
#   This test evaluates how LASSO and OLS behave when the target variable is constant.
#   In such cases:
#     - OLS will likely predict the constant perfectly, leading to RMSE = 0.
#     - LASSO may underfit due to regularization pushing coefficients toward zero.
def test_compare_lasso_vs_ols_constant_target():
    rng = numpy.random.default_rng(seed=42)
    X = rng.normal(0, 1, size=(100, 5))
    y = numpy.full((100, 1), 7.0) # Constant target

    # Split into train and test (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the model with a small regularization parameter
    # Train LASSO
    lasso = LassoHomotopyModel(reg_param=0.1)
    lasso_results = lasso.fit(X_train, y_train)
    lasso_preds = lasso_results.predict(X_test)
    lasso_rmse = lasso_results.rmse(y_test, lasso_preds)

    # Train OLS
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_preds = ols.predict(X_test)
    ols_rmse = numpy.sqrt(numpy.mean((ols_preds - y_test)**2))

    print("\n" + "="*30)
    print("TEST 10: OLS vs LASSO - Constant Output")
    print("="*30)

    # Print metrics for both models
    print("LASSO RMSE:", lasso_rmse)
    print("OLS RMSE:  ", ols_rmse)

    # Assertion: LASSO should not perform significantly worse than OLS
    assert lasso_rmse >= ols_rmse, "OLS should outperform LASSO for constant target due to no regularization"