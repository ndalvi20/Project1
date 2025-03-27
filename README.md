## Team Members
1. Nishant Dalvi - ndalvi@hawk.iit.edu **(A20556507)** (Member 1)
2. Shriniwas Oza - soza1@hawk.iit.edu **(A20568892)** (Member 2)

## Project 1 - LASSO Regularized Regression using the Homotopy Method

## Overview
This project implements the **LASSO (Least Absolute Shrinkage and Selection Operator)** regression model via the **Homotopy method**, as outlined in [this paper](https://people.eecs.berkeley.edu/~elghaoui/Pubs/hom_lasso_NIPS08.pdf). The model was implemented from the scratch in NumPy and SciPy, focusing on first principles and with no existing LASSO implementation available in libraries such as Scikit-Learn.

The LASSO technique is very suitable for feature selection and regularized regression, particularly for the high-dimensional and collinear data. Although Scikit-Learn has been utilized, it is used **only for evaluation and comparison purposes** (e.g., for OLS baseline and computing metrics).

For robustness, the project has several test cases to check for correctness, sparsity, and generalization.

## Team Roles

This implementation was completed as a team project. 

## Member 2 - Contributions (Testing, Debugging, Evaluation Metrics, Robustness Handling, Final Verification)

- Test Design and Implementation:
    - Created and executed 10 end-to-end test cases for:
        - Prediction correctness
        - Generalization on unseen data
        - Sparsity enforcement on collinear data sets
        - Robustness to noisy data
        - Dealing with edge cases like constant targets and all-zero features
        - Comparative analysis: OLS vs. LASSO under various conditions (collinearity, noise, constant targets)
- Evaluation Metrics:
    - Included substantive measures like RMSE, R2 Score, MAE, and Pearson Correlation to give an overall model performance evaluation.
- Edge Case Handling:
    - Identified and fixed edge-case failures, making `fit()` more robust with pre-emptive handling of such cases such as zero active features.
- Model Validation Enhancements:
    - Added a `summary()` method to print test evaluation statistics for each test to facilitate transparency and readability of model-side evaluations.

## How to Run

### Step 1: Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # For Linux/macOS
venv\Scripts\activate         # For Windows
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run the tests using PyTest
```bash
pytest -s LassoHomotopy/tests/test_LassoHomotopy.py
```
The `-s` flag prints formatted output for each test.

You will see **10 test outputs**, each formatted with headers and evaluation metrics.

## What the Model Does
This implementation solves the LASSO regression problem using the Homotopy algorithm. It is particularly useful when:
- You expect some features to be irrelevant (it **automatically sets some coefficients to zero**)
- You have **collinear features** (standard linear regression may break down)
- You want a simple, interpretable model that avoids overfitting

## How We Tested the Model
We implemented **10 rigorous test cases** to evaluate different aspects of model performance:

| Test No. | Description |
|----------|-------------|
| Test 1   | Sanity check using professor's small dataset |
| Test 2   | Generalization check on custom small dataset (train/test split) |
| Test 3   | Sparsity on professor's collinear dataset |
| Test 4   | Sparsity on custom collinear dataset |
| Test 5   | Robustness on noisy data |
| Test 6   | All-zero input features edge case |
| Test 7   | Constant target behavior test |
| Test 8   | Compare OLS vs LASSO (collinearity) |
| Test 9   | Compare OLS vs LASSO (noise) |
| Test 10  | Compare OLS vs LASSO (constant output) |

All tests include detailed output for RMSE, R2 score, MAE, and correlation.

## Limitations / Known Challenges
- When **all features are zero**, we handle it gracefully by returning a zero vector (fixed).
- When the **target is constant**, the R2 score and correlation become less meaningful (returns `NaN` where applicable).
- The model **may underfit** when regularization is too strong - this is by design in LASSO.

## Answering the README Questions

### What does the model do and when should it be used?
This model solves **LASSO regularized linear regression** using the Homotopy Method. It is particularly useful when:
- We expect **sparse solutions**, i.e., only a few features contribute significantly
- Data exhibits **collinearity** (highly correlated features)
- We want **feature selection** alongside regression

### How did you test your model?
- Using **10 test cases** on small, collinear, noisy, and synthetic datasets
- Compared performance against **OLS (Ordinary Least Squares)** in specific scenarios
- Checked for expected sparsity, robustness, and behavior under edge conditions (zero input, constant targets)
- Tests included **assertions** on RMSE, R2, MAE, correlation, and coefficient behavior

### What parameters have you exposed to users?
- `reg_param`: Regularization strength
- `max_iterations`: Cap on iteration loops
- `init_coef`: Optional starting coefficients (warm start)
- `scale_mu`: Scaling factor for in homotopy formulation
- `verbose`: Enable iterative debug logs

### Are there inputs the model struggles with?
- When input **features are all zeros**, model exits early with zero predictions (handled safely)
- For **constant target outputs**, model underfits due to regularization (expected)
- These are not implementation bugs but **inherent limitations of LASSO** (which penalizes magnitude)

## Hint Evaluation - Collinear Data
> "What should happen when you feed it highly collinear data?"

We confirmed that LASSO enforces **sparsity** and **eliminates redundant features** when data contains multicollinearity. This was explicitly tested in:
- `test_sparsity_on_professor_collinear_data`
- `test_sparsity_on_custom_collinear_data`
- `test_compare_lasso_vs_ols_collinearity`

In each case, LASSO produced significantly more zero coefficients compared to OLS.

## Final Notes
This submission includes:
- Fully implemented `LassoHomotopyModel`
- A rich test suite (`test_LassoHomotopy.py`) covering all required behaviors and edge cases
- Evaluation metrics implemented directly in the results class
