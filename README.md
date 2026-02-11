# BRFSS Classification — Regularized Logistic Regression with Robust Preprocessing 

This repository trains a binary classifier on tabular health survey data using a robust preprocessing pipeline and regularized logistic regression.  
The run.py file outputs a submission.csv file with predictions for the test set.

---------------------------------------------------------------------

## Highlights

- Careful handling of missing values and special codes
- Automatic categorical detection and selective one-hot encoding
- Median imputation using train-only statistics
- Standardization (mean/std computed from training set)
- PCA applied on continuous features to retain 95% of explained variance
- Regularized Logistic Regression, with optional k-fold cross-validation over lambda, gamma and threshold
- Tunable classification threshold (default: 0.208 for optimal F1 score)

---------------------------------------------------------------------

## Repository Structure

```text
├── data_cleaning/
│   ├── pca.py                      # PCA class and choose_d_by_variance()
│   ├── filling_features.py         # Functions to fill important sparse features
│   ├── col_clean.py                # Column utilities (one-hot encoding, dropping NaNs, etc.)
│   └── col_helper.py               # Helper methods for data filling
├── dataset/                        # Expected by load_csv_data("dataset", sub_sample=False)
│   ├── x_train.csv
│   ├── x_test.csv
│   └── y_train.csv
├──plots/                     
│   ├── plots.py                    # Generate plots of the data exploration part
│   ├── preprocessing_impact_plot.py # Generate the incremental effect of preprocessing steps plot
│   └── 
├── cross_validation.py             # K-fold cross-validation utilities
├── data_preprocessing.py           # full preprocessing pipeline
├── functions.py                    # functions used to implement the implementations.py methods (compute mse, compute gradient ...)
├── helpers.py                      # given helper functions 
├── implementations.py              # logistic_regression, reg_logistic_regression, etc.
├── README.md
├── run_other_models.py             # script to run other models (mean squred error gd/sgd, ridge regression etc ... )
├── run.py                          # Main script to run the full pipeline and train the regularized logistic regression model
└── submission.csv                  # Generated output file 
```


---------------------------------------------------------------------

## Quickstart to run the regulaized logistic regression model

1. Download the dataset  
   Download it from: https://github.com/epfml/ML_course/tree/main/projects/project1/data  
   Unzip it into the project root folder.

2. Place your data  
   Ensure the dataset is under ./dataset/ in the format expected by load_csv_data.

3. Run the script  
   Command:
   python run.py

This will:
- Load the data via load_csv_data("dataset", sub_sample=False)
- Run the full preprocessing pipeline
- Train a regularized logistic regression model with:
  max_iters = 2000, lambda = 0.001, gamma = 0.1
- Create submission.csv with predictions in {-1, +1}, using a threshold of 0.208 for classification

#### Command-Line Arguments

The script run.py accepts two optional arguments to control the training mode and model type.
You can specify them when running the script from the terminal:
python run.py --mode <mode> --model <model>
- --mode defines how the script runs:
   - full: trains the model on the full dataset and generates submission.csv (default)
   - cv: performs 5-fold cross-validation to jointly-tune hyperparameters (λ, γ, threshold)
   - val: runs a quick 80/20 validation for local performance evaluation
- --model defines the model variant for the cross validation mode
   - logreg: standard logistic regression
   - reglogreg: L2-regularized logistic regression 

The default command python run.py will train the model on the full dataset and generate a submission.csv file.

---------------------------------------------------------------------

## End-to-End Pipeline Overview

1. Load the dataset
   - Import the training and test data, along with labels and IDs.

2. Fill important sparse features
   - Fill specific medically relevant columns (e.g., diabetes or cardiac rehab indicators) with default values before dropping columns.

3. Remove irrelevant columns
   - Drop unnecessary features based on prior analysis to keep only useful subsets.

4. Normalize the labels
   - Convert labels from {-1, +1} to {0, 1} for logistic regression.

5. Detect and handle special codes such as 7777, 8888 etc...
   - Identify special codes in each column and replace them with NaN.

6. Drop columns with too many missing values
   - Remove columns where the fraction of missing values exceeds 60%.

7. Drop zero-variance columns
   - Remove features that are constant or provide no information.

8. Identify categorical and continuous features
   - Classify columns as categorical (few unique values) or continuous (many unique values).

9. One-hot encode categorical variables
   
10. Drop linearly dependent categorical columns
    - Remove linearly dependent features using QR decomposition.

11. Impute remaining missing values
    - Replace remaining NaN values in continuous columns with the median value computed from the training set.

12. Standardize all features
    - Apply z-score standardization using the mean and standard deviation from the training set.

13. Reduce dimensionality with PCA
    - Apply PCA on continuous features to retain 95% of the explained variance and reduce redundancy.

14. Recombine categorical and PCA-transformed features
    - Merge the one-hot encoded categorical columns with the PCA-reduced continuous features.

15. Add a bias term
    - Append a column of ones to the feature matrix to represent the intercept term.

16. Train the regularized logistic regression model
    - Fit the model on the processed training data using gradient descent.

17. Compute predicted probabilities on the test set

18. Apply a custom threshold (0.208)
    - Convert probabilities into binary predictions based on the empirically optimized F1 threshold.

19. Generate and save the submission file
    - Map predictions back to {-1, +1} and save them in submission.csv with their corresponding IDs.

---------------------------------------------------------------------

## Thresholding Tips

- The dataset is imbalanced, so using the default threshold of 0.5 would reduce F1 performance.
- The chosen threshold 0.208 was found empirically to maximize the F1 score.

---------------------------------------------------------------------

## Output

- submission.csv with two columns:
  - Id
  - Prediction
- Predictions are in {-1, +1}

---------------------------------------------------------------------
## Running other models
In addition to run.py, which trains the regularized logistic regression model, you can use run_other_models.py to train and generate predictions for the other baseline models implemented in the project.

To execute the script, specify the model you want to train using the --model argument:

Mean Squared Error (Gradient Descent)
- python run_other_models.py --model mean_square_gd

Mean Squared Error (Stochastic Gradient Descent)
- python run_other_models.py --model mean_square_sgd

Least Squares
- python run_other_models.py --model least_squares

Ridge Regression
- python run_other_models.py --model ridge_regression

Logistic Regression
- python run_other_models.py --model logistic_regression

