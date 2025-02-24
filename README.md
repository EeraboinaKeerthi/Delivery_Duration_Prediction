# Delivery_Duration_Prediction
Business Problem:

Predicting Delivery Duration for Better Customer Experience DoorDash aims to provide accurate estimated delivery times to improve customer satisfaction and operational efficiency. Incorrect delivery time predictions can lead to negative customer experiences, reduced customer retention, and increased costs due to inefficiencies in order fulfillment.

Problem Formulation(Machine Learning Problem):

This is a supervised regression problem where the goal is to predict a continuous target variable—the total delivery duration (in seconds). 
y = actual_delivery_time - created_at 
where: Input Features (X): Includes store details, order characteristics, market conditions, and model-generated estimates. Target Variable (y): Total delivery duration in seconds. Machine learning model should be trained on historical delivery data to identify patterns and predict the time required for future deliveries.

Collect and Label Data:

This step involves reading and transforming raw data into a structured format. The dataset is read from "historical_data.csv". Timestamps (created_at and actual_delivery_time) are converted to datetime format. The target variable actual_total_delivery_duration is created by calculating the time difference in seconds.

Evaluate Data (Data Preprocessing):

Checking unique values in categorical columns: Before encoding, the uniqueness of categorical variables was checked, to know if categorical columns should be one-hot encoded or if they contain too many unique values. 
Helps in identifying categorical features that may need grouping or treatment for missing values. 
Handling Missing values: Identifying missing values in store_primary_category field and missing values were imputed with mode value. 
Dropping Unnecessary Columns: Columns that are not required for model evaluation are removed. Replacing infinite values occured due to division errors with Nans and dropping them.

Feature Engineering:

New features were created to improve model performance.
A new feature busy_dashers_ratio is computed.This captures how occupied dashers are at the time of order placement, which can impact delivery time.
A new feature estimated_non_prep_duration is created.This estimates the total time taken outside of food preparation, helping the model understand how long travel & order placement take.
Encoding categorical features : One-hot encoded order_protocol, market_id, store_primary_category.
The original dataset is concatenated with the encoded categorical features.
The entire dataset is converted to float32.This helps in reducing memory usage and ensuring consistent numerical types.
Feature Selection and Redundancy handling:
Identifying redundant and collinear features:
A correlation matrix is calculated for all numerical features in train_df.
mask = np.triu(np.ones_like(corr, dtype=bool)) is used to hide the upper triangle, since correlation matrices are symmetric (i.e., correlation of A with B is the same as B with A).
Used Seaborn’s heatmap to visualize correlations, it helps identify highly correlated features, which might introduce multicollinearity in ML models.
Dropping highly correlated features:
total_onshift_dashers and total_busy_dashers are highly correlated.
category_indonesian has a zero standard deviation (all values are the same), making it useless.
estimated_non_prep_duration might be redundant as it's a sum of other features.
Creating percent_distinct_item_of_total,measures diversity in the order by computing the percentage of distinct items in the total order.Creating avg_price_per_item,Computes average price per item in an order, helps capture the pricing structure across orders.
Checks final correlations to ensure that redundant features are removed.

Why is the Correlation Matrix Important?
Identifies Redundant Features
If two features are highly correlated (|correlation| > 0.9), one might be unnecessary.
Example: If total_onshift_dashers and total_busy_dashers are highly correlated, we can remove one.
Reduces Multicollinearity
Multicollinearity occurs when independent variables in a regression model are too correlated.
This can make ML models unstable and less interpretable.
Dropping one of the correlated features improves performance.
Helps Feature Selection
If a feature has low correlation (near 0) with the target variable, it might be irrelevant.
We can remove weakly correlated features to simplify the model.
Improves Model Interpretability
Highly correlated features make it hard to understand which feature is impacting the target.
Removing redundancy improves clarity.

Multicollinearity Check:
Detecting and Removing Multicollinearity using Variance Inflation Factor (VIF).
Variance Inflation Factor (VIF) is a statistical measure used to detect multicollinearity in a dataset. Multicollinearity occurs when independent variables (features) are highly correlated, making it difficult for a machine learning model to distinguish their individual effects on the target variable.
Why is VIF Important?
Improves Model Interpretability: If features are highly correlated, it’s difficult to determine their true effect on the target variable.
Reduces Overfitting: Multicollinearity increases model complexity, leading to overfitting.
Enhances Model Stability: High correlation between independent variables causes unstable coefficients in regression models.
If VIF > 20, it should be removed to prevent multicollinearity.
The function calculates VIF for each feature and returns a sorted table.
Finding the feature with the highest VIF and remove it iteratively.
Recompute VIF scores and continue removing features until all features have VIF ≤ 20.
Final list of selected features is stored in selected_features.The feature percent_distinct_item_of_total was found to have high VIF and was dropped.

Feature Selection using Random Forest Importance: We determine the most influential features using Random Forest.
A Random Forest model is trained to predict delivery duration.
Feature importance is extracted using Gini importance (how much each feature reduces uncertainty in predictions).
The most important features are plotted.This helps select only the most relevant features for the model.Extracts and plots the top 35 most important features.

Principal Component Analysis (PCA) for Dimensionality Reduction:
PCA (Principal Component Analysis) is used to check whether dimensionality reduction is useful.
The dataset is standardized using StandardScaler() (PCA requires normalized data).
Cumulative variance is plotted, showing how many principal components are needed to explain 80% of the dataset.If PCA can explain most of the variance with fewer features, we might drop additional features.However, PCA shows that at least 60 components are needed, meaning feature selection based on importance is better than PCA here.PCA shows that we need to use at least 60 representative features to explain 80% of the dataset, which makes the PCA transformation useless since we already have 80 and could select the most important ones based on feature importance. However, if PCA would tell us it can explain the majority of variance with around 10 features - high reduction - we would continue with it.PCA doesn’t provide significant dimensionality reduction.Using top 35 features from feature importance, NOT PCA


Select and Train Model:

Define a Generic Function to Train Any Regression Model:
A generic function to train any regression model (model) and compute RMSE.
The function trains the model, makes predictions on training and test sets, and calculates RMSE.
The verbose flag controls whether the function prints results.
Six different regression models are defined:
Ridge Regression
Decision Tree
Random Forest
XGBoost
LightGBM (LGBM)
MLP (Multi-Layer Perceptron - Neural Network)
Define Different Feature Sets:
Four feature sets are created:
Full Dataset → Uses all features.
Top 40 Features → Selected based on feature importance.
Top 20 Features → Only the most relevant features.
Top 10 Features → The smallest feature subset.
Define Different Scaling Methods
Three different scaling methods:
StandardScaler() → Normalizes data to mean=0, variance=1.
MinMaxScaler() → Scales data between 0 and 1.
NotScale → No scaling applied.

Define a Function to Train and Evaluate a Model
Trains the given model using model.fit(X_train, y_train).
Computes RMSE for both training and test data.
Prints error values for evaluation.
Train Models Across Feature Sets and Scalers
Loop Through All Feature Sets, Scalers, and Models
Iterates over different feature sets (full dataset, top 40, 20, 10 features).
Iterates over different scaling techniques (StandardScaler, MinMaxScaler, No Scaling).
Iterates over multiple regression models (Ridge, Decision Tree, Random Forest, XGBoost, LGBM, MLP).
Trains each model on the dataset with corresponding feature selection and scaling.
Stores RMSE results for comparison.
Stores results in pred_dict for later comparison.


Evaluate Model and Results:
The primary metric used for evaluation is Root Mean Squared Error (RMSE).
RMSE measures how far predictions are from actual values in terms of seconds.
Lower RMSE is better.
RMSE penalizes large errors more heavily, making it a good metric for time-sensitive predictions like delivery time.


Observations:
XGBoost & LGBM consistently outperform other models.
Feature selection helps but doesn’t drastically improve results.
Scaling is essential for MLP but has minimal effect on tree-based models.
Random Forest showed overfitting and needs hyperparameter tuning.

Conclusions:
Best RMSE is 2033 sec (~34 minutes).
A 34-minute average error is too high → Many deliveries could be late by 30-40 minutes.
Customers expect 5-10 min accuracy, not 30+ min delays.
Conclusion: The model does not fully meet business goals because RMSE is too high.
RMSE should be below ~900-1200 seconds (15-20 minutes) for practical use.
If the model is consistently off by 30+ minutes, customers will lose trust in ETAs.


Fine Tuning:
Fine-tune hyperparameters for LGBM and XGBoost (Grid Search, Bayesian Optimization).
Reduce overfitting in Random Forest (increase regularization, limit tree depth).
Test ensemble models (combine LGBM and XGBoost for better predictions).
Analyze which features contribute most to error.
Try deep learning models with better tuning to see if performance improves.

Fine tuning 1:
Changing the Target Variable (Reframing the Problem):
Instead of predicting the total delivery duration,if the model predicts food preparation time:

Observations:

LGBM still has the lowest RMSE (2035 sec), showing it remains the best model.
XGBoost slightly improved but is still behind LGBM.
Random Forest improved a little but still overfits.
Decision Tree performed worse, likely due to dropping features.
MLP did not improve significantly, suggesting neural networks are not optimal for this problem.

Reframing the problem (predicting prep time instead of total time) helped a bit but didn’t drastically lower RMSE.RMSE (2035 sec) is still too high for practical use
Further improvements are needed.



