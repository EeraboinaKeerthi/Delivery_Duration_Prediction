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

