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



