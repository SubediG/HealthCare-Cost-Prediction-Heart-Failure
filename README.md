**Healthcare Cost Prediction by Linear regression and Decision Tree**

Project Overview
This project aims to analyze hospital-related data from two CSV files: Hospital General Information and Medicare Inpatient Hospital by Provider and Service 2018. The analysis involves data cleaning, merging, feature selection, and predictive modeling to understand key factors influencing Medicare payment amounts. The project also includes visualizations and evaluation metrics to assess model performance.

Data Sources : cms.gov
Contains general information about hospitals, including hospital type, ownership, and ratings.
Medicare_Inpatient_Hospital_by_Provider_and_Service_2018_data.csv
Provides Medicare payment details for various inpatient services offered by hospitals.

**Libraries Used**
The following R libraries are utilized in this project:
dplyr
tidyverse
ggplot2
caret
rpart
rpart.plot
rattle
neuralnet
corrplot
reshape2

**Key Steps in the Analysis**
1. Data Loading and Preprocessing
Load the two datasets using read.csv().
Rename the ZIP.Code column in the Medicare dataset to ensure consistency for merging.
Merge the datasets using inner_join() on the ZIP.Code column.

2. Missing Value Identification
Calculate the total number of missing values in the merged dataset using is.na().

3. Correlation Analysis
Extract numeric features from the dataset.
Calculate a correlation matrix for numeric features.
Create a heatmap to visualize the correlations using ggplot2.

4. Categorical Variable Identification
Identify categorical variables in the dataset.

5. ANOVA TestPerform an ANOVA test to identify categorical variables that significantly impact the Avg_Mdcr_Pymt_Amt feature.

6. Feature Selection
Select relevant features based on correlation analysis and ANOVA test results.

7. Data Filtering
Filter the dataset to include only heart failure-related cases (DRG Codes 291, 292, 293).
Select the identified features for further analysis.

8. Data Transformation
Convert categorical variables to factors for modeling purposes.

9. Train-Test Split
Split the dataset into training (80%) and testing (20%) sets using createDataPartition() from the caret package.

10. Data Normalization
Normalize the training and testing datasets using the preProcess() function.

11. Linear Regression Model
Build a linear regression model with interaction and polynomial terms to capture non-linearity.
Use cross-validation to evaluate the model.
Predict values on the test dataset and calculate evaluation metrics:
Build a Lasso regression model using the glmnet method to address potential multicollinearity.
Perform cross-validation to select the best lambda value.

13. Decision Tree Model
Build a decision tree model using the rpart package.
Plot the decision tree using rpart.plot() for visualization.
Use cross-validation to prune the tree and avoid overfitting.

14. Model Evaluation
Calculate evaluation metrics for both training and testing datasets.
Display the results in a data frame.

15. Visualization
Create a scatter plot of actual vs. predicted values using ggplot2.
Include a red line representing perfect prediction.
Create a pruned decision tree 


**Evaluation Metrics**
The project evaluates model performance using the following metrics:
RMSE (Root Mean Square Error): Measures the average prediction error.
R² (Coefficient of Determination): Indicates the proportion of variance explained by the model.

**Future Improvements**
Explore additional machine learning models for improved predictions.
Perform feature engineering to create new variables that may improve model accuracy.
Investigate the impact of other DRG codes on Medicare payment amounts.


**Conclusion**
This project demonstrates a comprehensive approach to hospital data analysis, including data cleaning, feature selection, and predictive modeling. The models built in this project provide insights into factors influencing Medicare payment amounts and can be further refined for better accuracy and interpretability.


