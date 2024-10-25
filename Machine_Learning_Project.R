
library(dplyr)
library(tidyverse)
library(ggplot2)
library(maps)
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(neuralnet)
library(corrplot)
library(reshape2)

# Load the data
hospital_data <- read.csv("Hospital_General_Information.csv")
medicare_data <- read.csv("Medicare_Inpatient_Hospital_by_Provider_and_Service_2018_data.csv")

# Rename ZIP.Code column as it is ZIP.Code in hospital data
colnames(medicare_data)[7] <- "ZIP.Code"

# Merging datasets on ZIP.Code
merged_data <- inner_join(medicare_data, hospital_data, "ZIP.Code")

# Identifying missing values
miss_val <- sum(is.na(merged_data))

# Calculating correlation matrix for numeric features to identify highly variable and related features
numeric_data <- merged_data %>% select_if(is.numeric)
correlation_matrix <- cor(numeric_data, use = "pairwise.complete.obs")


# Creating a heatmap of correlation matric
melted_corr <- melt(correlation_matrix)
ggplot(data = melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", limit = c(-1, 1), name = "Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Correlation Heatmap") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3)

#Collecting only categorical variables
categorical_vars <- names(merged_data)[sapply(merged_data, function(x) is.factor(x) || is.character(x))]
print(categorical_vars)

#Selecting only categorical variables to perform anova test for selecting highly variable feature to predict
anova_result <- aov(Avg_Mdcr_Pymt_Amt ~  Hospital.Type  + Hospital.Ownership+ State + Hospital.overall.rating +Emergency.Services+Meets.criteria.for.meaningful.use.of.EHRs+ Mortality.national.comparison+Patient.experience.national.comparison+Timeliness.of.care.national.comparison+Efficient.use.of.medical.imaging.national.comparison, data = merged_data)
summary(anova_result)


# Selecting Features based on Correlation Matrix and anova result
selected_vars <- c("Avg_Mdcr_Pymt_Amt", "Avg_Tot_Pymt_Amt", "Hospital.Type", "Hospital.Ownership", 
                   "Hospital.overall.rating", "Meets.criteria.for.meaningful.use.of.EHRs", 
                   "Mortality.national.comparison", "Patient.experience.national.comparison", 
                   "Timeliness.of.care.national.comparison", "Efficient.use.of.medical.imaging.national.comparison")


#Filtering dataset for HeartFailure DRG_Cd Code
final_data_HF <- filter(merged_data, DRG_Cd %in% c("291", "292", "293"))
selected_data_HF <- final_data_HF %>% select(all_of(selected_vars))

#Converting categorical variables to factor to pass in model- Used as optional for one-hot encoding
variables_to_factor <- c("Hospital.Type", "Hospital.Ownership", "Hospital.overall.rating", 
                         "Meets.criteria.for.meaningful.use.of.EHRs", "Mortality.national.comparison", 
                         "Patient.experience.national.comparison", 
                         "Timeliness.of.care.national.comparison", 
                         "Efficient.use.of.medical.imaging.national.comparison")

selected_data_HF[variables_to_factor] <- lapply(selected_data_HF[variables_to_factor], as.factor)


# Train-test split
set.seed(123)
train_index <- createDataPartition(selected_data_HF$Avg_Mdcr_Pymt_Amt, p = 0.8, list = FALSE)
train_data <- selected_data_HF[train_index, ]
test_data <- selected_data_HF[-train_index, ]

# Preprocessing data
pre_process <- preProcess(train_data %>% select(-Avg_Mdcr_Pymt_Amt), method = c("center", "scale"))
train_data_normalized <- predict(pre_process, newdata = train_data %>% select(-Avg_Mdcr_Pymt_Amt))
test_data_normalized <- predict(pre_process, newdata = test_data %>% select(-Avg_Mdcr_Pymt_Amt))

# Adding the target variable back
train_data_normalized$Avg_Mdcr_Pymt_Amt <- train_data$Avg_Mdcr_Pymt_Amt
test_data_normalized$Avg_Mdcr_Pymt_Amt <- test_data$Avg_Mdcr_Pymt_Amt

# Creating linear model with interaction and polynomial features for capturing non-linearity and better model complexity
linear_model_cv <- train(Avg_Mdcr_Pymt_Amt ~ . + I(Avg_Tot_Pymt_Amt^2) + 
                           Avg_Tot_Pymt_Amt:Hospital.Type + 
                           Avg_Tot_Pymt_Amt:Hospital.Ownership, 
                         data = train_data_normalized, method = "lm", 
                         trControl = trainControl(method = "repeatedcv", number = 10,repeats = 3))

predictions <- predict(linear_model_cv, newdata = test_data_normalized)

# Calculating evaluation metrics
train_predictions <- predict(linear_model_cv, newdata = train_data_normalized)
train_rmse <- RMSE(train_predictions, train_data_normalized$Avg_Mdcr_Pymt_Amt)
test_rmse <- RMSE(predictions, test_data_normalized$Avg_Mdcr_Pymt_Amt)
train_R2 <- R2(train_predictions, train_data_normalized$Avg_Mdcr_Pymt_Amt)
test_R2 <- R2(predictions, test_data_normalized$Avg_Mdcr_Pymt_Amt)


# Creating Dataframe of the result
evaluation_results <- data.frame(
  RMSE = c(train_rmse, test_rmse),
  R2 = c(train_R2, test_R2),
  row.names = c("Train", "Test")
)
print(evaluation_results)


# Creating a scatter plot of actual vs. predicted values
results_df <- data.frame(
  Actual = test_data$Avg_Mdcr_Pymt_Amt,
  Predicted = predictions
)

ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.5) + 
  geom_abline(slope = 1, intercept = 0, color = "red") +  
  labs(title = "Actual vs Predicted Values",
       x = "Actual Values",
       y = "Predicted Values") +
  theme_minimal() +
  xlim(0, max(results_df$Actual) * 1.1) +  
  ylim(0, max(results_df$Predicted) * 1.1)



#Lasso Model to check overfitting or possible mutiple colinearity
lasso_model <- train(Avg_Mdcr_Pymt_Amt ~ ., data = train_data_normalized, 
                     method = "glmnet", 
                     trControl = trainControl(method = "cv", number = 10), 
                     tuneGrid = expand.grid(alpha = 1, lambda = seq(0.01, 1, length = 10))) 
print(lasso_model)
