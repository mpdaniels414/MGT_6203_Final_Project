 
rm(list=ls())

#install required packages
#install.packages("randomForest")
#install.packages('rpart.plot')
#install.packages('ggstatsplot')

# load library
library(randomForest)
library(tidyverse)
library(readr)
library(DBI)
#library(RSQLite)
library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(tree)
library(pROC)
library(rpart.plot)
library(outliers)
library(tidyr)
library(purrr)
library(zoo)


# load dataset
df <- read_csv('2024-04-10 - pared.csv')
summary(df)

# create a copy of original dataset
df1 <- df
# to list if any missing values in each column present
sapply(df1, function(x) sum(is.na(x)))
# based on output, we know EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3 has 64 missing values.
# fill missing values columns with mean values
num <- sapply(df1,is.numeric)
df1[num] <- lapply(df1[num], na.aggregate)
# check if there are still NA values existing in dataset
sum(is.na(df1))
# determine any outliers existing in this dataset
# use logistic regression model to see which variables at significant level alpha = 0.01
log_mod <- glm(TARGET~., data = df1, family = "binomial")
summary(log_mod)

# determine outliers for insignificant numeric columns
num_col <- c('CNT_CHILDREN','AMT_INCOME_TOTAL', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
             'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION',
             'LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY','OBS_30_CNT_SOCIAL_CIRCLE',
             'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','AMT_REQ_CREDIT_BUREAU_HOUR',
             'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
             'total_amt_credit_sum', 'total_amt_credit_sum_debt','CREDIT_MISSING')


# identify outliers
outlier1 <- boxplot(df1$CNT_CHILDREN, plot = FALSE)$out
outlier2 <- boxplot(df1$AMT_INCOME_TOTAL, plot = FALSE)$out
outlier3 <- boxplot(df1$REGION_POPULATION_RELATIVE, plot = FALSE)$out
outlier4 <- boxplot(df1$DAYS_BIRTH, plot = FALSE)$out  # no outliers
outlier5 <- boxplot(df1$HOUR_APPR_PROCESS_START, plot = FALSE)$out
outlier6 <- boxplot(df1$REG_REGION_NOT_LIVE_REGION, plot = FALSE)$out
outlier7 <- boxplot(df1$REG_REGION_NOT_WORK_REGION, plot = FALSE)$out 
outlier8 <- boxplot(df1$LIVE_REGION_NOT_WORK_REGION, plot = FALSE)$out # no outliers
outlier9 <- boxplot(df1$REG_CITY_NOT_WORK_CITY, plot = FALSE)$out 
outlier10 <- boxplot(df1$LIVE_CITY_NOT_WORK_CITY, plot = FALSE)$out
outlier11 <- boxplot(df1$OBS_30_CNT_SOCIAL_CIRCLE, plot = FALSE)$out 
outlier12 <- boxplot(df1$OBS_60_CNT_SOCIAL_CIRCLE, plot = FALSE)$out # no outliers
outlier13 <- boxplot(df1$DEF_60_CNT_SOCIAL_CIRCLE, plot = FALSE)$out
outlier14 <- boxplot(df1$AMT_REQ_CREDIT_BUREAU_HOUR, plot = FALSE)$out
outlier15 <- boxplot(df1$AMT_REQ_CREDIT_BUREAU_DAY, plot = FALSE)$out
outlier16 <- boxplot(df1$AMT_REQ_CREDIT_BUREAU_WEEK, plot = FALSE)$out
outlier17 <- boxplot(df1$AMT_REQ_CREDIT_BUREAU_MON, plot = FALSE)$out
outlier18 <- boxplot(df1$total_amt_credit_sum, plot = FALSE)$out
outlier19 <- boxplot(df1$total_amt_credit_sum_debt, plot = FALSE)$out
outlier20 <- boxplot(df1$CREDIT_MISSING, plot = FALSE)$out


# remove outliers from our dataset
df1 <- df1[-which(df1$CNT_CHILDREN %in% outlier1),]
df1 <- df1[-which(df1$AMT_INCOME_TOTAL %in% outlier2),]
df1 <- df1[-which(df1$REGION_POPULATION_RELATIVE %in% outlier3),]
df1 <- df1[-which(df1$HOUR_APPR_PROCESS_START %in% outlier5),]
df1 <- df1[-which(df1$REG_REGION_NOT_LIVE_REGION %in% outlier6),]
df1 <- df1[-which(df1$REG_REGION_NOT_WORK_REGION %in% outlier7),]
df1 <- df1[-which(df1$REG_CITY_NOT_WORK_CITY %in% outlier9),]
df1 <- df1[-which(df1$LIVE_CITY_NOT_WORK_CITY %in% outlier10),]
df1 <- df1[-which(df1$OBS_30_CNT_SOCIAL_CIRCLE %in% outlier11),]
df1 <- df1[-which(df1$DEF_60_CNT_SOCIAL_CIRCLE %in% outlier13),]
df1 <- df1[-which(df1$AMT_REQ_CREDIT_BUREAU_HOUR %in% outlier14),]
df1 <- df1[-which(df1$AMT_REQ_CREDIT_BUREAU_DAY %in% outlier15),]
df1 <- df1[-which(df1$AMT_REQ_CREDIT_BUREAU_WEEK %in% outlier16),]
df1 <- df1[-which(df1$AMT_REQ_CREDIT_BUREAU_MON %in% outlier17),]
df1 <- df1[-which(df1$total_amt_credit_sum %in% outlier18),]
df1 <- df1[-which(df1$total_amt_credit_sum_debt %in% outlier19),]
df1 <- df1[-which(df1$CREDIT_MISSING %in% outlier20),]

# check sample size after remove outliers
nrow(df1)

# split data into training and test sets
set.seed(123)
# training will be 70% and test will be 30%
trainIndex <- createDataPartition(df1$TARGET, p = 0.7, list = FALSE, times = 1)
TRAIN <- df1[trainIndex, ]
TEST <- df1[-trainIndex, ]

# train randomForest model
#target.rf <- randomForest(TARGET~., data = TRAIN, ntree = 500, mtry = 3, keep.forest = TRUE, importance = TRUE)
#target.rf

# plot the test MSE by number of trees
#plot(target.rf)

# use turnRF() for searching best optimal mtry values given for our data
#set.seed(123)
#bestmtry <- tuneRF(TRAIN, TRAIN$TARGET, stepFactor = 1.2, improve = 0.01, ntree=500)
#bestmtry

# train the random forest regression model with optimal value
target.rf_final <- randomForest(TARGET~., data = TRAIN, ntree = 500, mtry = 31, 
                            keep.forest = TRUE, importance = TRUE)
target.rf_final

# extract tree information
str(target.rf_final, 1)
getTree(target.rf_final, k = 1, labelVar = FALSE)

# feature importances
importance(target.rf_final)
varImpPlot(target.rf_final)

# search cutoff value with highest accuracy
df2 <- TEST %>%
  mutate(pred_prob = predict(target.rf_final, newdata = TEST, type = "response")) %>%
  mutate(pred_outcome_0.2 = ifelse(pred_prob >= 0.2, 1, 0)) %>%
  mutate(pred_outcome_0.3 = ifelse(pred_prob >= 0.3, 1, 0)) %>%
  mutate(pred_outcome_0.4 = ifelse(pred_prob >= 0.4, 1, 0)) %>%
  mutate(pred_outcome_0.5 = ifelse(pred_prob >= 0.5, 1, 0)) %>%
  mutate(pred_outcome_0.6 = ifelse(pred_prob >= 0.6, 1, 0)) %>%
  mutate(pred_outcome_0.7 = ifelse(pred_prob >= 0.7, 1, 0)) %>%
  mutate(pred_outcome_0.8 = ifelse(pred_prob >= 0.8, 1, 0))

# create vectors and tables for confusion matrix
expected_value <- factor(df2$TARGET)
predicted_value_0.2 <- factor(df2$pred_outcome_0.2)
predicted_value_0.3 <- factor(df2$pred_outcome_0.3)
predicted_value_0.4 <- factor(df2$pred_outcome_0.4)
predicted_value_0.5 <- factor(df2$pred_outcome_0.5)
predicted_value_0.6 <- factor(df2$pred_outcome_0.6)
predicted_value_0.7 <- factor(df2$pred_outcome_0.7)

# create confusion matrix
result_0.2 <- confusionMatrix(predicted_value_0.2, expected_value)
result_0.2
result_0.3 <- confusionMatrix(predicted_value_0.3, expected_value)
result_0.3
result_0.4 <- confusionMatrix(predicted_value_0.4, expected_value)
result_0.4
result_0.5 <- confusionMatrix(predicted_value_0.5, expected_value)
result_0.5
result_0.6 <- confusionMatrix(predicted_value_0.6, expected_value)
result_0.6
result_0.7 <- confusionMatrix(predicted_value_0.7, expected_value)
result_0.7

# determine the accuracy, specificity and sensitivity from the results above
accuracy <- c(result_0.2$overall['Accuracy'],result_0.3$overall['Accuracy'],
              result_0.4$overall['Accuracy'],result_0.5$overall['Accuracy'],
              result_0.6$overall['Accuracy'],result_0.7$overall['Accuracy'])

plot(accuracy)
which.max(accuracy) # the cutoff value 0.5 has the highest accuracy
round(max(accuracy), 3)  # highest accuracy is 0.925


# analyze cost

df2_pred_test <- predict(target.rf_final, newdata = TEST, type = 'response')

threshold <- seq(0.025, 1, by = 0.025)
cost <- numeric(length(threshold))
names(cost) <- as.character(threshold)

cost_fp <- 10000 # cost of a false positive
cost_fn <- cost_fp*5 # cost of a false negative equals to 5 times of false positive


for (i in threshold) {
  target_test <- ifelse(df2_pred_test>i, 1, 0)
  result <- table(pred_value = target_test, exp_value = df2$TARGET)
  
  # calculate false positives and false negatives values
  fp <- ifelse('1' %in% rownames(result) & '0' %in% colnames(result), 
               result['1', '0'], 0)
  fn <- ifelse('0' %in% rownames(result) & '1' %in% colnames(result), 
               result['0', '1'], 0)
  
  # total cost
  total_cost <- (cost_fp * fp) + (cost_fn * fn)
  cost[as.character(i)] <- total_cost
}

# determine the best threshold value
best_threshold_index <- which.min(cost)
best_threshold_index
best_threshold <- threshold[best_threshold_index]
best_threshold

# visualization
plot(threshold, cost)

# we now can use the best threshold value to make predictions
target_test1 <- factor(ifelse(df2_pred_test >= best_threshold, 1, 0))
str(target_test1)

cost_final <- confusionMatrix(target_test1, expected_value)
cost_final
model_accuracy <- cost_final$overall['Accuracy']
model_accuracy
model_sensitivity <- cost_final$byClass['Sensitivity']
model_sensitivity
model_specificity <- cost_final$byClass['Specificity']
model_specificity

# plot ROC curve
df3 <- TEST %>%
  mutate(pred_prob = predict(target.rf_final, newdata = ., type = "response")) %>%
  mutate(pred_outcome = ifelse(pred_prob >= best_threshold,1,0))

roc_cur <- roc(TARGET~pred_prob, data = df3, plot = TRUE, print.auc = TRUE)

