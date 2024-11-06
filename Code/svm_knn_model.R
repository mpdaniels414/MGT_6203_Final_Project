# Clear environment
rm(list = ls())

# Load the necessary libraries
library(kernlab)
library(caret)
library(dplyr)
library(outliers)
library(httr)
library(readr)


# ---------------------------- Data manipulation -------------------------------------

# load dataset into R and omit any rows with NA values. For SVM model, predict() 
# does not work if any rows contain NA values (We also need to reduce
# the size of the dataset anyways for performance purposes).

# URL of the Dropbox file
file_url <- "https://www.dropbox.com/scl/fo/oxy3yrx9xkgfme82y3p3u/h/Data/2024-04-10%20-%20pared.csv?dl=1&rlkey=j1dizwta8ie429w70dpg6g1bp"

# Destination where the file will be saved
destination_file <- "2024-04-10 - pared.csv"

# Download the file
GET(url = file_url, write_disk(destination_file, overwrite = TRUE))

# Read the downloaded file (optional, to verify it's downloaded correctly)
data <- read_csv(destination_file)
data = na.omit(data)

# Setting the random number generator seed so that our results are reproducible
set.seed(123)


# use logistic regression model to filter variables to most significant
model <- glm( TARGET ~ ., data, family = 'binomial')

#Summary of the model. use this to find most viable variables
options(max.print=20000)
summary(model)

# filter data for relevant columns with low p-value. 
keep_cols = c(1,2,4,5,7,8,9,10,17,18,19,33,34,35,46,47,49)

# create "filtered" dataset, keeping only columns deemed significant
filtered_data = data[,keep_cols]

# create dummy variables for categorical data. For SVM models, we create a 
# categorical variable for each unique value
filtered_data$NAME_CONTRACT_TYPE_CL = ifelse(filtered_data$NAME_CONTRACT_TYPE == 'Cash loans', 1, 0)
filtered_data$NAME_CONTRACT_TYPE_RL = ifelse(filtered_data$NAME_CONTRACT_TYPE == 'Revolving loans', 1, 0)
filtered_data$NAME_CONTRACT_TYPE = NULL

filtered_data$FLAG_OWN_CAR_Y = ifelse(filtered_data$FLAG_OWN_CAR == 'Y', 1, 0)
filtered_data$FLAG_OWN_CAR_N = ifelse(filtered_data$FLAG_OWN_CAR == 'N', 1, 0)
filtered_data$FLAG_OWN_CAR = NULL

filtered_data$FLAG_OWN_REALTY_Y = ifelse(filtered_data$FLAG_OWN_REALTY == 'Y', 1, 0)
filtered_data$FLAG_OWN_REALTY_N = ifelse(filtered_data$FLAG_OWN_REALTY == 'N', 1, 0)
filtered_data$FLAG_OWN_REALTY = NULL

filtered_data$INCOME_CAT_Upper = ifelse(filtered_data$INCOME_CAT == 'Upper', 1, 0)
filtered_data$INCOME_CAT_Upper_middle = ifelse(filtered_data$INCOME_CAT == 'Upper-middle', 1, 0)
filtered_data$INCOME_CAT_Middle = ifelse(filtered_data$INCOME_CAT == 'Middle', 1, 0)
filtered_data$INCOME_CAT_Lower_middle = ifelse(filtered_data$INCOME_CAT == 'Lower-middle', 1, 0)
filtered_data$INCOME_CAT_Lower = ifelse(filtered_data$INCOME_CAT == 'Lower', 1, 0)
filtered_data$INCOME_CAT = NULL

# We'll sample 5000 data points with a TARGET value of 0, and 5000 with a TARGET 
# value of 1.
zeros = filter(filtered_data, filtered_data$TARGET == 0)
ones = filter(filtered_data, filtered_data$TARGET == 1)

mask_zeros = sample(nrow(zeros), size = 5000)
zeros = zeros[mask_zeros,]
mask_ones = sample(nrow(ones), size = 5000)
ones = ones[mask_ones,]

# combine zeros and ones dataset. Now we have an equal number of target values
# for training and testing
reduced_data = rbind.data.frame(ones,zeros)

# investigate possible outliers and remove if necessary to improve accuracy of model

# boxplot(reduced_data$AMT_CREDIT)
# reduced_data[order(reduced_data$AMT_CREDIT, decreasing = TRUE),]$AMT_CREDIT[1:10]
# reduced_data = filter(reduced_data, reduced_data$AMT_CREDIT < 3860019 )

boxplot(reduced_data$AMT_ANNUITY)
reduced_data[order(reduced_data$AMT_ANNUITY, decreasing = TRUE),]$AMT_ANNUITY[1:10]
reduced_data = filter(reduced_data, reduced_data$AMT_ANNUITY < 149211.0)

# boxplot(reduced_data$AMT_GOODS_PRICE)
# unique(reduced_data[order(reduced_data$AMT_GOODS_PRICE, decreasing = TRUE),]$AMT_GOODS_PRICE)[1:10]
# reduced_data = filter(reduced_data, reduced_data$AMT_GOODS_PRICE < 2250000 )

boxplot(reduced_data$DAYS_BIRTH)

boxplot(reduced_data$DAYS_EMPLOYED)
unique(reduced_data[order(reduced_data$DAYS_EMPLOYED, decreasing = TRUE),]$DAYS_EMPLOYED)[1:10]
reduced_data = filter(reduced_data, reduced_data$DAYS_EMPLOYED < 365243)

boxplot(reduced_data$total_amt_credit_sum)
unique(reduced_data[order(reduced_data$total_amt_credit_sum, decreasing = TRUE),]$total_amt_credit_sum)[1:10]
reduced_data = filter(reduced_data, reduced_data$total_amt_credit_sum < 28163669)

boxplot(reduced_data$total_amt_credit_sum_debt)
unique(reduced_data[order(reduced_data$total_amt_credit_sum_debt, decreasing = TRUE),]$total_amt_credit_sum_debt)[1:10]
reduced_data = filter(reduced_data, reduced_data$total_amt_credit_sum < 27251439 )

# create training dataset
mask_train = sample(nrow(reduced_data), size = floor(nrow(reduced_data) * 0.7))
train = reduced_data[mask_train,]

# Using the remaining data for test data
test = reduced_data[-mask_train, ]  




# --------------- Test out different C-values for SVM model. Using a linear kernel for performance purposes -------------------

target = test$TARGET

# values of C to test
amounts <- c(0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000) 

# vectors to store stats for of each C value
Accuracy = c()
Sensitivity = c()
Specificity = c()
for (i in 1:9) {
  
  # fit model using training set
  model <- ksvm(as.matrix(train[,2:ncol(train)]),
                train[,1],
                type = "C-svc", # Use C-classification method
                kernel = "vanilladot", # Use simple linear kernel
                C = amounts[i],
                scaled=TRUE) 
  
  # make prediction on test set
  pred <- predict(model,test[,2:ncol(test)],type='response')
  results = data.frame(target, pred)
  cm = confusionMatrix(as.factor(results$pred), 
                       as.factor(results$target),
                       positive = '1')
  
  # store stats into vectors
  Accuracy = append(Accuracy, cm$overall['Accuracy'][1])
  Sensitivity = append(Sensitivity, cm$byClass['Sensitivity'])
  Specificity = append(Specificity, cm$byClass['Specificity'])
  
}

# create dataframe of stats based on cutoff values
results = data.frame(amounts, Accuracy, Sensitivity, Specificity)
results
# amounts  Accuracy Sensitivity Specificity
# 1   1e-05 0.5126508   1.0000000   0.0000000
# 2   1e-04 0.6418840   0.8580106   0.4145367
# 3   1e-03 0.6683534   0.7000759   0.6349840
# 4   1e-02 0.6722460   0.6977980   0.6453674
# 5   1e-01 0.6773063   0.7038724   0.6493610
# 6   1e+00 0.6784741   0.7031131   0.6525559
# 7   1e+01 0.6792526   0.7023538   0.6549521
# 8   1e+02 0.6780849   0.7069096   0.6477636
# 9   1e+03 0.6687427   0.7479119   0.5854633

ggplot(results, aes(x=amounts)) + 
  geom_line(aes(y = Accuracy, color = 'Accuracy')) + 
  geom_line(aes(y = Sensitivity, color = 'Sensitivity')) + 
  geom_line(aes(y = Specificity, color = 'Specificity')) 
  

## While SVM model 7 appears to be most accurate, the most useful model for detecting loan default appears 
## to be model 2 (relatively high sensitivity and accuracy), and it's only around 4% less accurate. 

# retrain the best model (since I've overwritten it above)
model <- ksvm(as.matrix(train[,2:ncol(train)]),
                     train[,1],
                     type = "C-svc", # Use C-classification method
                     kernel = "vanilladot", # Use simple linear kernel
                     C = amounts[2],
                     scaled=TRUE,
                     ) # have ksvm scale the data for you


pred <- predict(model,test[,2:ncol(test)],type='response')
results = data.frame(target, pred)
cm = confusionMatrix(as.factor(results$pred), 
                     as.factor(results$target),
                     positive = '1')
cm

# Confusion Matrix and Statistics
# 
#             Reference
# Prediction    0    1
#           0  519  187
#           1  733 1130
# 
# Accuracy : 0.6419         
# 95% CI : (0.623, 0.6604)
# No Information Rate : 0.5127         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.2755         
# 
# Mcnemar's Test P-Value : < 2.2e-16      
#                                          
#             Sensitivity : 0.8580         
#             Specificity : 0.4145         
#          Pos Pred Value : 0.6065         
#          Neg Pred Value : 0.7351         
#              Prevalence : 0.5127         
#          Detection Rate : 0.4399         
#    Detection Prevalence : 0.7252         
#       Balanced Accuracy : 0.6363         
#                                          
#        'Positive' Class : 1              








### Let's try modeling the target variable on each individual factor and store
### the accuracy of each factor in a vector

# vector to store stats of each C value
Accuracy = c()
Sensitivity = c()
Specificity = c()

# loop through each variable, create an SVM model against the target variable,
# and store the stats in a vector.
for (i in 2:ncol(train)) {
  
  # fit model using training set
  model <- ksvm(as.matrix(train[,i]),
                train[,1],
                type = "C-svc", # Use C-classification method
                kernel = "vanilladot", # Use simple linear kernel
                C = amounts[2],
                scaled=TRUE) 
  
  # make prediction on test set
  pred <- predict(model,test[,i],type='response')
  results = data.frame(target, pred)
  cm = confusionMatrix(as.factor(results$pred), 
                       as.factor(results$target),
                       positive = '1')
  
  # store statistical data
  Accuracy = append(Accuracy, cm$overall['Accuracy'][1])
  Sensitivity = append(Sensitivity, cm$byClass['Sensitivity'])
  Specificity = append(Specificity, cm$byClass['Specificity'])
}

results = data.frame(Accuracy, Sensitivity, Specificity)
results
## This didn't appear to yield significantly better results than training and testing on the 
## 'full' dataset






### Try a KNN Model ###


library(kknn)

#
# --------------- Train KNN models -------------------
#
Accuracy = c()
Sensitivity = c()
Specificity = c()

# set maximum value of k (number of neighbors) to test
k_values = 1:100
for (k in k_values) {
  
  # fit k-nearest-neighbor model using training set, validate on test set
  knn_model <- kknn(TARGET~.,train,test,k=k,scale=TRUE)
  cm = confusionMatrix(as.factor(round(knn_model$fitted.values)), 
                       as.factor(test$TARGET),
                       positive = '1')
  Accuracy = append(Accuracy, cm$overall['Accuracy'][1])
  Sensitivity = append(Sensitivity, cm$byClass['Sensitivity'])
  Specificity = append(Specificity, cm$byClass['Specificity'])
}  
results = data.frame(Accuracy, Sensitivity, Specificity)
results

ggplot(results, aes(x=k_values)) + 
  geom_line(aes(y = Accuracy, color = 'Accuracy')) + 
  geom_line(aes(y = Sensitivity, color = 'Sensitivity')) + 
  geom_line(aes(y = Specificity, color = 'Specificity'))


# 75 appears to be an optimal k-value for obtaining a high accuracy 
# and sensitivity, without overfitting the model. 
knn_model <- kknn(TARGET~.,train,test,
                  k=75,
                  scale=TRUE)


# calculate prediction qualities
pred <- knn_model$fitted.values

# test different cutoff values using a k-value of 75
cutoff = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
Accuracy = c()
Sensitivity = c()
Specificity = c()
target_pred = data.frame(target, pred)

for (i in cutoff){
  copy = target_pred
  copy$pred = ifelse(copy$pred >= i, 1, 0)
  cm = confusionMatrix(as.factor(copy$pred), 
                       as.factor(copy$target),
                       positive = '1')
  Accuracy = append(Accuracy, cm$overall['Accuracy'][1])
  Sensitivity = append(Sensitivity, cm$byClass['Sensitivity'])
  Specificity = append(Specificity, cm$byClass['Specificity'])
}
results = data.frame(cutoff, Accuracy, Sensitivity, Specificity)
results
# cutoff  Accuracy Sensitivity Specificity
# 1    0.1 0.5126508  1.00000000  0.00000000
# 2    0.2 0.5192682  0.99696279  0.01677316
# 3    0.3 0.5780459  0.95747912  0.17891374
# 4    0.4 0.6379914  0.85573273  0.40894569
# 5    0.5 0.6652394  0.69779803  0.63099042
# 6    0.6 0.6469443  0.49734244  0.80431310
# 7    0.7 0.5963410  0.28929385  0.91932907
# 8    0.8 0.5406773  0.12528474  0.97763578
# 9    0.9 0.4892954  0.00531511  0.99840256

ggplot(results, aes(x=cutoff)) + 
  geom_line(aes(y = Accuracy, color = 'Accuracy')) + 
  geom_line(aes(y = Sensitivity, color = 'Sensitivity')) + 
  geom_line(aes(y = Specificity, color = 'Specificity'))

# Most optimal cutoff value for KNN model for accuracy and sensitivity appears to be 0.4.
copy = target_pred
copy$pred = ifelse(copy$pred >= 0.4, 1, 0)
cm = confusionMatrix(as.factor(copy$pred), 
                     as.factor(copy$target),
                     positive = '1')
cm

# Confusion Matrix and Statistics
# 
#             Reference
# Prediction    0    1
#           0  512  190
#           1  740 1127
# 
# Accuracy : 0.638           
# 95% CI : (0.6191, 0.6566)
# No Information Rate : 0.5127          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.2676          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.8557          
#             Specificity : 0.4089          
#          Pos Pred Value : 0.6036          
#          Neg Pred Value : 0.7293          
#              Prevalence : 0.5127          
#          Detection Rate : 0.4387          
#    Detection Prevalence : 0.7267          
#       Balanced Accuracy : 0.6323          
#                                           
#        'Positive' Class : 1               
