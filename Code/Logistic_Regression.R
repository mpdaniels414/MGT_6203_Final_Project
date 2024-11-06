rm(list=ls())
#library(forcats)
library(stringr)
library(dplyr)
library(tidyr)
library(lubridate)
#library(magrittr)
#library(tidyr)
#library(tibble)
library(ggplot2)
library(ggthemes)
#library(viridis)
#library(sf)
library(kernlab)
library(kknn)
library(caret)
library(class)
library(caTools)
library(lattice)
library(scales)
#library(splitTools)
library(factoextra)
library(cluster)
library(plotly)
# library(rgl)
library(scatterplot3d)
library(stringi)
library(GGally)
library(quantreg)
library(outliers)
library(tidyverse)
library(reshape2)
library(ggpubr)
library(forecast)
library(ggfortify)
#library(fpp)
library(timetk)
library(pander)
library(fBasics)
library(DAAG)
library(xtable)
library(MASS)
library(corrplot)
set.seed(123)

df = read.csv("2024-04-10 - pared.csv",header = TRUE, stringsAsFactors = T)

# #Keeping the needed columns
df1<-df%>%dplyr::select(TARGET,NAME_CONTRACT_TYPE,FLAG_OWN_CAR,
                        FLAG_OWN_REALTY,CNT_CHILDREN,AMT_INCOME_TOTAL,AMT_CREDIT,NAME_EDUCATION_TYPE,
                        NAME_HOUSING_TYPE,	REGION_POPULATION_RELATIVE,	OCCUPATION_TYPE,EXT_SOURCE_1,	EXT_SOURCE_2,	EXT_SOURCE_3,
                        total_amt_credit_sum,	total_amt_credit_sum_debt, INCOME_CAT)
summary(df1) 	
df1_melt <- melt(df1)
ggplot(df1_melt, aes(variable, value)) + geom_boxplot() + facet_wrap(~variable, scale="free")+labs(title="Box plot of variables")

out<-lapply(df1[,-1], function(x) boxplot.stats(x)$out)
out_ind  <- sapply(names(df1[,-1]), \(col) which(df1[[col]] %in% out[[col]])) 

#Grubb's test
# grubbs.test(df1[,5], type=10)

# df1<-df1%>%mutate(CNT_CHILDREN = ifelse(CNT_CHILDREN>4,4,CNT_CHILDREN))%>%
#   mutate(AMT_INCOME_TOTAL = ifelse(AMT_INCOME_TOTAL>10000000,10000000,AMT_INCOME_TOTAL))

# df1[, 1:19] <- sapply(df1[, 1:19], as.character)
# df1[, 1:19] <- sapply(df1[, 1:19], as.numeric)

#PCA
# pc<-prcomp(df3[,-1],center=T,scale=T)
# summary(pc)

# normalizing the data
df2<-preProcess(as.data.frame(df1), method=c("range"))
df3<-predict(df2, as.data.frame(df1))

# Splotting the data between train (70%) and test (30%)
df3_sort = sort(sample(nrow(df3), size=nrow(df3)*.7))
train <- df3[df3_sort,]
test <- df3[-df3_sort,]

#creating a logistic regression model
loan_model <- glm(TARGET ~., data = train, family = binomial(link = "logit"))
summary(loan_model) 

#making predictions
p=0.5
df3_predict<- test %>% 
  mutate(pred_prob = predict(loan_model, newdata = ., type = "response")) %>%
  mutate(pred_outcome = ifelse(pred_prob >= p,1,0))

#Creates vectors having data points
expected_value <- as.factor(df3_predict$TARGET)
predicted_value <- as.factor(df3_predict$pred_outcome)

#Creating confusion matrix
confusion <- confusionMatrix(data=predicted_value, reference = expected_value)
confusion
accuracy<-round(confusion$overall['Accuracy'],3)
accuracy
sensitivty<-round(confusion[["byClass"]][["Sensitivity"]],3)
specificity<-round(confusion[["byClass"]][["Specificity"]],3)

cat("Accuracy for cutoff probability value of", p, "is ",(accuracy)*100,"%", "\n")
cat("Sensitivty for cutoff probability value of", p, "is ",(sensitivty)*100, "%","\n")
cat("Specificity for cutoff probability value of", p, "is ",(specificity)*100,"%", "\n")

#making VIF plots
vif_data<-car::vif(loan_model)
vif_data
vif_names<-row.names(vif_data)
vif_data2 <- as.data.frame(vif_data) 

#cost analysis. Making the False negative 5 times more costly than false positive
cost_vector<-vector("numeric")

df3_predict_test<- predict(loan_model, newdata = test, type = "response")

m<-seq(0,1,0.025)
for (i in seq(0,1,0.025)) {
  loan_model_test_fact<-as.factor(ifelse(df3_predict_test>i,1,0))
  cm<-confusionMatrix(loan_model_test_fact,as.factor(df3_predict$TARGET))
  fp<-cm$table[2,1]
  fn<-cm$table[1,2]
  cost<-fn*5+fp*1
  cost_vector<-c(cost_vector,cost)
}

plot(x=seq(0,1,0.025), y=cost_vector, xlab="probability threshold", ylab="cost vector", main="Optimizing cost for different probability thresholds")

temp2<-min(cost_vector)
temp3<-which.min(cost_vector)
thd_optimized<-m[temp3]

accuracy_df <- data.frame(Threshold = seq(0,1,0.025), Accuracy = cost_vector)
ggplot(accuracy_df, aes(x = Threshold, y = Accuracy)) + 
  geom_line() + 
  geom_point() +
  geom_vline(xintercept = thd_optimized, linetype="dashed", color = "red") +
  theme_minimal() +
  labs(title = paste("Optimizing cost for different probability thresholds (Linear Regression)"), 
       x = "Threshold", y = "Cost")


cat("The optimized probability threshold is ", thd_optimized,"\n")
loan_model_test_fact<-as.factor(ifelse(df3_predict_test>thd_optimized,1,0))
cm_final<-confusionMatrix(loan_model_test_fact,as.factor(df3_predict$TARGET))
cm_final
b<-cm_final$byClass
c<-cm_final$overall

cat("The accuracy of the model (after cost optimizing) is ", c[1],"\n")
cat("The sensitivity of the model (after cost optimizing) is ", b[1],"\n")
cat("The specificity of the model (after cost optimizing) is ", b[2],"\n")

df3_predict<- test %>% 
  mutate(pred_prob = predict(loan_model, newdata = ., type = "response")) %>%
  mutate(pred_outcome = ifelse(pred_prob >= thd_optimized,1,0))

#Making ROC plot
library(pROC)
rocobj <- roc(TARGET ~ pred_prob, data = df3_predict, plot = TRUE, print.auc = TRUE)
