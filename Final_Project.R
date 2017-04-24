library(caret)
library(rpart)
library(randomForest)
library(gbm)
library(survival)
library(splines)
library(parallel)
library(fastAdaboost)

testing <- read.csv("C:/Users/jose_/Documents/Data Science/Machine Learning/pml-testing.csv", na.strings = c("NA", ""))
training <- read.csv("C:/Users/jose_/Documents/Data Science/Machine Learning/pml-training.csv", na.strings = c("NA", ""))

dim(training)

training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
nrow(training); nrow(testing)

set.seed(2882) 
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
train <- training[inTrain, ]
valid <- training[-inTrain, ]
test <- testing

control <- trainControl(method="cv", number = 5)

mod_trees <- train(classe~., method="rpart", data=train, trControl=control)
pred_trees <- predict(mod_trees, valid)
confusionMatrix(valid$classe, pred_trees)$overall[1]

mod_rf <- train(classe~., method="rf", data=train, trControl=control)
pred_rf <- predict(mod_rf, valid)
confusionMatrix(valid$classe, pred_rf)$overall[1]

mod_boost <- train(classe~., method="gbm", data=train, verbose=FALSE, trControl=control)
pred_boost <- predict(mod_boost, valid)
confusionMatrix(valid$classe, pred_boost)$overall[1]

confusionMatrix(valid$classe, pred_rf)$table
confusionMatrix(valid$classe, pred_boost)$table

df_comb <- data.frame(pred_rf, pred_boost, classe=valid$classe)
mod_comb <- train(classe~., method="treebag", data=df_comb)
pred_comb <- predict(mod_comb, df_comb)
confusionMatrix(valid$classe, pred_comb)$overall[1]

pred_final <- predict(mod_rf, test)
pred_final