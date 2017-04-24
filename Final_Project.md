Final Project
================
Jose Rivas
22 de abril de 2017

``` r
library(caret)
library(rpart)
library(plyr)
library(randomForest)
library(gbm)
library(ipred)
library(e1071)
```

Getting Data
------------

The following code reads Training and Testing datasets into R, and display a view of their structure.

``` r
#Read datasets into R
testing <- read.csv("C:/Users/jose_/Documents/Data Science/Machine Learning/pml-testing.csv", na.strings = c("NA", ""))
training <- read.csv("C:/Users/jose_/Documents/Data Science/Machine Learning/pml-training.csv", na.strings = c("NA", ""))
#display Training structure
str(training, list.len=25)
```

    ## 'data.frame':    19622 obs. of  160 variables:
    ##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
    ##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
    ##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
    ##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
    ##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
    ##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ kurtosis_roll_belt      : Factor w/ 396 levels "-0.016850","-0.021024",..: NA NA NA NA NA NA NA NA NA NA ...
    ##  $ kurtosis_picth_belt     : Factor w/ 316 levels "-0.021887","-0.060755",..: NA NA NA NA NA NA NA NA NA NA ...
    ##  $ kurtosis_yaw_belt       : Factor w/ 1 level "#DIV/0!": NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_roll_belt      : Factor w/ 394 levels "-0.003095","-0.010002",..: NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_roll_belt.1    : Factor w/ 337 levels "-0.005928","-0.005960",..: NA NA NA NA NA NA NA NA NA NA ...
    ##  $ skewness_yaw_belt       : Factor w/ 1 level "#DIV/0!": NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_picth_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ max_yaw_belt            : Factor w/ 67 levels "-0.1","-0.2",..: NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_roll_belt           : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_pitch_belt          : int  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ min_yaw_belt            : Factor w/ 67 levels "-0.1","-0.2",..: NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_roll_belt     : num  NA NA NA NA NA NA NA NA NA NA ...
    ##  $ amplitude_pitch_belt    : int  NA NA NA NA NA NA NA NA NA NA ...
    ##   [list output truncated]

Cleaning Data
-------------

There are some variables that have NA or \#DIV/O! in some of their rows. As they cannot be used for prediction, they must be deleted. Some other variables as "user\_name" cannot be used as predictors as they are not related to the variable to predict. First seven variables will then be deleted as well.

``` r
#Eliminate columns with NA
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
#Eliminate first seven columns
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
nrow(training); nrow(testing)
```

    ## [1] 19622

    ## [1] 20

Create Partition of Training dataset into Training and Validating
-----------------------------------------------------------------

The size of Training and Testing datasets is highly unbalanced. We need to create a new Validating dataset with a partition of Training to reduce the probability of out-of-sample deviations. The chosen percentage was 70% Training, 30% Validating

``` r
#Set seed and create Train and Valid datasets from Training
set.seed(2882) 
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
train <- training[inTrain, ]
valid <- training[-inTrain, ]
test <- testing
```

Create Predictive Models
------------------------

To make a selection of the best predictive model, I am going to use three different models: Trees, Random Forest and Boosting. And then, compare their resulting accuracies on predictions for the Validation Dataset.

The computing timing of some of these models make them unfeasible. To arrange this, we're going to create a control variable with some parameter to pass to the trControl parameter on the Train Function: method = CV (to define cross validation as the method for resampling) number = 5 (limit the number of resampling iterations to 5)

``` r
#Trcontrol parameters
control <- trainControl(method="cv", number = 5)
#Trees Model
mod_trees <- train(classe~., method="rpart", data=train, trControl=control)
pred_trees <- predict(mod_trees, valid)
confusionMatrix(valid$classe, pred_trees)$overall[1]
```

    ##  Accuracy 
    ## 0.5004248

``` r
#Random Forest Model
mod_rf <- train(classe~., method="rf", data=train, trControl=control)
pred_rf <- predict(mod_rf, valid)
confusionMatrix(valid$classe, pred_rf)$overall[1]
```

    ##  Accuracy 
    ## 0.9932031

``` r
#Boosting Model
mod_boost <- train(classe~., method="gbm", data=train, verbose=FALSE, trControl=control)
pred_boost <- predict(mod_boost, valid)
confusionMatrix(valid$classe, pred_boost)$overall[1]
```

    ##  Accuracy 
    ## 0.9643161

Select the best models
----------------------

The best models by accuracy are Random Forest (0.99) and Boosting (0.96). They both seem to perform pretty well, but this high values may be indicating data overfitting. I´m going to compare matrixes for both predictions

``` r
#Compare predictions on Validating dataset
confusionMatrix(valid$classe, pred_rf)$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1670    2    2    0    0
    ##          B    9 1127    3    0    0
    ##          C    0    7 1018    1    0
    ##          D    0    1   13  950    0
    ##          E    0    0    2    0 1080

``` r
confusionMatrix(valid$classe, pred_boost)$table
```

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1649   13   10    1    1
    ##          B   25 1078   34    1    1
    ##          C    0   38  977    9    2
    ##          D    0    4   30  924    6
    ##          E    2   14    6   13 1047

Ensemble Model
--------------

To check if there is a possible improvement in the prediction, I´m going to ensemble a new model combining the best two predictions: Random Forest and Boosting

``` r
#Create dataset with best two predictions
df_comb <- data.frame(pred_rf, pred_boost, classe=valid$classe)
#ensemble them using the Treebag model
mod_comb <- train(classe~., method="treebag", data=df_comb)
pred_comb <- predict(mod_comb, df_comb)
confusionMatrix(valid$classe, pred_comb)$overall[1]
```

    ##  Accuracy 
    ## 0.9932031

The accuracy of this ensembled model is exactly the same as the Random Forest, so it seems this model is prevailing after all, and it offers the most accurate prediction.

Make predictions for Testing Dataset
------------------------------------

Using the Random Forest model on the Testing datasets gives us the final predictions.

``` r
#Use random Forest model to predict Testing
pred_final <- predict(mod_rf, test)
pred_final
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
