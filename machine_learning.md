---
title: 'Human Activity Recognition'
author: 'Kevin Wu'
date: 'November 22, 2015'
output: html_document
---

## Executive Summary

Six participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions, or classes. Using accelerometer data from the belt, forearm, arm, and dumbell of the participants, we can build a machine learning algorithm to predict which of the five classes a subject is performing.

## Import and Clean Data

Load packages and read in the training and test data.


```r
library(caret)
library(randomForest)

## Read data
pml_training <- read.csv('pml-training.csv')
pml_testing <- read.csv('pml-testing.csv')
```

Clean the data by removing the variables with mostly blank or missing values. Then, remove the factor variables from the available predictors.

Split the clean training data into training and validation datasets.


```r
## Remove blank, missing, and factor variables
predictors <- colSums(is.na(pml_training) |
                          pml_training == '') / nrow(pml_training) < .9
predictors[c(1:2, 5:6)] <- FALSE

## Split training data
set.seed(383)
train <- createDataPartition(y = pml_training$classe, p = 0.7, list = FALSE)
training <- pml_training[train, predictors]
validation <- pml_training[-train, predictors]
```

The datasets have 55 predictors.

## Machine Learning Algorithm

Use the training dataset to build a random forest prediction model with activity class as the outcome.


```r
## Random forest model
fit <- randomForest(classe ~ ., training)
```

Predict activity class using the validation dataset.


```r
## Predict with validation predictors
prediction <- predict(fit, validation[, -56])
```

## Out of Sample Error

With the predicted values from the validation dataset, calculate a confusion matrix to determine the model's accuracy and estimate the out of sample error.


```r
## Confusion matrix to calculate accuracy and error
validation_matrix <- confusionMatrix(validation$classe, prediction)
accuracy <- validation_matrix$overall[[1]]
oos_error <- 1 - accuracy
```

Accuracy is: 99.93%

Out of sample error is: 0.07%

## Test Cases

We can now apply our prediction model to the test cases.


```r
## Predict with test data
testing <- pml_testing[, predictors]
test_prediction <- as.character(predict(fit, testing[, -56]))
print(test_prediction)
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```
