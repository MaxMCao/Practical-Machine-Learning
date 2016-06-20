---
title: "Practical Machine Learning Course Project"
author: "Max Cao"
date: "20 June 2016"
output: html_document
---


```r
knitr::opts_chunk$set(echo = TRUE)
```
# Intoduction

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. This project aims to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants and predict the manner in which they did the exercise.

##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

##Data

The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

# Model Building and Data Analysis


```r
library(caret); 
library(rattle)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.2.5
```

```r
library(rpart)
library(randomForest)
```
## Data import and Data clean


```r
set.seed(11111)
# import data and replace all missing value by NA
train <- read.csv("inTraining.csv", header = TRUE, na.strings = c("NA", "#DIV/0!", ""))
test <- read.csv("inTesting.csv", header = TRUE, na.strings = c("NA", "#DIV/0!", ""))
# check the dimmensions and the names 
dim(train)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```

```r
all.equal(colnames(test)[1:length(colnames(test))-1], colnames(train)[1:length(colnames(train))-1])
```

```
## [1] TRUE
```

```r
# drop the inrelative variables and variables with all NAs
train <- train[,-c(1:7)]
train <- train[,colSums(is.na(train))==0]
test <- test[,-(1:7)]
test <- test[,colSums(is.na(test))==0]
# check the cleaned datasets
dim(train)
```

```
## [1] 19622    53
```

```r
dim(test)
```

```
## [1] 20 53
```
## Cross Validation


```r
# partitioning the training data into 2 sets
index <- createDataPartition(train$classe, p = 0.75, list = FALSE)
inTrain <- train[index,]
inTest <- train[-index,]
# plot the frequency of each level of classe 
plot(inTrain$classe, col = "red", main = "Variable classe in the inTrain dataset",
                                  xlab = "classe levels",
                                  ylab = "frequency")
```

![plot of chunk cross validation](figure/cross validation-1.png)
This work builds two prediction models, decision tree and random forest.


```r
# using decision tree to predict 
modelTree <- rpart(classe ~ ., data = inTrain, method = "class")
# plot the decision tree
fancyRpartPlot(modelTree, main = "Decision trees")
```

```
## Warning: labs do not fit even at cex 0.15, there may be some overplotting
```

![plot of chunk prediction model: Decision Tree](figure/prediction model: Decision Tree-1.png)

```r
# predict
prediction1 <- predict(modelTree, inTest, type = "class")
# statistical results
confusionMatrix(prediction1, inTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1288  138   23   33    5
##          B   43  550   60   66   78
##          C   31  174  711   88  116
##          D   24   68   60  541   61
##          E    9   19    1   76  641
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7608          
##                  95% CI : (0.7486, 0.7727)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.697           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9233   0.5796   0.8316   0.6729   0.7114
## Specificity            0.9433   0.9375   0.8990   0.9480   0.9738
## Pos Pred Value         0.8662   0.6901   0.6348   0.7175   0.8592
## Neg Pred Value         0.9687   0.9028   0.9619   0.9366   0.9375
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2626   0.1122   0.1450   0.1103   0.1307
## Detection Prevalence   0.3032   0.1625   0.2284   0.1538   0.1521
## Balanced Accuracy      0.9333   0.7586   0.8653   0.8105   0.8426
```


```r
# using random forest
modelForest <- randomForest(classe ~ ., data= inTrain, method="class")
# predict
prediction2 <- predict(modelForest, inTest, type = "class")
# statistical results
confusionMatrix(prediction2, inTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    5    0    0    0
##          B    0  942    2    0    0
##          C    0    2  852    5    0
##          D    0    0    1  796    0
##          E    0    0    0    3  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9963          
##                  95% CI : (0.9942, 0.9978)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9954          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9926   0.9965   0.9900   1.0000
## Specificity            0.9986   0.9995   0.9983   0.9998   0.9993
## Pos Pred Value         0.9964   0.9979   0.9919   0.9987   0.9967
## Neg Pred Value         1.0000   0.9982   0.9993   0.9981   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1921   0.1737   0.1623   0.1837
## Detection Prevalence   0.2855   0.1925   0.1752   0.1625   0.1843
## Balanced Accuracy      0.9993   0.9961   0.9974   0.9949   0.9996
```
## conclusion

According to the comparison, random forest algorithm is better than the decision tree algorithm.
Precisely, the random forest method has an accuracy of 99.6% compared to 76.1% of decision tree.
We choose the random tree method to test the 20 cases.

# Predict the 20 cases

Using random forest method to predict the 20 cases.


```r
# using random forest method to predict the 20 cases
predictionFinal <- predict(modelForest, test, type = "class")
print(predictionFinal)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```











