#setwd("H:/Coursera/MachineLearning/Project/work")

require(caret)
require(AppliedPredictiveModeling)
require(rpart)
require(randomForest)
require(e1071)
require(ggplot2)
require(grid)

## getting and cleaning the data

# load the training data (160 columns)
rawTraining <- read.csv("pml-training.csv",na.strings = c("NA", ""))

# select only variables with a valid first observation (no. columns reduced to 60)
tidyTraining <- rawTraining[,!is.na(rawTraining[1,])]

# check there are no more NAs
sum(is.na(tidyTraining[,])); sum(complete.cases(tidyTraining))

# eliminate columns: row id, time stamps, new_window, num_window
subTidyTraining <- tidyTraining[,c(-1,-3,-4,-5,-6,-7)]

# Convert user name into a text variable into integer and investigate if there is an obvious correlation with classe.
subTidyTraining$user_name <- as.integer(subTidyTraining$user_name)
attach(mtcars)
par(mfrow=c(3,2))
hist(as.integer(subTidyTraining[subTidyTraining$user_name==1,]$classe),main="user 1",xlab="classe",breaks=seq(0.5,5.5,l=6))
hist(as.integer(subTidyTraining[subTidyTraining$user_name==2,]$classe),main="user 2",xlab="classe",breaks=seq(0.5,5.5,l=6))
hist(as.integer(subTidyTraining[subTidyTraining$user_name==3,]$classe),main="user 3",xlab="classe",breaks=seq(0.5,5.5,l=6))
hist(as.integer(subTidyTraining[subTidyTraining$user_name==4,]$classe),main="user 4",xlab="classe",breaks=seq(0.5,5.5,l=6))
hist(as.integer(subTidyTraining[subTidyTraining$user_name==5,]$classe),main="user 5",xlab="classe",breaks=seq(0.5,5.5,l=6))
hist(as.integer(subTidyTraining[subTidyTraining$user_name==6,]$classe),main="user 6",xlab="classe",breaks=seq(0.5,5.5,l=6))
par(mfrow=c(1,1))

# remove the user_name variable since (i) the prediction should work for other people and (ii) there is 
# no strong correlation in figure 1
subTidyTraining <- subTidyTraining[,-1]

# double check that "classe" is in column  53 in the subset 
names(subTidyTraining[53])


## PREPROCESSING
# check that there are no near zero variables
sum(nearZeroVar(subTidyTraining,saveMetrics = TRUE)$nzv)
# other types of preprocessing are possible but we skip that step to improve the clarity of the interpretation



# split data into myTraining and myValidation subsets to estimate out-of-sample error
inTrain <- createDataPartition(subTidyTraining$classe, p=0.6, list=FALSE)
myTraining <- subTidyTraining[inTrain,]
myValidation <- subTidyTraining[-inTrain,]
dim(myTraining); dim(myValidation)


## PREDICTION
set.seed(3433)
# the caret version does not work well
modFitRpart <- rpart(classe ~ ., data=myTraining, method="class")
print(modFitRpart$finalModel)
predRpart <- predict(modFitRpart,newdata = myValidation, type = "class")
confusionMatrix(myValidation$classe,predRpart)
#add a measure of homogeneity, if if I have time


set.seed(3435)
modFitSVM <- svm(classe~.,data=myTraining)
print(modFitSVM)
predSVM <- predict(modFitSVM,newdata = myValidation)
confusionMatrix(myValidation$classe,predSVM)


set.seed(3434)
# the caret version does not work well
modFitRF <- randomForest(classe~.,data=myTraining,ntree=400,importance=TRUE)
predRF <- predict(modFitRF,newdata = myValidation)
confusionMatrix(myValidation$classe,predRF)

# since RF is the winner, let's focus on RF do so interpretation
plot(modFitRF,main="No singificant improvement in RF fit for >200 trees")

# importance of variables
importanceDF <- data.frame(importance(modFitRF)[,c(6,7)])
importanceDF <- cbind(rownames(importanceDF),importanceDF)
colnames(importanceDF)=c("Variable","MeanDecreaseAccuracy","MeanDecreaseGini")

importanceDFaccuSort <- transform(importanceDF, Variable = reorder(Variable, MeanDecreaseAccuracy))
accuracyPlot<-ggplot(data=importanceDFaccuSort, aes(x=Variable, y=MeanDecreaseAccuracy)) + 
  ylab("Mean Decrease Accuracy")+xlab("")+geom_bar(stat="identity",width=.7)+coord_flip() 
grid.draw(accuracyPlot)

importanceDFginiSort <- transform(importanceDF, Variable = reorder(Variable, MeanDecreaseGini))
giniPlot=ggplot(data=importanceDFginiSort, aes(x=Variable, y=MeanDecreaseGini)) + 
  ylab("Mean Decrease Gini")+xlab("")+geom_bar(stat="identity",width=.7)+ coord_flip() 
grid.draw(giniPlot)


## APPLY PREDICTION TO TESTING DATA
rawTesting <- read.csv("pml-testing.csv",na.strings = c("NA", ""))

# use the same subset of columns as in the  Training dataset (except the classe select)
colSelect <- colnames(subTidyTraining[,-53])
subTidyTesting <- rawTesting[colSelect]

# predict using Random Forest
predRFtesting <- predict(modFitRF,newdata = subTidyTesting)
predRFtesting