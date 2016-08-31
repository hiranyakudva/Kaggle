# Packages to be used
library(dplyr)
library(caret)

# Load dataset into mydata
mydata <- read.csv('train.csv', header=T, na.strings=c(""))

#check for missing values
sapply(mydata, function(x) sum(is.na(x)))

# Variable 'Cabin' has too many missing values, we will not use it. Also PassengerId and Ticket do not serve much purpose.
# make a new dataset with only required variables
newdata <- select(mydata, Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

# Take care of other missing values
newdata$Age[is.na(newdata$Age)] <- mean(newdata$Age, na.rm=T)
newdata$Embarked[is.na(newdata$Embarked)] <- 'S'

# Creating training and testing set
Index <- createDataPartition(newdata$Survived, p=0.8, list=FALSE)
train <- newdata[Index,]
test <- newdata[-Index,]

# Fit logistic regression model
model <- glm(Survived ~., family=binomial(link="logit"), data=train)
summary(model)

# Test the model on test dataset
fit <- predict(model, newdata=test, type='response')
fit <- ifelse(fit > 0.5,1,0)

# Accuracy of the model
misClassificationError <- mean(fit != test$Survived)
Accuracy <- 1-misClassificationError
Accuracy
