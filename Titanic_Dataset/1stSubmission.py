#Import libraries
import pandas
import numpy as np
import scipy
import scikits.learn

#********************** The training set ********************************

#Read train data
titanic=pandas.read_csv("train.csv")
#Print first 5 rows of data
print(titanic.head(5))
#Print high level descriptors of data
print(titanic.describe())
#Fill missing values of Age with median
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
#Find all unique genders
print(titanic["Sex"].unique())
#Replace all occurences of male with 0 and female with 1
titanic.loc[titanic["Sex"]=="male","Sex"]=0
titanic.loc[titanic["Sex"]=="female","Sex"]=1
#Find all unique values for Embarked
print(titanic["Embarked"].unique())
#Replace missing values of Embarked with "S"; because that is the most
#common embarkation port
titanic["Embarked"]=titanic["Embarked"].fillna("S")
#Assign 0 to S, 1 to C, 2 to Q
titanic.loc[titanic["Embarked"]=="S", "Embarked"]=0
titanic.loc[titanic["Embarked"]=="C", "Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q", "Embarked"]=2

#************************** Linear Regression *****************************

#Import linear regression class
from sklearn.linear_model import LinearRegression
#Helper to do cross validation
from sklearn.cross_validation import KFold
#Columns used to predict target
predictors=["Pclass","Sex","Age","SibSp","Parch", "Fare", "Embarked"]
#Initialize algorithm class
alg=LinearRegression()
#Generate cross validation folds
kf=KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions=[]
for train, test in kf:
    #The predictors used to train the algorithm
    train_predictors=(titanic[predictors].iloc[train,:])
    #The target used to train the algorithm
    train_target=titanic["Survived"].iloc[train]
    #Training the algorithm using predictors and target
    alg.fit(train_predictors, train_target)
    #Make predictions on the test fold
    test_predictions=alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
#Concatenate the three numpy arrays into one
predictions=np.concatenate(predictions,axis=0)
#Map predictions to outcomes
predictions[predictions > .5]=1
predictions[predictions <= .5]=0
#Calculate accuracy as number of values in predictions that are same as
#their counterparts in titanic["Survived"] divided by total no of passengers
accuracy=sum(predictions[predictions==titanic["Survived"]])/len(predictions)

#************************ Logistic Regression ***************************

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
#Initialize algorithm
alg=LogisticRegression(random_state=1)
#Compute accuracy score for all the cross validation folds
scores=cross_validation.cross_val_score(alg,titanic[predictors], titanic["Survived"], cv=3)
#Take the mean of the scores
print(scores.mean())

#************************ The test set **********************************

titanic_test = pandas.read_csv("test.csv")
#Process titanic_test the same way as titanic
titanic_test["Age"]=titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test.loc[titanic_test["Sex"]=="male","Sex"]=0
titanic_test.loc[titanic_test["Sex"]=="female","Sex"]=1
titanic_test["Embarked"]=titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"]=="S", "Embarked"]=0
titanic_test.loc[titanic_test["Embarked"]=="C", "Embarked"]=1
titanic_test.loc[titanic_test["Embarked"]=="Q", "Embarked"]=2
#Replace missing value in Fare with median
titanic_test["Fare"]=titanic_test["Fare"].fillna(titanic_test["Fare"].median())

#******************** Kaggle submission ********************************

#Initialize algorithm class
alg=LogisticRegression(random_state=1)
#Train algorithm using training data
alg.fit(titanic[predictors],titanic["Survived"])
#Make predictions using test set
predictions=alg.predict(titanic_test[predictors])
#Create new dataframe with only the columns Kaggle wants from the dataset
submission=pandas.DataFrame({"PassengerID":titanic_test["PassengerId"],"Survived":predictions})
#Output to csv
submission.to_csv("kaggle.csv", index=False)
