#Import libraries
import pandas
import numpy as np
import scipy
import scikits.learn
import re
import operator
import matplotlib.pyplot as plt

#********************** The training set ********************************

#Read train data
titanic=pandas.read_csv("C:\Users\Hiranya\Downloads\python/titanic/train.csv")

#Fill missing values of Age with median
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())

#Replace all occurences of male with 0 and female with 1
titanic.loc[titanic["Sex"]=="male","Sex"]=0
titanic.loc[titanic["Sex"]=="female","Sex"]=1

#Replace missing values of Embarked with "S"; because that is the most
#common embarkation port
titanic["Embarked"]=titanic["Embarked"].fillna("S")

#Assign 0 to S, 1 to C, 2 to Q
titanic.loc[titanic["Embarked"]=="S", "Embarked"]=0
titanic.loc[titanic["Embarked"]=="C", "Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q", "Embarked"]=2

#*************************** Random forest *******************************

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
predictors=["Pclass","Sex","Age","SibSp","Parch", "Fare", "Embarked"]

#Initialize our algorithm with the default paramters
#n_estimators is the number of trees we want to make
#min_samples_split is the minimum number of rows we need to make a split
#min_samples_leaf is the minimum number of samples we can have at the 
#place where a tree branch ends
alg=RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
scores=cross_validation.cross_val_score(alg, titanic[predictors],titanic["Survived"],cv=3)
print(scores.mean())

#Increasing n_estimators, min_samples_split and min_samples_leaf to reduce overfitting
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores=cross_validation.cross_val_score(alg, titanic[predictors],titanic["Survived"],cv=3)
print(scores.mean())

#********************** Generating new features ***************************

#Generating family size column
titanic["FamilySize"]=titanic["SibSp"]+titanic["Parch"]

#The .apply method generates a new series
titanic["NameLength"]=titanic["Name"].apply(lambda x:len(x))

#A function to get title from name
def get_title(name):
    #Use regular expression to search for a title
    #Titles consist of uppercase, lowercaae and period
    title_search=re.search('([A-Za-z]+)\.',name)
    #If title exists, extract and return it
    if title_search:
        return title_search.group(1)
    return ""
    
#Get all titles and count how often each one occurs
titles=titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

#Map each title to integer
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, 
"Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10,
"Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles==k]=v
    
#Verify that we converted everything
print(pandas.value_counts(titles))

#Add in the title column
titanic["Title"]=titles

#Create family id by concatenating last name with FamilySize
# A dictionary mapping family name to id
family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# Get the family ids with the apply method
family_ids = titanic.apply(get_family_id, axis=1)

#Compress family ids of families under 3 members into one code
family_ids[titanic["FamilySize"]<3]= -1

#Print count of each unique id
print(pandas.value_counts(family_ids))

titanic["FamilyId"]=family_ids

#*********************** Finding best feature ******************************

#Find columns that correlate most with what we are trying to predict (Survived)
from sklearn.feature_selection import SelectKBest, f_classif

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

#Perform feature selection
selector= SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

#Get raw p-values for each feature and transform from p-values to scores
scores= -np.log10(selector.pvalues_)

#Plot the scores
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

#Pick only the four best features
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
scores=cross_validation.cross_val_score(alg, titanic[predictors],titanic["Survived"],cv=3)
print(scores.mean())

#******************************* Ensembling *****************************

#We can ensemble different classifiers to improve predictions
#Here we will ensemble logistic regression  (trained on most linear predictors) and gradient boosted tree (trained on all predictors)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

#The algorithms we want to ensemble
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
    ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

#Initialize cross validation folds
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions=[]
for train, test in kf:
    train_target= titanic["Survived"].iloc[train]
    full_test_predictions=[]
    #Make predictions for each algorithm on each fold
    for alg, predictors in algorithms:
        #Fit the algorithm on the training data
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        #Select and predict on the test fold
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    #Use a simple ensembling scheme, just average predictions
    test_predictions = (full_test_predictions[0]+full_test_predictions[1])/2
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)
    
#Put all predictions together into one array
predictions = np.concatenate(predictions, axis=0)

#Compute accuracy by comparing to the training data
accuracy = sum(predictions[predictions==titanic["Survived"]])/len(predictions)
print(accuracy)

#******************************* Test set *****************************

titanic_test = pandas.read_csv("C:\Users\Hiranya\Downloads\python/titanic/test.csv")

#Process titanic_test the same way as titanic
titanic_test["Age"]=titanic_test["Age"].fillna(titanic["Age"].median())
titanic_test.loc[titanic_test["Sex"]=="male","Sex"]=0
titanic_test.loc[titanic_test["Sex"]=="female","Sex"]=1
titanic_test["Embarked"]=titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"]=="S", "Embarked"]=0
titanic_test.loc[titanic_test["Embarked"]=="C", "Embarked"]=1
titanic_test.loc[titanic_test["Embarked"]=="Q", "Embarked"]=2
titanic_test["Fare"]=titanic_test["Fare"].fillna(titanic_test["Fare"].median())

#First, we'll add titles to the test set.
titles = titanic_test["Name"].apply(get_title)

#We're adding the Dona title to the mapping, because it's in the test set, but not the training set
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles

#Check the counts of each unique title.
print(pandas.value_counts(titanic_test["Title"]))

#Now, we add the family size column.
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

# Now we can add family ids.
# We'll use the same ids that we did earlier.
print(family_id_mapping)
family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids
titanic_test["NameLength"]=titanic_test["Name"].apply(lambda x:len(x))

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)
    
# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1
predictions=predictions.astype(int)
#Submission file
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("kaggle.csv", index=False)