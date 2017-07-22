import pandas
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
titanic = pandas.read_csv("train.csv")
#print titanic.head(3)
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic.loc[titanic["Sex"] == "male","Sex"] = 0
titanic.loc[titanic["Sex"] == "female","Sex"] = 1
titanic['Embarked'] = titanic['Embarked'].fillna("S")
titanic.loc[titanic["Embarked"] == "S","Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C","Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q","Embarked"] = 2
#evaluate the value of the algorithm
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
alg = LinearRegression()
kf = KFold(titanic.shape[0],n_folds=3,random_state = 1)
predictions = []
for train,test in kf:
    train_predictors = (titanic[predictors].iloc[train,:])
    train_target = titanic['Survived'].iloc[train]
    #print train_target
    alg.fit(train_predictors,train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    #print test_predictions
    predictions.append(test_predictions)
import numpy as np
predictions = np.concatenate(predictions,axis=0)
predictions[predictions > .5]=1
predictions[predictions <= .5]=0
for i,item in enumerate(predictions):
    predictions[i] = int(item)
accuracy = sum(predictions[predictions == titanic["Survived"].astype(int)])/len(predictions)
print accuracy
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
#random forestn algorithm
alg = RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=4,min_samples_leaf=2)
kf = cross_validation.KFold(titanic.shape[0],n_folds=3,random_state=1)
scores = cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=kf)
print scores.mean()
#reset the parameter
titanic["FamilySize"]=titanic["SibSp"] + titanic["Parch"]
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
import re
#calculate the length of the name
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ""
titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

title_mapping = {"Mr": 1,"Miss": 2,"Mrs": 3,"Master": 4,"Dr": 5,"Rev": 6,"Major": 7,"Col": 7,"Mlle": 8,"Countess": 9,"Ms": 10,"Lady": 11,"Jonkheer": 12,"Don": 13,
                 "Mme": 14,"Capt": 15,"Sir": 16}
for k,v in title_mapping.items():
    titles[titles == k] = v
print(pandas.value_counts(titles))
titanic["Title"]=titles
import numpy as np
from sklearn import feature_selection
import matplotlib.pyplot as plt
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize", "Title", "NameLength"]
selector = feature_selection.SelectKBest(feature_selection.f_classif,k=5)
selector.fit(titanic[predictors],titanic["Survived"])
scores = -np.log10(selector.pvalues_)
plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)),predictors,rotation="vertical")
plt.show()
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
algorithm = [[GradientBoostingClassifier(random_state=1,n_estimators=25,max_depth=3),["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize", "Title", "NameLength"]],
             [LogisticRegression(random_state=1),["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize", "Title", "NameLength"]]
             ]
kf = KFold(titanic.shape[0],n_folds=3,random_state=1)
predictions = []
full_prediction = []
for train,test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    for alg,predictors in algorithm:
        alg.fit((titanic[predictors].iloc[train,:]),train_target)
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0]+full_test_predictions[1])/2
    test_predictions[test_predictions<=.5] = 0
    test_predictions[test_predictions>.5] = 1
    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)
result = sum(predictions[predictions == titanic["Survived"]])/len(predictions)
print result



