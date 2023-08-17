def best_percent():
    all_best_percent = []
    for i in range(100):
        train_data = pd.read_csv("./Proyectos/Titanic/train.csv") 

        train_data.replace("female", 0, inplace=True)
        train_data.replace("male", 1, inplace=True)

        train_data.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
        train_data = train_data.dropna()

        X = train_data.drop(['Survived'], axis=1)
        y = train_data.Survived

        train_X, test_X, train_y, test_y = train_test_split(X, y) #Random

        forest_model = RandomForestRegressor(random_state=1)
        forest_model.fit(train_X, train_y)
        survived_preds = forest_model.predict(test_X)

        best_hits = 0

        for percent in range(50,100):
            percent = percent/100

            try_survived_preds = np.copy(survived_preds)


            try_survived_preds[try_survived_preds >= percent] = 1
            try_survived_preds[try_survived_preds < percent] = 0
            actual_hits = (test_y == try_survived_preds).sum()

            if (actual_hits > best_hits):
                best_percent = percent
                best_hits = actual_hits
        
        all_best_percent += [best_percent]
    
    average_percent = np.mean(all_best_percent)
    return average_percent

def best_feactures(percent_around):
    train_data = pd.read_csv("./Proyectos/Titanic/train.csv") 

    train_data.replace("female", 0, inplace=True)
    train_data.replace("male", 1, inplace=True)

    train_data.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    train_data = train_data.dropna()

    features = train_data.columns.drop("Survived").values

    all_features = []
    for name in features:
        all_features += [name]

    var_best_feactures = []
    best_hits = 0
    for how_many_features in range(2,len(all_features)+1):
        for i in range(20*2):
            some_features = random.sample(all_features, how_many_features)

            X = train_data[some_features]
            y = train_data.Survived

            train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)
            
            forest_model = RandomForestRegressor(random_state=1)
            forest_model.fit(train_X, train_y)
            survived_preds = forest_model.predict(test_X)

            survived_preds[survived_preds >= percent_around] = 1       
            survived_preds[survived_preds < percent_around] = 0

            actual_hits = (test_y == survived_preds).sum()

            if (actual_hits > best_hits):
                var_best_feactures = some_features
                best_hits = actual_hits
    
    return var_best_feactures
    


import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv("./Proyectos/Titanic/train.csv") 
test_data = pd.read_csv("./Proyectos/Titanic/test.csv") 

train_data.replace("female", 0, inplace=True)
train_data.replace("male", 1, inplace=True)

train_data.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
train_data = train_data.dropna()

print(train_data.describe())
print(train_data.columns)


test_data.replace("female", 0, inplace=True)
test_data.replace("male", 1, inplace=True)

test_PassengerId = test_data.PassengerId

test_data.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
test_data.Fare.fillna(test_data.Fare.mean(), inplace = True)
test_data.Age.fillna(test_data.Age.mean(), inplace = True)

print(test_data.describe())
print(test_data.columns)

percent_around = best_percent()
the_best_feactures = best_feactures(percent_around)

print(percent_around)
print(the_best_feactures)

#train_X = train_data.drop(['Survived'], axis=1)
train_X = train_data[the_best_feactures]
train_y = train_data.Survived

test_X = test_data[the_best_feactures]


forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
survived_preds = forest_model.predict(test_X)



survived_preds[survived_preds >= percent_around] = 1
survived_preds[survived_preds < percent_around] = 0


output = pd.DataFrame({'PassengerId': test_PassengerId, 'Survived': survived_preds.astype(int)})
output.to_csv('./Proyectos/Titanic/submission.csv', index=False)




