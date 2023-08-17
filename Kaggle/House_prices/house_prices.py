def best_feactures(all_features,how_many_features):
    train_data = pd.read_csv("./Proyectos/House_prices/train.csv") 

    global var_best_feactures
    global best_mae
    for i in range(100):
        some_features = random.sample(all_features, how_many_features)
        X = train_data[some_features]
        y = train_data["SalePrice"]

        train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)

        forest_model = RandomForestRegressor(random_state=1)
        forest_model.fit(train_X, train_y)
        house_prices_preds = forest_model.predict(test_X)
        actual_mae = mean_absolute_error(test_y, house_prices_preds)
        if (actual_mae < best_mae):
            var_best_feactures = some_features
            best_mae = actual_mae


import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error




train_data = pd.read_csv("./Proyectos/House_prices/train.csv") 
test_data = pd.read_csv("./Proyectos/House_prices/test.csv") 
print(train_data.describe())
print(train_data.columns[1:-1])

total_row = len(train_data.index)

possible_features = []
for column_name in train_data.columns[1:-1]:
    df_per_column = train_data[column_name]
    if(df_per_column.dtypes == "int64" and df_per_column.describe()["count"] == total_row):
        if (not test_data[column_name].isnull().values.any()):
            possible_features += [column_name]

#print(possible_features)
var_best_feactures = []
best_mae = 1000000000000000
print(len(possible_features))
for i in range(1,len(possible_features)+1):
    best_feactures(possible_features,i)

features = var_best_feactures

#Columnas del curso de machine learning de Kaggle:
#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#Mejores columnas de momento:
# features = ['YearBuilt', 'MiscVal', 'BedroomAbvGr', 'Fireplaces', 'LotArea', 'OverallQual', 'GrLivArea', '1stFlrSF', 'MSSubClass']
#Supuestamente el mejor MAE que corresponde a esas columnas:
# MAE = 19545.055710067405

print(best_mae)
print(features)


train_X = train_data[features]
train_y = train_data["SalePrice"]

test_X = test_data[features]


forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
house_prices_preds = forest_model.predict(test_X)



output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': house_prices_preds})
output.to_csv('./Proyectos/House_prices/submissionPrueba.csv', index=False)


