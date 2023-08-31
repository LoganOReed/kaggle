import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error

# Setup Data
train_data = pd.read_csv("data/housing-prices/train.csv")
test_data = pd.read_csv("data/housing-prices/test.csv")

y_train = train_data["SalePrice"]
X_train = train_data[["MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"]]

# test csv doesnt have salePrice
# y_test = test_data["SalePrice"]
X_test = test_data[["MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"]]


# Setup Model
M = RandomForestRegressor(random_state=1)
M.fit(X_train, y_train)

pred = M.predict(X_test)
# print("Validation MAE for Random Forest Model: {:,.0f}".format(pred_mae))
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': pred})
output.to_csv('data/housing-prices/submission.csv', index=False)
