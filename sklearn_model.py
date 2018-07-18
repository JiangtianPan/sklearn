from sklearn import datasets
from sklearn.linear_model import LinearRegression

loaded_data = datasets.load_boston()
data_x = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
# model.fit, model.predict
model.fit(data_x, data_y)
print(model.predict(data_x[:4, :]))

# model.coef_, model.intercept_ are model attributes
# for LinearRegression, they are slope and intercept
print(model.coef_)
print(model.intercept_)
# model.get_params(), obtain the pre-defined parameters
print(model.get_params())
# model.score(data_x, data_y), score the model based on R^2 coefficient of determination
# R^2 is the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
# the output is the precision of the model
print(model.score(data_x, data_y))