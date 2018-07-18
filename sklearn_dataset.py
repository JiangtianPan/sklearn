# self_made data sklearn.datasets.make_regression(n_samples=100, n_features=100, n_informative=10, n_targets=1,
# bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)[source]
from __future__ import  print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import  matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_x = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_x, data_y)

print(model.predict(data_x[:4, :]))
print(data_y[:4])

# create virtual data
x1, y1 = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
x2, y2 = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=50)
# the larger noise, the more scatter sampler nodes
# visualize created data, scatter() then show()
plt.scatter(x1, y1)
plt.show()
plt.scatter(x2, y2)
plt.show()
