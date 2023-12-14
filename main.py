from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()

X = data.data
features = data.feature_names
y = data.target
df_full = pd.DataFrame(X, columns=features)
df_full['target'] = y

df_full.head()

df = df_full.iloc[:15]

X_train, X_test, y_train, y_test = train_test_split(
    df[features],
    df['target'],
    test_size=0.2,
    shuffle=True,
    random_state=3
)

tree = DecisionTreeRegressor(random_state=1)
tree.fit(X_train, y_train)
pred_train = tree.predict(X_train)
pred_test = tree.predict(X_test)
mse_train = mean_squared_error(y_train, pred_train)
mse_test = mean_squared_error(y_test, pred_test)

print(f'squared_error на обучении {mse_train:.2f}')
print(f'squared_error на тесте {mse_test:.2f}')

plt.figure(figsize=(20, 15))
plot_tree(tree, feature_names=features, filled=True)
plt.savefig(f'tree1')

X_train, X_test, y_train, y_test = train_test_split(
    df_full[features],
    df_full['target'],
    test_size=0.2,
    shuffle=True,
    random_state=3
)

tree = DecisionTreeRegressor(random_state=1)
tree.fit(X_train, y_train)

pred_train = tree.predict(X_train)
pred_test = tree.predict(X_test)

mse_train = mean_squared_error(y_train, pred_train)
mse_test = mean_squared_error(y_test, pred_test)

print(f'squared_error на обучении {mse_train:.2f}')
print(f'squared_error на тесте {mse_test:.2f}')

plot_tree(tree, feature_names=features, filled=True)
plt.savefig(f'tree2')

tree = DecisionTreeRegressor(random_state=1,
                             max_depth=15,
                             min_samples_leaf=1,
                             max_leaf_nodes=None)
tree.fit(X_train, y_train)

pred_train = tree.predict(X_train)
pred_test = tree.predict(X_test)

mse_train = mean_squared_error(y_train, pred_train)
mse_test = mean_squared_error(y_test, pred_test)

print(f'squared_error на обучении {mse_train:.2f}')
print(f'squared_error на тесте {mse_test:.2f}')

plot_tree(tree, feature_names=features, filled=True)
plt.savefig(f'tree3')

tree = DecisionTreeRegressor(random_state=1,
                             max_depth=14,
                             min_samples_leaf=36,
                             max_leaf_nodes=350)
tree.fit(X_train, y_train)

pred_train = tree.predict(X_train)
pred_test = tree.predict(X_test)

mse_train = mean_squared_error(y_train, pred_train)
mse_test = mean_squared_error(y_test, pred_test)

print(f'squared_error на обучении {mse_train:.2f}')
print(f'squared_error на тесте {mse_test:.2f}')

plot_tree(tree, feature_names=features, filled=True)
plt.savefig('tree')
plt.show()
