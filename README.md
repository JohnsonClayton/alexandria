# alexandria
This is a high-level machine learning framework that allows for the users to easily run multiple types of machine learning experiments at the drop of a hat. An example for the API is below:

```python
# main.py - DataBunch and DataFrame demonstrations
# Data preprocessing
print('With sklearn DataBunch object...')
iris = load_iris()
X, y = iris.data, iris.target

experiment = Experiment('iris example experiment', models=['rf', 'dt', 'ab', 'knn'], exp_type='classification')
experiment.train(
    X, 
    y, 
    metrics=['acc', 'rec', 'prec'], 
    cv=True, 
    n_folds=10, 
    shuffle=True)
experiment.summarizeMetrics()

# Data preprocessing for dataframe object
print('With pandas DataFrame object...')
iris_df = load_iris(as_frame=True).frame
X = iris_df.loc[:, iris_df.columns != 'target']
y = iris_df['target']

experiment = Experiment('iris example experiment', models=['rf', 'dt', 'ab', 'knn'], exp_type='classification')
experiment.train(
    X, 
    y, 
    metrics=['acc', 'rec', 'prec'], 
    cv=True, 
    n_folds=10, 
    shuffle=True)
experiment.summarizeMetrics()
```
```
With sklearn DataBunch object...
model      acc_avg    acc_std    rec_avg    rec_std    prec_avg    prec_std
-------  ---------  ---------  ---------  ---------  ----------  ----------
rf        0.958333  0.0559017   0.958333  0.0559017    0.965278   0.0516883
dt        0.95      0.0666667   0.95      0.0666667    0.9625     0.0534497
ab        0.95      0.0666667   0.95      0.0666667    0.9625     0.0534497
knn       0.966667  0.0408248   0.966667  0.0408248    0.975      0.030932
With pandas DataFrame object...
model      acc_avg    acc_std    rec_avg    rec_std    prec_avg    prec_std
-------  ---------  ---------  ---------  ---------  ----------  ----------
rf        0.958333  0.0559017   0.958333  0.0559017    0.965278   0.0516883
dt        0.95      0.0666667   0.95      0.0666667    0.9625     0.0534497
ab        0.95      0.0666667   0.95      0.0666667    0.9625     0.0534497
knn       0.966667  0.0408248   0.966667  0.0408248    0.975      0.030932
```
