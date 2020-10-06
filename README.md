# alexandria
This is a high-level machine learning framework that allows for the users to easily run multiple types of machine learning experiments at the drop of a hat. An example for the API is below:

```python
# main.py
from sklearn.datasets import load_iris

from experiments import Experiment

if __name__ == '__main__':
    # Data preprocessing
    iris = load_iris()
    X_train, y_train = iris.data[:120], iris.target[:120]
    X_test, y_test = iris.data[120:], iris.target[120:]

    experiment = Experiment('iris example experiment', 
                            models=['rf', 'dt', 'ab', 'knn'], 
                            exp_type='classification')
    experiment.train(
        X_train, 
        y_train, 
        metrics=['acc', 'rec', 'prec'], 
        cv=True, 
        n_folds=10, 
        shuffle=True)
    experiment.summarizeMetrics()
```
```
model      acc_avg    acc_std    rec_avg    rec_std    prec_avg    prec_std
-------  ---------  ---------  ---------  ---------  ----------  ----------
rf        0.958333  0.0559017   0.958333  0.0559017    0.965278   0.0516883
dt        0.95      0.0666667   0.95      0.0666667    0.9625     0.0534497
ab        0.95      0.0666667   0.95      0.0666667    0.9625     0.0534497
knn       0.966667  0.0408248   0.966667  0.0408248    0.975      0.030932
```
