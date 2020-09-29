# alexandria
This is a high-level machine learning framework that (will) allow for the users to easily run multiple types of machine learning experiments at the drop of a hat. An example for the desired API is below:

```
> experiment_1 = Experiment(name='exp_1', 
                        exp_type='classification', 
                        models=[
                            'random forest', 
                            'CART', 
                            AdaBoostClassifier()
                        ], 
                        metrics=[
                            'training time', 
                            'prediction time', 
                            'accuracy', 
                            'recall', 
                            'precision', 
                            'conf_matrix'
                  ])
> experiment_1.setRandomState(0)
> experiment_1.run(X, y, cv=True, n_folds=10)
Running...done
> metrics = experiment_1.collectMetrics()
> metrics.present()
```
