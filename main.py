from sklearn.datasets import load_iris

from experiments import Experiment

if __name__ == '__main__':
    # Data preprocessing
    iris = load_iris()
    X_train, y_train = iris.data[:120], iris.target[:120]
    X_test, y_test = iris.data[120:], iris.target[120:]

    experiment = Experiment('iris example experiment', models=['rf', 'dt', 'ab', 'knn'], exp_type='classification')
    experiment.train(
        X_train, 
        y_train, 
        metrics=['acc', 'rec', 'prec'], 
        cv=True, 
        n_folds=10, 
        shuffle=True)
    experiment.summarizeMetrics()