from sklearn.datasets import load_iris

from experiments import Experiment

if __name__ == '__main__':
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

    #print('in main: {}'.format(y))

    experiment = Experiment('iris example experiment', models=['rf', 'dt', 'ab', 'knn'], exp_type='classification')
    experiment.train(
        X, 
        y, 
        metrics=['acc', 'rec', 'prec'], 
        cv=True, 
        n_folds=10, 
        shuffle=True)
    experiment.summarizeMetrics()