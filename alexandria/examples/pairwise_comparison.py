from alexandria.experiment import Experiment

from sklearn.datasets import load_iris

if __name__ == '__main__':
    iris = load_iris()
    exp = Experiment(
        name='Pairwise Comparison Example',
        dataset=iris,
        xlabels='data',
        ylabels='target',
        models=[
            'rf',
            'dt',
            'knn',
            'nb',
            'da',
            'ab',
            'gb',
            'lr',
            'xgb',
            'svm'
        ]
    )

    exp.compareModels_tTest(a=0.05)