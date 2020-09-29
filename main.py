from experiments import Experiments, Experiment

from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    exps = Experiments()

    exp1 = Experiment(name='test_experiment_1', models='rf')
    exp1.addModel(RandomForestClassifier())
    exp1.setRandomState(0)
