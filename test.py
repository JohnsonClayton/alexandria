from experiments import Experiments, Experiment
from utils import Helper
from model import Model

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from abc import ABCMeta

from sklearn.datasets import load_iris, make_classification

import sys

import unittest

class TestExperiment(unittest.TestCase):
    def test_init(self):
        experiment = Experiment()
        self.assertEqual( experiment.getName(), 'unnamed_experiment' )
        self.assertEqual( experiment.getModels(), {} )

    def test_isValidExperimentType(self):
        experiment = Experiment()
        experiment_type = 'classification'
        self.assertEqual( experiment.isValidExperimentType(experiment_type), True )

        experiment_type = 'bloogus!'
        self.assertEqual( experiment.isValidExperimentType(experiment_type), False )

    def test_addModel(self):
        experiment = Experiment('experiment_1')
        experiment.addModel('rf')
        dt_model = Model('decision tree', DecisionTreeClassifier, {'random_state': 0, 'max_depth': 2})
        experiment.addModel(dt_model)

        models_dict = experiment.getModels()

        print(models_dict)
        self.assertTrue( 'rf' in models_dict )

    def test_setExperimentType(self):
        experiment = Experiment()
        experiment_type = 'bloogus!'
        try:
            experiment.setExperimentType(experiment_type)

            # This should never run!
            self.assertEqual(0, 1)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment must be \'regression\' or \'classification\', cannot be bloogus!' )

        experiment_type = 'classification'
        experiment.setExperimentType(experiment_type)
       
        self.assertEqual( experiment.getExperimentType(), experiment_type )

    def test_run(self):
        experiment = Experiment('exp1', models=['rf', 'dt'], exp_type='classification')
        iris = load_iris()

        experiment.run(iris.data[:120], iris.target[:120])

        try:
            experiment.run(iris.data[:120], iris.target[:12])

            # This should never be executed!
            self.assertEqual( 0, 1 )
        except ValueError as ve:
            self.assertEqual( str(ve), 'Data and target provided to \'exp1\' must be same length:\n\tlen of data: 120\n\tlen of target: 12' )

        experiment = Experiment(models=['rf', 'dt'], exp_type='classification')
        experiment.run(iris.data[:120], iris.target[:120])

        try:
            experiment.run(iris.data[:47], iris.target)

            # This should never be executed!
            self.assertEqual( 0, 1 )
        except ValueError as ve:
            self.assertEqual( str(ve), 'Data and target provided to \'unnamed_experiment\' must be same length:\n\tlen of data: 47\n\tlen of target: 150' )

    def test_predict(self):
        experiment = Experiment('exp1', models=['rf', 'dt'], exp_type='classification')
        iris = load_iris()

        experiment.run(iris.data[:120], iris.target[:120])
        predictions = experiment.predict(iris.data[120:])
        actual = {
                'rf':
                    [2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
                'dt':
                    [2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}
        self.assertEqual( predictions, actual )

class TestExperiments(unittest.TestCase):
    def test_init(self):
        experiments = Experiments()
        self.assertEqual( experiments.getNumOfExperiments(), 0 )
        self.assertEqual( experiments.getExperiments(), {} )

        try:
            experiments.runAllExperiments()

            # This should never be reached in the case of a successful execution
            self.assertEqual( 0, 1)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiments object has no models to run!')

        try:
            experiments.addExperiment('random forest')

            # This should never be reached in the case of a successful execution
            self.assertEqual( 0, 1)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Object must be Experiment object: random forest')

        try:
            experiments.addExperiment( Experiment(1) )

            
            # This should never be reached in the case of a successful execution
            self.assertEqual( 0, 1)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment name attribute must be string: 1' )
        self.assertEqual( experiments.getNumOfExperiments(), 0 )

        experiments.addExperiment( Experiment('1') )
        experiments.addExperiment( Experiment('2') )
        experiments.addExperiment( Experiment('3') )
        experiments.addExperiment( Experiment('4') )

        self.assertEqual( experiments.getNumOfExperiments(), 4 )
        self.assertEqual( experiments.getExperimentNames(), ['1', '2', '3', '4'] )


class TestHelper(unittest.TestCase):
    def test_init(self):
        helper = Helper()

        model_name = 'rf:classifier'
        model = helper.getDefaultModel(model_name)
        self.assertEqual( model, RandomForestClassifier ) 
        
        args = helper.getDefaultArgs(model_name)
        self.assertEqual( args, {'random_state':0} )
        
        model = model(**args)
        self.assertEqual( type(model), RandomForestClassifier )

        try:
            helper.setRandomState( 'cheeseburger' )

            # This should never run!
            self.assertEqual(0, 1)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Random state must be an integer: cheeseburger' )

        helper.setRandomState(1)

        model_name = 'dt:regressor'
        model = helper.getDefaultModel(model_name)
        self.assertEqual( model, DecisionTreeRegressor ) 
        
        args = helper.getDefaultArgs(model_name)
        self.assertEqual( args, {'random_state':1} )
        
        model = model(**args)
        self.assertEqual( type(model), DecisionTreeRegressor )

class TestModel(unittest.TestCase):
    def test_init(self):
        model = Model()
        self.assertEqual( model.getName(), None )


    def test_setConstructor(self):
        model = Model('best name ever!', lambda x : x**2, {'x':1})
        self.assertEqual( model.getConstructor()(5), 25 )
        self.assertEqual( model.getConstructor()(4), 16 )
        self.assertEqual( model.getConstructorArgs(), {'x':1} )

        model.setConstructor( lambda x : x**3 )
        self.assertEqual( model.getConstructor()(3), 27 )
        self.assertEqual( model.getConstructor()(1), 1 )
        self.assertEqual( model.getConstructor()(4), 64 )

        model.setConstructor()
        self.assertEqual( model.getConstructor(), None )
        model.setConstructor(None)
        self.assertEqual( model.getConstructor(), None )
        
        try:
            model.setConstructor('the lazy brown dog jumped over the fox')

            # This should never run!
            self.assertEqual( 0, 1 )
        except ValueError as ve:
            self.assertEqual( str(ve), 'Model \'best name ever!\' cannot set constructor as non-callable value: the lazy brown dog jumped over the fox')

    def test_run(self):
        model = Model(name='rf1', constr=RandomForestClassifier, constr_args={'max_depth': 2, 'random_state': 0})

        # Creating small dataset as per https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        X, y = make_classification(n_samples=1000, n_features=4,
                n_informative=2, n_redundant=0,
                random_state=0, shuffle=False)
        model.run(X=X, y=y)
        self.assertListEqual( model.predict( [[0, 0, 0, 0]] ), [1] )

        model = Model('dt1', DecisionTreeClassifier, {'random_state':0})
        iris = load_iris()
        model.run( iris.data[:120], iris.target[:120] )
        self.assertListEqual( model.predict( iris.data[120:] ), [2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] )

        try:
            model.run( iris.data[:120], iris.target[:119] )

            # This should never execute!
            self.assertEqual( 0, 1 )
        except ValueError as ve:
            self.assertEqual( str(ve), 'Model dt1 cannot be trained when data and target are different lengths:\n\tlen of data: 120\n\tlen of target: 119' )

if __name__ == '__main__':
    unittest.main()
