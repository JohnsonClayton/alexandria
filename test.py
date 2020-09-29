from experiments import Experiments, Experiment
from utils import Helper

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from abc import ABCMeta

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
        try:
            experiment.setExperimentType(experiment_type)

            # This should never run!
            self.assertEqual(0, 1)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment must be \'regression\' or \'classification\', cannot be bloogus!' )

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

        

if __name__ == '__main__':
    unittest.main()
