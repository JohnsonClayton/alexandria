import unittest

from alexandria.experiment import Experiment
from alexandria.dataset import DatasetManager
from alexandria.models import ModelsManager

from alexandria.models import sklearn

def fail(who):
    who.assertTrue( False )

class TestExperiment(unittest.TestCase):
    def test_init(self):
        # Check that the initializations occur correctly
        name = 'experiment 1'
        exp = Experiment(name=name)

        self.assertEqual( exp.name, name )
        self.assertIsInstance( exp.dm, DatasetManager )
        self.assertIsInstance( exp.models_manager, ModelsManager )

        # Fail if Experiment name is not a string
        try:
            name = None
            exp = Experiment(name=name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( name ) ) ))

        try:
            name = ['name 1', 'name 2']
            exp = Experiment(name=name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( name ) ) ))

        try:
            name = 512
            exp = Experiment(name=name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( name ) ) ))

        # Check that we can add models from the constructor
        #  There are a few different ways we can do this...

        # Provide one library and a list (or single) of models
        exp = Experiment(name='experiment 1', libs='sklearn', models=['rf', 'dt'])
        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertIsInstance( models[1], sklearn.DecisionTree )

        # Provide a list of libraries and a list of models, they must be the same length and will match one-to-one
        exp = Experiment(name='experiment 1', libs=['sklearn', 'scikit-learn'], models=['rf', 'decision tree'])
        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertIsInstance( models[1], sklearn.DecisionTree )

        # Provide a dictionary matching a desired library to a desired model
        default_args = {'random_state': 19}
        modellibdict = {
            'scikit-learn': {'model': 'random forest', 'args': default_args},
            'sklearn': 'dt'
        }
        exp = Experiment(name='experiment 1', modellibdict=modellibdict)
        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertEqual( models[0].default_args, default_args )
        self.assertIsInstance( models[1], sklearn.DecisionTree )

        # Provide a dictionary where each key is the library and values are lists (or single vals) of models
        default_args = {'random_state': 19}
        modellibdict = {
            'scikit-learn': [
                {'model': 'rf', 'args': default_args}, 
                'decision tree'
                ]
        }
        exp = Experiment(name='experiment 1', modellibdict=modellibdict)
        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertEqual( models[0].default_args, default_args )
        self.assertIsInstance( models[1], sklearn.DecisionTree )



    def test_setName(self):
        # Check that it will set the Experiment name
        old_name = 'exp 1'
        exp = Experiment(name=old_name)
        new_name = 'cooler exp 1'
        exp.setName(new_name)
        self.assertEqual( exp.getName(), new_name )

        # Fail if incorrect types are provided
        old_name = 'exp 1'
        exp = Experiment(name=old_name)
        try:
            new_name = None
            exp.setName(new_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( new_name ) ) ))
    
        self.assertEqual( exp.getName(), old_name )

        exp = Experiment(name=old_name)
        try:
            new_name = ['hot cool name!']
            exp.setName(new_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( new_name ) ) ))
    
        self.assertEqual( exp.getName(), old_name )

        exp = Experiment(name=old_name)
        try:
            new_name = 512
            exp.setName(new_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( new_name ) ) ))
    
        self.assertEqual( exp.getName(), old_name )

        exp = Experiment(name=old_name)
        try:
            new_name = {'name':'cooler name'}
            exp.setName(new_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( new_name ) ) ))
    
        self.assertEqual( exp.getName(), old_name )

    def test_addModel(self):
        # Check that we can add models from a method call
        #  There are a few different ways we can do this...

        # Provide one library and a list (or single) of models
        exp = Experiment(name='experiment 1')
        exp.addModel(lib='sklearn', model='rf')
        exp.addModel(lib='scikit-learn', model='dt')
        
        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertIsInstance( models[1], sklearn.DecisionTree )

        # Provide a dictionary matching a desired library to a desired model
        default_args = {'random_state': 19}
        modellibdict = {
            'scikit-learn': {'model': 'random forest', 'args': default_args}
        }
        exp = Experiment(name='experiment 1')
        exp.addModel(modellibdict=modellibdict)
        exp.addModel(lib='sklearn', model='decision tree')

        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertEqual( models[0].default_args, default_args )
        self.assertIsInstance( models[1], sklearn.DecisionTree )

        # Provide a dictionary where each key is the library and values are lists (or single vals) of models
        default_args = {'random_state': 19}
        modellibdict = {
            'scikit-learn': 
                {'model': 'rf', 'args': default_args}
        }
        exp = Experiment(name='experiment 1')
        exp.addModel(modellibdict=modellibdict)
        exp.addModel(lib='sklearn', model='dt')
        
        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertEqual( models[0].default_args, default_args )
        self.assertIsInstance( models[1], sklearn.DecisionTree )

    def test_addModels(self):
        # Check that we can add models from a method call
        #  There are a few different ways we can do this...

        # Provide one library and a list (or single) of models
        exp = Experiment(name='experiment 1')
        exp.addModels(libs='sklearn', models=['rf', 'dt'])
        
        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertIsInstance( models[1], sklearn.DecisionTree )

        # Provide a list of libraries and a list of models, they must be the same length and will match one-to-one
        exp = Experiment(name='experiment 1') 
        exp.addModels(libs=['sklearn', 'scikit-learn'], models=['rf', 'decision tree'])
        
        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertIsInstance( models[1], sklearn.DecisionTree )

        # Provide a dictionary matching a desired library to a desired model
        default_args = {'random_state': 19}
        modellibdict = {
            'scikit-learn': {'model': 'random forest', 'args': default_args},
            'sklearn': 'dt'
        }
        exp = Experiment(name='experiment 1')
        exp.addModels(modellibdict=modellibdict)

        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertEqual( models[0].default_args, default_args )
        self.assertIsInstance( models[1], sklearn.DecisionTree )

        # Provide a dictionary where each key is the library and values are lists (or single vals) of models
        default_args = {'random_state': 19}
        modellibdict = {
            'scikit-learn': [
                {'model': 'rf', 'args': default_args}, 
                'decision tree'
                ]
        }
        exp = Experiment(name='experiment 1')
        exp.addModels(modellibdict=modellibdict)
        
        models = exp.getModels(aslist=True)
        self.assertEqual( len(models), 2 )
        self.assertIsInstance( models[0], sklearn.RandomForest )
        self.assertEqual( models[0].default_args, default_args )
        self.assertIsInstance( models[1], sklearn.DecisionTree )

if __name__ == '__main__':
    unittest.main()
