import unittest

from alexandria.experiment import Experiment
from alexandria.dataset import DatasetManager
from alexandria.models import ModelsManager, sklearn

from sklearn.datasets import load_iris, load_diabetes


def fail(who):
    who.assertTrue( False )

class TestExperiment(unittest.TestCase):
    def test_init(self):
        # Check that the initializations occur correctly
        name = 'experiment 1'
        exp = Experiment(name=name)

        self.assertEqual( exp.name, name )
        self.assertIsInstance( exp.dm, DatasetManager )
        self.assertIsInstance( exp.mm, ModelsManager )

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


        # Check that we can add a dataset to the datasetmanager from the constructor
        iris = load_iris()
        exp = Experiment(name='experiment 1', dataset=iris, xlabels='data', ylabels='target')
        self.assertEqual( exp.dm.dataset, iris )
        self.assertEqual( exp.dm.target_type, 'classification' )

        iris = load_iris(as_frame=True).frame
        data_cols = iris.columns[:-1]
        target_col = 'target'
        exp = Experiment(name='experiment 1', dataset=iris, xlabels=data_cols, ylabels=target_col)
        self.assertTrue( iris.equals( exp.dm.dataset ) )
        self.assertEqual( exp.dm.target_type, 'classification' )

        diabetes = load_diabetes()
        exp = Experiment(name='experiment 1', dataset=diabetes, xlabels='data', ylabels='target')
        self.assertEqual( exp.dm.dataset, diabetes )
        self.assertEqual( exp.dm.target_type, 'regression' )

        diabetes = load_diabetes(as_frame=True).frame
        data_cols = diabetes.columns[:-1]
        target_col = 'target'
        exp = Experiment(name='experiment 1', dataset=diabetes, xlabels=data_cols, ylabels=target_col)
        self.assertTrue( diabetes.equals( exp.dm.dataset ) )
        self.assertEqual( exp.dm.target_type, 'regression' )

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
