import unittest

from alexandria.models import sklearn
from models import ModelsManager

import warnings


def fail(who):
    who.assertTrue( False )

class TestModelsManager(unittest.TestCase):
    def test_init(self):
        # Test instantiations
        mm = ModelsManager()

        self.assertEqual( mm.models, {} )

        # Check that we can set the default library, these tests will be more verbose as
        #   we implement more libraries
        mm = ModelsManager(default_library='sklearn')
        self.assertEqual( mm.default_library, 'scikit-learn' )

    def test_addModel(self):
        mm = ModelsManager()
        self.assertEqual( mm.models, {} )

        model_name = 'random forest'
        lib = 'sklearn'
        
        arguments = {'random_state': 0}
        mm.addModel(lib=lib, model=model_name)
        expected_id = 8444

        self.assertTrue( expected_id in mm.models )
        self.assertIsInstance( mm.models[ expected_id ], sklearn.RandomForest )
        self.assertEqual( mm.models[ expected_id ].getArguments(), arguments )

        arguments = {'random_state': 19, 'max_depth': 2}
        mm.addModel(lib=lib, model=model_name, default_args=arguments)
        expected_id = 7579

        self.assertTrue( expected_id in mm.models )
        self.assertIsInstance( mm.models[ expected_id ], sklearn.RandomForest )
        self.assertEqual( mm.models[ expected_id ].getArguments(), arguments )
        
        arguments = {'random_state': 84, 'max_features': 'sqrt'}
        mm.addModel(lib=lib, model=model_name, default_args=arguments)
        expected_id = 4205
        
        self.assertTrue( expected_id in mm.models )
        self.assertIsInstance( mm.models[ expected_id ], sklearn.RandomForest )
        self.assertEqual( mm.models[ expected_id ].getArguments(), arguments )

        # Fail if we cannot find a model to add
        mm = ModelsManager()
        try:
            lib = 'unknown library'
            model = 'random forest'
            mm.addModel(lib=lib, model=model)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Library \'{}\' not found or supported. If you believe this is in error, please create an issue at https://github.com/JohnsonClayton/alexandria/issues'.format(lib))

        mm = ModelsManager()
        try:
            lib = 'sklearn'
            model = 'unknown classifier or regressor'
            mm.addModel(lib=lib, model=model)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'alexandria does not recognize model name or alias \'{}\', or this is unsupported in alexandria. If you believe this is in error, please create an issue at https://github.com/JohnsonClayton/alexandria/issues'.format( model ))

        # Raise a warning when no library is provided
        print('start warnings')
        mm = ModelsManager()
        with warnings.catch_warnings():
            try:
                warnings.filterwarnings('error')

                model = 'random forest'
                mm.addModel( model=model )

                fail(self)
            except Warning as uw:
                self.assertEqual( str(uw), 'Library unspecified, using default library: \'scikit-learn\'' )

        # Fail if types are incorrect for model, lib, and args
        # Check model type
        mm = ModelsManager()
        try:
            model = 512
            lib = 'sklearn'
            args = {'random_state': 0}
            mm.addModel(model=model, lib=lib, default_args=args)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'model argument must be string type, not {}'.format( str(type(model)) ) )

        mm = ModelsManager()
        try:
            model = {'name': 'random forest'}
            lib = 'sklearn'
            args = {'random_state': 0}
            mm.addModel(model=model, lib=lib, default_args=args)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'model argument must be string type, not {}'.format( str(type(model)) ) )

        # Check lib type
        mm = ModelsManager()
        try:
            model = 'random forest'
            lib = 512
            args = {'random_state': 0}
            mm.addModel(model=model, lib=lib, default_args=args)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'lib argument must be string type, not {}'.format( str(type(model)) ) )

        mm = ModelsManager()
        try:
            model = 'random forest'
            lib = ['sklearn', 'scikit-learn']
            args = {'random_state': 0}
            mm.addModel(model=model, lib=lib, default_args=args)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'lib argument must be string type, not {}'.format( str(type(model)) ) )

        # Check args type
        mm = ModelsManager()
        try:
            model = 'random forest'
            lib = 'scikit-learn'
            args = [{'random_state': 0}]
            mm.addModel(model=model, lib=lib, default_args=args)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Default arguments must be dictionary type, not {}'.format( type(args)))
        mm = ModelsManager()
        try:
            model = 'random forest'
            lib = 'sklearn'
            args = 512
            mm.addModel(model=model, lib=lib, default_args=args)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Default arguments must be dictionary type, not {}'.format( type(args)))

        # TO-DO: Make sure we can add a model hand-crafted by the user
        # Model object (alexandria object)
        # ML Library object

    def test_getObjectFromName(self):
        # Check that all the models can be found using all possible paths
        mm = ModelsManager()
        self.assertEqual( mm.getObjectFromName(lib='scikit-learn', model='Random Forest'), sklearn.RandomForest )

        # Return None since the lib name has not been aliased
        mm = ModelsManager()
        self.assertEqual( mm.getObjectFromName(lib='sklearn', model='Random Forest'), None )
        self.assertEqual( mm.getObjectFromName(lib='scikit-learn', model='random forest'), None )

        # Fail if the library or model arguments are not strings
        mm = ModelsManager()
        try:
            lib = 512
            model = 'random forest'
            mm.getObjectFromName(lib=lib, model=model)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'lib argument must be string type, not {}'.format( str( type(lib) ) ) )

        mm = ModelsManager()
        try:
            lib = [512, 512, 512]
            model = 'random forest'
            mm.getObjectFromName(lib=lib, model=model)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'lib argument must be string type, not {}'.format( str( type(lib) ) ) )

        mm = ModelsManager()
        try:
            lib = 'sklearn'
            model = {'random forest':True}
            mm.getObjectFromName(lib=lib, model=model)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'model argument must be string type, not {}'.format( str( type(lib) ) ) )

        mm = ModelsManager()
        try:
            lib = 'scikit-learn'
            model = ['random forest']
            mm.getObjectFromName(lib=lib, model=model)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'model argument must be string type, not {}'.format( str( type(lib) ) ) )

    def test_addModels(self):
        mm = ModelsManager()

        self.assertEqual( mm.getNumModels(), 0 )

        modelstoadd = {
            'sklearn': [
                {
                    'model': 'random forest',
                    'args': {
                        'random_state': 19,
                        'n_estimators': 250
                    }
                },
                'decision tree'
            ]
        }
        mm.addModels(modelstoadd)

        self.assertEqual( mm.getNumModels(), 2 )

if __name__ == '__main__':
    unittest.main()