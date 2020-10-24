import unittest

from alexandria.models import sklearn
from models import ModelsManager


def fail(who):
    who.assertTrue( False )

class TestModelsManager(unittest.TestCase):
    def test_init(self):
        # Test instantiations
        mm = ModelsManager()

        self.assertEqual( mm.models, {} )

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
        mm.addModel(lib=lib, model=model_name, args=arguments)
        expected_id = 7579

        self.assertTrue( expected_id in mm.models )
        self.assertIsInstance( mm.models[ expected_id ], sklearn.RandomForest )
        self.assertEqual( mm.models[ expected_id ].getArguments(), arguments )
        
        arguments = {'random_state': 84, 'max_features': 'sqrt'}
        mm.addModel(lib=lib, model=model_name, args=arguments)
        expected_id = 4205
        
        self.assertTrue( expected_id in mm.models )
        self.assertIsInstance( mm.models[ expected_id ], sklearn.RandomForest )
        self.assertEqual( mm.models[ expected_id ].getArguments(), arguments )

        # Fail if we cannot find a model to add

        # Fail if types are incorrect for model, lib, and args
        fail(self)


    def test_getObjectFromName(self):
        # Check that all the models can be found using all possible paths
        mm = ModelsManager()
        self.assertEqual( mm.getObjectFromName(lib='sklearn', model='rf'), sklearn.RandomForest )
        self.assertEqual( mm.getObjectFromName(lib='scikit-learn', model='random forest'), sklearn.RandomForest )

        # If library is not provided, then search all the libraries for something with that name

        # Fail if the library or model arguments are not strings
        fail(self)

if __name__ == '__main__':
    unittest.main()