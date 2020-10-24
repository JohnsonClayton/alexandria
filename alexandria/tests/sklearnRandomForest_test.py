import unittest

from alexandria.models.sklearn import RandomForest

def fail(who):
    who.assertTrue( False )

class TestSklearnRandomForest(unittest.TestCase):
    def test_init(self):
        # Check to make sure initialization occurs correctly
        sklearn_rf = RandomForest()

        # Nothing actually needs to be set when the model is initialized!
        self.assertEqual( sklearn_rf.model, None )



if __name__ == '__main__':
    unittest.main()