import unittest

from alexandria.metrics import SklearnMetricsGenerator

def fail(who):
    who.assertTrue( False )

class TestSklearnMetricsGenerator(unittest.TestCase):
    def test_init(self):
        # Test to make sure we can hand over the correct library
        mg = SklearnMetricsGenerator()
        self.assertEqual( mg.lib, 'scikit-learn' )
        

    def test_getValue(self):
        # Test sklearn example provided by accuracy_score documentation
        mg = SklearnMetricsGenerator()

        y_true = [0, 1, 2, 3]
        y_pred = [0, 2, 1, 3]

        expected_acc = 0.5
        actual_acc = mg.getValue( y_true, y_pred, mtype='accuracy', exp_type='classification')
        self.assertEqual( actual_acc, expected_acc )
        
        expected_count = 2
        actual_count = mg.getValue( y_true, y_pred, mtype='acc', exp_type='classification', normalize=False )
        self.assertEqual( actual_count, expected_count )

        # Test sklearn recall
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]

        expected_rec = 1 / 3
        actual_rec = mg.getValue( y_true, y_pred, mtype='rec', exp_type='classification' )
        self.assertEqual( actual_rec, expected_rec )

        # Test sklearn precision
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 0, 1]

        expected_rec = 2 / 9
        actual_rec = mg.getValue( y_true, y_pred, mtype='prec', exp_type='classification' )
        self.assertEqual( actual_rec, expected_rec )

        # As we add more ML libraries to the mix, add more tests here

        # TO-DO: Failure cases?

if __name__ == '__main__':
    unittest.main()