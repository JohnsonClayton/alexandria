import unittest

from alexandria.dataset import DatasetManager

from sklearn.datasets import load_iris, load_boston

def fail(who):
    who.assertTrue( False )
class TestDatasetManager(unittest.TestCase):
    def test_init(self):
        # Check that initializations occur correctly
        dm = DatasetManager()

        self.assertEqual( dm.target_type, None )
        self.assertEqual( dm.dataset, None )
        self.assertEqual( dm.datatype, None )
        self.assertEqual( dm.xlabels, None )
        self.assertEqual( dm.ylabels, None )
        self.assertEqual( dm.num_classes, None )

        # Check that it handles input data correctly - sklearn.Bunch
        #   Classification dataset
        iris = load_iris()
        dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target')
        self.assertEqual( dm.target_type, 'classification' )
        self.assertEqual( dm.dataset, iris )
        self.assertEqual( dm.datatype, type(iris) )
        self.assertEqual( dm.xlabels, 'data' )
        self.assertEqual( dm.ylabels, 'target' )
        self.assertEqual( dm.num_classes, 3 )

        #  Regression dataset
        boston = load_boston()
        dm = DatasetManager(dataset=boston, xlabels=['data'], ylabels='target')
        self.assertEqual( dm.target_type, 'regression' )
        self.assertEqual( dm.dataset, boston )
        self.assertEqual( dm.datatype, type(iris) )
        self.assertEqual( dm.xlabels, ['data'] )
        self.assertEqual( dm.ylabels, 'target' )
        self.assertEqual( dm.num_classes, None )

        # Check that it handles input data correctly - pandas.DataFrame

        # Fail if the xlabels or ylabels don't exist within the provided dataset

        # Fail if the dataset type is not supported

        # Fail if more than one ylabel is provided

        # Fail if datatypes for xlabel and ylabel are wrong

        # 


if __name__ == '__main__':
    unittest.main()