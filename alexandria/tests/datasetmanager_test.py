import unittest
from sklearn.datasets import load_iris, load_boston, load_diabetes


from alexandria.dataset import DatasetManager


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
        self.assertEqual( dm.datatype, type(boston) )
        self.assertEqual( dm.xlabels, ['data'] )
        self.assertEqual( dm.ylabels, 'target' )
        self.assertEqual( dm.num_classes, None )

        # Check that it handles input data correctly - pandas.DataFrame
        #   Classification dataset
        iris = load_iris(as_frame=True)
        iris = iris.frame
        target_col = 'target'
        data_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        dm = DatasetManager(dataset=iris, xlabels=data_cols, ylabels=target_col)
        self.assertEqual( dm.target_type, 'classification' )
        self.assertEqual( dm.datatype, type(iris) )
        self.assertEqual( dm.xlabels, data_cols )
        self.assertEqual( dm.ylabels, target_col )
        self.assertEqual( dm.num_classes, 3 )

        #  Regression dataset
        diabetes = load_diabetes(as_frame=True)
        diabetes = diabetes.frame
        target_col = 'target'
        data_cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        dm = DatasetManager(dataset=diabetes, xlabels=data_cols, ylabels=target_col)
        self.assertEqual( dm.target_type, 'regression' )
        self.assertEqual( dm.datatype, type(diabetes) )
        self.assertEqual( dm.xlabels, data_cols )
        self.assertEqual( dm.ylabels, target_col )
        self.assertEqual( dm.num_classes, None )

        # Fail if the xlabels or ylabels don't exist within the provided dataset

        # Fail if the dataset type is not supported

        # Fail if more than one ylabel is provided

        # Fail if datatypes for xlabel and ylabel are wrong


if __name__ == '__main__':
    unittest.main()