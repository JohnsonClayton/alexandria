import unittest
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_breast_cancer, load_wine
import warnings

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
        #   sklearn.Bunch
        iris = load_iris()
        try:
            fake_attr = 'not_present'
            dm = DatasetManager(dataset=iris, xlabels=fake_attr, ylabels='target')

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), '{} is not an attribute of the provided dataset!'.format( fake_attr ))

        #   pandas.DataFrame
        iris = load_iris(as_frame=True)
        iris = iris.frame
        try:
            fake_cols = ['fake column 1', 'fake column 2', 'sepal length (cm)', 'fake column 3']
            dm = DatasetManager(dataset=iris, xlabels=fake_cols, ylabels='target')

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'These columns don\'t exist in the dataset: {}'.format(['fake column 1', 'fake column 2', 'fake column 3']) )

        # If the internal logic and user-input disagree, throw a warning notifying the user
        diabetes = load_diabetes(as_frame=True)
        diabetes = diabetes.frame
        target_col = 'target'
        data_cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        with warnings.catch_warnings():
            try:
                warnings.filterwarnings('error')
                dm = DatasetManager(dataset=diabetes, xlabels=data_cols, ylabels=target_col, target_type='classification')
    
                fail(self)
            except Warning as uw:
                self.assertEqual( str(uw), 'User specified classification target type, but alexandria found regression target type. Assuming the user is correct...' )

        iris = load_iris()
        with warnings.catch_warnings():
            try:
                warnings.filterwarnings('error')
                dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target', target_type='regression')
    
                fail(self)
            except Warning as uw:
                self.assertEqual( str(uw), 'User specified regression target type, but alexandria found classification target type. Assuming the user is correct...' )

        iris = load_iris()
        with warnings.catch_warnings():
            try:
                warnings.filterwarnings('error')
                dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target', num_classes=4)
    
                fail(self)
            except Warning as uw:
                self.assertEqual( str(uw), 'User specified 4 classes, but alexandria found 3 classes. Assuming user is correct...' )

        # Fail if the dataset type is not supported
        iris = load_iris()
        try:
            fake_type = 'fake type'
            dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target', target_type=fake_type)
    
            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'target_type argument must be \'regression\' or \'classification\', not {}'.format( fake_type ) )

        # Fail if target_type is not valid data type
        iris = load_iris()
        try:
            fake_type = ['list', 'of', 'vals', 512]
            dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target', target_type=fake_type)
    
            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'target type must be of string type, not {}'.format(type(fake_type)) )

        # Fail if num_classes is not an integer
        iris = load_iris()
        try:
            fake_n_classes = 10.77733
            dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target', num_classes=fake_n_classes)
    
            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'num_classes argument must be integer, not {}'.format( str( type( fake_n_classes ) ) ))

        # Fail if more than one ylabel is provided
        iris = load_iris()
        try:
            xlabels = ['data']
            ylabels = ['test1', 'test2']
            dm = DatasetManager(dataset=iris, xlabels=xlabels, ylabels=ylabels)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Multi-column target is not supported!' )

        # Fail if datatypes for xlabel and ylabel are wrong
        iris = load_iris()
        try:
            xlabels = ['data']
            ylabels = 512.39
            dm = DatasetManager(dataset=iris, xlabels=xlabels, ylabels=ylabels)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'ylabels argument must be string, not {}'.format( type(ylabels)) )

        iris = load_iris()
        try:
            xlabels = {'name':'value'}
            ylabels = ['test1', 'test2']
            dm = DatasetManager(dataset=iris, xlabels=xlabels, ylabels=ylabels)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'xlabels argument must be string or list of strings, not {}'.format( type(xlabels)) ) 

    def test_setData(self):
        # Check that we can add the data into the DatasetManager object
        iris = load_iris()
        dm = DatasetManager()

        dm.setData(iris)
        self.assertEqual( dm.dataset, iris )
        self.assertEqual( dm.xlabels, None )
        self.assertEqual( dm.xlabels, None )
        self.assertEqual( dm.target_type, None )
        self.assertEqual( dm.num_classes, None )
        xlabels = 'data'
        ylabels = 'target'
        dm.setData(iris, xlabels=xlabels, ylabels=ylabels)
        self.assertEqual( dm.dataset, iris )
        self.assertEqual( dm.xlabels, xlabels )
        self.assertEqual( dm.ylabels, ylabels )
        self.assertEqual( dm.target_type, 'classification' )
        self.assertEqual( dm.num_classes, 3 )

        # Fail if datatypes for xlabel and ylabel are wrong
        iris = load_iris()
        try:
            xlabels = ['data']
            ylabels = 512.39
            dm = DatasetManager()
            dm.setData(dataset=iris, xlabels=xlabels, ylabels=ylabels)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'ylabels argument must be string, not {}'.format( type(ylabels)) )

        iris = load_iris()
        try:
            xlabels = {'name':'value'}
            ylabels = ['test1', 'test2']
            dm = DatasetManager()
            dm.setData(dataset=iris, xlabels=xlabels, ylabels=ylabels)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'xlabels argument must be string or list of strings, not {}'.format( type(xlabels)) ) 

    def test_getX(self):
        # Check that it works as expected
        iris = load_iris()
        dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target')
        expected_X = iris.data
        actual_X = dm.getX()
        self.assertTrue( (actual_X == expected_X).any() )

        boston = load_boston()
        dm = DatasetManager(dataset=boston, xlabels='data', ylabels='target')
        expected_X = boston.data
        actual_X = dm.getX()
        self.assertTrue( (actual_X == expected_X).any() )

        wine = load_wine()
        dm = DatasetManager(dataset=wine, xlabels='data', ylabels='target')
        expected_X = wine.data
        actual_X = dm.getX()
        self.assertTrue( (actual_X == expected_X).any() )

        iris = load_iris(as_frame=True)
        iris = iris.frame
        data_cols = iris.columns[:-1]
        target_col = 'target'
        dm = DatasetManager(dataset=iris, xlabels=data_cols, ylabels=target_col)
        expected_X = iris.loc[:, iris.columns != target_col]
        actual_X = dm.getX()
        self.assertTrue( actual_X.equals( expected_X ) )

        diabetes = load_diabetes(as_frame=True)
        diabetes = diabetes.frame
        data_cols = diabetes.columns[:-1]
        target_col = 'target'
        dm = DatasetManager(dataset=diabetes, xlabels=data_cols, ylabels=target_col)
        expected_X = diabetes.loc[:, diabetes.columns != target_col]
        actual_X = dm.getX()
        self.assertTrue( actual_X.equals( expected_X ) )

    def test_gety(self):
        # Check that it works as expected
        iris = load_iris()
        dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target')
        expected_y = iris.target
        actual_y = dm.gety()
        self.assertTrue( (actual_y == expected_y).any() )

        boston = load_boston()
        dm = DatasetManager(dataset=boston, xlabels='data', ylabels='target')
        expected_y = boston.target
        actual_y = dm.gety()
        self.assertTrue( (actual_y == expected_y).any() )

        wine = load_wine()
        dm = DatasetManager(dataset=wine, xlabels='data', ylabels='target')
        expected_y = wine.target
        actual_y = dm.gety()
        self.assertTrue( (actual_y == expected_y).any() )

        iris = load_iris(as_frame=True)
        iris = iris.frame
        data_cols = iris.columns[:-1]
        target_col = 'target'
        dm = DatasetManager(dataset=iris, xlabels=data_cols, ylabels=target_col)
        expected_y = iris[target_col]
        actual_y = dm.gety()
        self.assertTrue( actual_y.equals( expected_y ) )

        diabetes = load_diabetes(as_frame=True)
        diabetes = diabetes.frame
        data_cols = diabetes.columns[:-1]
        target_col = 'target'
        dm = DatasetManager(dataset=diabetes, xlabels=data_cols, ylabels=target_col)
        expected_y = diabetes[target_col]
        actual_y = dm.gety()
        self.assertTrue( actual_y.equals( expected_y ) )

    def test_setTargetType(self):
        # If the internal logic and user-input disagree, throw a warning notifying the user
        diabetes = load_diabetes(as_frame=True)
        diabetes = diabetes.frame
        target_col = 'target'
        data_cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        with warnings.catch_warnings():
            try:
                warnings.filterwarnings('error')
                dm = DatasetManager(dataset=diabetes, xlabels=data_cols, ylabels=target_col)
                dm.setTargetType(target_type='classification')
    
                fail(self)
            except Warning as uw:
                self.assertEqual( str(uw), 'User specified classification target type, but alexandria found regression target type. Assuming the user is correct...' )

        iris = load_iris()
        with warnings.catch_warnings():
            try:
                warnings.filterwarnings('error')
                dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target')
                dm.setTargetType(target_type='regression')
    
                fail(self)
            except Warning as uw:
                self.assertEqual( str(uw), 'User specified regression target type, but alexandria found classification target type. Assuming the user is correct...' )

    def test_getTargetType(self):
        # Check that it works as expected
        iris = load_iris()
        dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target')
        expected = 'classification'
        actual = dm.getTargetType()
        self.assertEqual( actual, expected )

        bc = load_breast_cancer(as_frame=True)
        bc = bc.frame
        data_cols = bc.columns[:-1]
        target_col = 'target'
        dm = DatasetManager(dataset=bc, xlabels=data_cols, ylabels=target_col)
        expected = 'classification'
        actual = dm.getTargetType()
        self.assertEqual( actual, expected )

        wine = load_wine()
        dm = DatasetManager(dataset=wine, xlabels='data', ylabels='target')
        expected = 'classification'
        actual = dm.getTargetType()
        self.assertEqual( actual, expected )

        # Return None if the target is regression, not classification
        diabetes = load_diabetes(as_frame=True)
        diabetes = diabetes.frame
        data_cols = diabetes.columns[:-1]
        target_col = 'target'
        dm = DatasetManager(dataset=diabetes, xlabels=data_cols, ylabels=target_col)
        expected = 'regression'
        actual = dm.getTargetType()
        self.assertEqual( actual, expected )

        boston = load_boston()
        dm = DatasetManager(dataset=boston, xlabels='data', ylabels='target')
        expected = 'regression'
        actual = dm.getTargetType()
        self.assertEqual( actual, expected )

    def test_getClasses(self):
        # Check that it works as expected
        iris = load_iris()
        dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target')
        expected = ['setosa', 'versicolor', 'virginica']
        actual = dm.getClasses()
        self.assertEqual( actual, expected )

        bc = load_breast_cancer(as_frame=True)
        bc = bc.frame
        data_cols = bc.columns[:-1]
        target_col = 'target'
        dm = DatasetManager(dataset=bc, xlabels=data_cols, ylabels=target_col)
        expected = [0, 1]
        actual = dm.getClasses()
        self.assertEqual( actual, expected )

        wine = load_wine()
        dm = DatasetManager(dataset=wine, xlabels='data', ylabels='target')
        expected = ['class_0', 'class_1', 'class_2']
        actual = dm.getClasses()
        self.assertEqual( actual, expected )

        # Return None if the target is regression, not classification
        diabetes = load_diabetes(as_frame=True)
        diabetes = diabetes.frame
        data_cols = diabetes.columns[:-1]
        target_col = 'target'
        dm = DatasetManager(dataset=diabetes, xlabels=data_cols, ylabels=target_col)
        expected = None
        actual = dm.getClasses()
        self.assertEqual( actual, expected )

        boston = load_boston()
        dm = DatasetManager(dataset=boston, xlabels='data', ylabels='target')
        expected = None
        actual = dm.getClasses()
        self.assertEqual( actual, expected )


    def test_getNumClasses(self):
        # Check that it works as expected
        iris = load_iris()
        dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target')
        expected = 3
        actual = dm.getNumClasses()
        self.assertEqual( actual, expected )

        bc = load_breast_cancer(as_frame=True)
        bc = bc.frame
        data_cols = bc.columns[:-1]
        target_col = 'target'
        dm = DatasetManager(dataset=bc, xlabels=data_cols, ylabels=target_col)
        expected = 2
        actual = dm.getNumClasses()
        self.assertEqual( actual, expected )

        wine = load_wine()
        dm = DatasetManager(dataset=wine, xlabels='data', ylabels='target')
        expected = 3
        actual = dm.getNumClasses()
        self.assertEqual( actual, expected )

        # Return None if the target is regression, not classification
        diabetes = load_diabetes(as_frame=True)
        diabetes = diabetes.frame
        data_cols = diabetes.columns[:-1]
        target_col = 'target'
        dm = DatasetManager(dataset=diabetes, xlabels=data_cols, ylabels=target_col)
        expected = None
        actual = dm.getNumClasses()
        self.assertEqual( actual, expected )

        boston = load_boston()
        dm = DatasetManager(dataset=boston, xlabels='data', ylabels='target')
        expected = None
        actual = dm.getNumClasses()
        self.assertEqual( actual, expected )

    def test_setNumClasses(self):
        # Fails if non-integer is provided
        iris = load_iris()
        try:
            dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target')
            
            new_num = 3.2
            dm.setNumClasses( new_num )
        except ValueError as ve:
            self.assertEqual( str(ve), 'num_classes argument must be integer, not {}'.format(type(new_num)) )

        # Don't fail if non-integer is added when type is regression
        boston = load_boston
        dm = DatasetManager(dataset=boston, xlabels='data', ylabels='target')

        new_num = 3.2
        dm.setNumClasses( new_num )
        self.assertEqual( dm.num_classes, None )

        # Check that it throws a warning to the user when the suggested number of classes is different
        iris = load_iris()
        with warnings.catch_warnings():
            try:
                warnings.filterwarnings('error')
                dm = DatasetManager(dataset=iris, xlabels='data', ylabels='target')
                dm.setNumClasses(num_classes=4)
    
                fail(self)
            except Warning as uw:
                self.assertEqual( str(uw), 'User specified 4 classes, but alexandria found 3 classes. Assuming user is correct...' )

if __name__ == '__main__':
    unittest.main()