import unittest

from alexandria.models.sklearn import RandomForest

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, load_boston, load_diabetes, load_wine
from sklearn.metrics import accuracy_score, recall_score, precision_score, r2_score
from sklearn.model_selection import train_test_split


def fail(who):
    who.assertTrue( False )

class TestSklearnRandomForest(unittest.TestCase):
    def test_init(self):
        # Check to make sure initialization occurs correctly
        # Nothing actually needs to be set when the model is initialized!
        rf = RandomForest()

        self.assertEqual( rf.model, None )
        self.assertEqual( rf.lib, 'sklearn' )
        self.assertEqual( rf.model_name, 'random forest' )

        # You can change the name of the model
        rf = RandomForest(model_name='random forest 1')

        self.assertEqual( rf.model, None )
        self.assertEqual( rf.lib, 'sklearn' )
        self.assertEqual( rf.model_name, 'random forest 1' )

        # But you cannot change it to some non-string type!
        try:
            model_name = ['random forest in an array!', 'again!', 'and another!']
            rf = RandomForest(model_name=model_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Model name must be string, not {}'.format( type(model_name) ) )

        try:
            model_name = 512
            rf = RandomForest(model_name=model_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Model name must be string, not {}'.format( type(model_name) ) )

        try:
            model_name = {'model_name':'random forest'}
            rf = RandomForest(model_name=model_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Model name must be string, not {}'.format( type(model_name) ) )

        # Check that you can set the experiment type
        exp_type = 'classification'
        rf = RandomForest(exp_type=exp_type)
        self.assertEqual( rf.exp_type, exp_type )

        exp_type = 'regression'
        rf = RandomForest(exp_type=exp_type)
        self.assertEqual( rf.exp_type, exp_type )

        # Experiment type must be a string
        try:
            exp_type = ['regression']
            rf = RandomForest(exp_type=exp_type)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment type argument must be string type, not {}'.format( type(exp_type) ) )

        try:
            exp_type = {'regression': 'test'}
            rf = RandomForest(exp_type=exp_type)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment type argument must be string type, not {}'.format( type(exp_type) ) )

        try:
            exp_type = 512
            rf = RandomForest(exp_type=exp_type)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment type argument must be string type, not {}'.format( type(exp_type) ) )

        # Experiment type must be 'classification' or 'regression'
        try:
            exp_type = 'not regression'
            rf = RandomForest(exp_type=exp_type)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment type argument must be \'classification\' or \'regression\', not {}'.format( exp_type ))


        # Check that you can set the default arguments for the model
        default_args = {'random_state': 5, 'n_estimators': 200, 'verbose': 1}
        rf = RandomForest(default_args=default_args)

        self.assertEqual( rf.default_args, default_args )

        # Make sure that the type of the keys in default args cannot be anything except strings
        try:
            default_args = {5: 5, 'n_estimators': 200, 'verbose': 1}
            rf = RandomForest(default_args=default_args)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'All argument names must be strings!' )

        try:
            default_args = {'random_state': 5, 'n_estimators': 200, 'verbose': 1, 6: {'test': 5}}
            rf = RandomForest(default_args=default_args)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'All argument names must be strings!' )

        # You can add default arguments to the list that won't actually work, but we don't build the models right now so no errors
        #  should be thrown
        default_args = {'non-existant argument name': 5, 'whatever': 200, 'verbose': 1}
        rf = RandomForest(default_args=default_args)

        self.assertEqual( rf.default_args, default_args )


    def test_predict(self):
        # Check to make sure that the model will train as expected with sklearn.Bunch objects
        data = load_iris()
        X_train, y_train = data.data[:120], data.target[:120]
        X_test, y_test = data.data[120:], data.target[120:]


        # All '2' variables are the baseline test and what we should match up with
        default_args = {'random_state': 19}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestClassifier(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict(X_test)
        preds2 = rf2.predict(X_test)

        self.assertTrue( (preds1 == preds2).all() )

        data = load_boston()
        X_train, y_train = data.data[:120], data.target[:120]
        X_test, y_test = data.data[120:], data.target[120:]


        # All '2' variables are the baseline test and what we should match up with
        default_args = {'random_state': 30}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict(X_test)
        preds2 = rf2.predict(X_test)

        self.assertTrue( (preds1 == preds2).all() )

        data = load_diabetes()
        X_train, y_train = data.data[:120], data.target[:120]
        X_test, y_test = data.data[120:], data.target[120:]


        # All '2' variables are the baseline test and what we should match up with
        default_args = {'random_state': 15}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict(X_test)
        preds2 = rf2.predict(X_test)

        self.assertTrue( (preds1 == preds2).all() )

        data = load_wine()
        X_train, y_train = data.data[:120], data.target[:120]
        X_test, y_test = data.data[120:], data.target[120:]


        # All '2' variables are the baseline test and what we should match up with
        default_args = {'random_state': 90, 'max_depth': 3}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestClassifier(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict(X_test)
        preds2 = rf2.predict(X_test)

        self.assertTrue( (preds1 == preds2).all() )

        # Check to make sure that the model will train as expected with sklearn.Bunch objects
        data = load_iris(as_frame=True)
        data = data.frame
        X = data.loc[:, data.columns != 'target']
        y = data['target']
        X_train, y_train = X.iloc[:120], y.iloc[:120]
        X_test, y_test = X.iloc[120:], y.iloc[120:]
        
        default_args = {'random_state': 19, 'n_estimators' : 200, 'max_features': 'auto'}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestClassifier(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict(X_test)
        preds2 = rf2.predict(X_test)

        self.assertTrue( (preds1 == preds2).all() )

        data = load_diabetes(as_frame=True)
        data = data.frame
        X = data.loc[:, data.columns != 'target']
        y = data['target']
        X_train, y_train = X.iloc[:120], y.iloc[:120]
        X_test, y_test = X.iloc[120:], y.iloc[120:]
        
        default_args = {'random_state': 19, 'n_estimators' : 200, 'min_samples_split': 4}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict(X_test)
        preds2 = rf2.predict(X_test)

        self.assertTrue( (preds1 == preds2).all() )

        data = load_wine(as_frame=True)
        data = data.frame
        X = data.loc[:, data.columns != 'target']
        y = data['target']
        X_train, y_train = X.iloc[:120], y.iloc[:120]
        X_test, y_test = X.iloc[120:], y.iloc[120:]
        
        default_args = {'random_state': 19, 'n_estimators' : 200, 'criterion': 'gini'}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestClassifier(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict(X_test)
        preds2 = rf2.predict(X_test)

        self.assertTrue( (preds1 == preds2).all() )

    def test_predict_proba(self):
        # Check to make sure that the model will train as expected with sklearn.Bunch objects
        data = load_iris()
        X_train, y_train = data.data[:120], data.target[:120]
        X_test, y_test = data.data[120:], data.target[120:]


        # All '2' variables are the baseline test and what we should match up with
        default_args = {'random_state': 19}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestClassifier(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict_proba(X_test)
        preds2 = rf2.predict_proba(X_test)

        self.assertTrue( (preds1 == preds2).all() )

        data = load_boston()
        X_train, y_train = data.data[:120], data.target[:120]
        X_test, y_test = data.data[120:], data.target[120:]


        # All '2' variables are the baseline test and what we should match up with
        default_args = {'random_state': 30}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)

        try: 
            preds1 = rf1.predict_proba(X_test)

            # The following line is not implemented in sklearn, however I've included to match the others
            #preds2 = rf2.predict_proba(X_test)
        except NotImplementedError as err:
            self.assertEqual( str(err), 'The \'predict_proba\' method is not implemented for regression problems (this is an scikit-learn issue, not an alexandria issue!)' )


        data = load_diabetes()
        X_train, y_train = data.data[:120], data.target[:120]
        X_test, y_test = data.data[120:], data.target[120:]


        # All '2' variables are the baseline test and what we should match up with
        default_args = {'random_state': 15}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)

        try: 
            preds1 = rf1.predict_proba(X_test)

            # The following line is not implemented in sklearn, however I've included to match the others
            #preds2 = rf2.predict_proba(X_test)
        except NotImplementedError as err:
            self.assertEqual( str(err), 'The \'predict_proba\' method is not implemented for regression problems (this is an scikit-learn issue, not an alexandria issue!)' )

        data = load_wine()
        X_train, y_train = data.data[:120], data.target[:120]
        X_test, y_test = data.data[120:], data.target[120:]


        # All '2' variables are the baseline test and what we should match up with
        default_args = {'random_state': 90, 'max_depth': 3}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestClassifier(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict_proba(X_test)
        preds2 = rf2.predict_proba(X_test)

        self.assertTrue( (preds1 == preds2).all() )

        # Check to make sure that the model will train as expected with sklearn.Bunch objects
        data = load_iris(as_frame=True)
        data = data.frame
        X = data.loc[:, data.columns != 'target']
        y = data['target']
        X_train, y_train = X.iloc[:120], y.iloc[:120]
        X_test, y_test = X.iloc[120:], y.iloc[120:]
        
        default_args = {'random_state': 19, 'n_estimators' : 200, 'max_features': 'auto'}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestClassifier(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict_proba(X_test)
        preds2 = rf2.predict_proba(X_test)

        self.assertTrue( (preds1 == preds2).all() )

        data = load_diabetes(as_frame=True)
        data = data.frame
        X = data.loc[:, data.columns != 'target']
        y = data['target']
        X_train, y_train = X.iloc[:120], y.iloc[:120]
        X_test, y_test = X.iloc[120:], y.iloc[120:]
        
        default_args = {'random_state': 19, 'n_estimators' : 200, 'min_samples_split': 4}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)

        try: 
            preds1 = rf1.predict_proba(X_test)

            # The following line is not implemented in sklearn, however I've included to match the others
            #preds2 = rf2.predict_proba(X_test)
        except NotImplementedError as err:
            self.assertEqual( str(err), 'The \'predict_proba\' method is not implemented for regression problems (this is an scikit-learn issue, not an alexandria issue!)' )

        data = load_wine(as_frame=True)
        data = data.frame
        X = data.loc[:, data.columns != 'target']
        y = data['target']
        X_train, y_train = X.iloc[:120], y.iloc[:120]
        X_test, y_test = X.iloc[120:], y.iloc[120:]
        
        default_args = {'random_state': 19, 'n_estimators' : 200, 'criterion': 'gini'}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestClassifier(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)

        preds1 = rf1.predict_proba(X_test)
        preds2 = rf2.predict_proba(X_test)

        self.assertTrue( (preds1 == preds2).all() )

    def test_setExperimentType(self):
        # Test that it works as expected
        exp_type = 'classification'
        rf = RandomForest()
        rf.setExperimentType(exp_type=exp_type)
        self.assertEqual( rf.exp_type, exp_type )

        exp_type = 'regression'
        rf = RandomForest()
        rf.setExperimentType(exp_type=exp_type)        
        self.assertEqual( rf.exp_type, exp_type )

        # Experiment type must be a string
        try:
            exp_type = ['regression']
            rf = RandomForest()
            rf.setExperimentType(exp_type=exp_type)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment type argument must be string type, not {}'.format( type(exp_type) ) )

        try:
            exp_type = {'regression': 'test'}
            rf = RandomForest()
            rf.setExperimentType(exp_type=exp_type)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment type argument must be string type, not {}'.format( type(exp_type) ) )

        try:
            exp_type = 512
            rf = RandomForest()
            rf.setExperimentType(exp_type=exp_type)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment type argument must be string type, not {}'.format( type(exp_type) ) )

        # Experiment type must be 'classification' or 'regression'
        try:
            exp_type = 'not regression'
            rf = RandomForest()
            rf.setExperimentType(exp_type=exp_type)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment type argument must be \'classification\' or \'regression\', not {}'.format( exp_type ))

    def test_eval(self):
        # Check to make sure that the model will evaluate as expected with sklearn.Bunch objects
        
        # Classification
        #  sklearn.Bunch
        data = load_iris()
        X_train, y_train = data.data[:120], data.target[:120]
        X_test, y_test = data.data[120:], data.target[120:]


        # All '2' variables are the baseline test and what we should match up with
        default_args = {'random_state': 19}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestClassifier(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)

        rf1.eval(X_test, y_test, metrics='acc')
        actual_acc = rf1.getMetric('acc')
        preds = rf2.predict(X_test)
        expected_acc = accuracy_score(y_test, preds)
        self.assertEqual( actual_acc, expected_acc )

        # Error if r-squared is wanted in classification problem
        # Recall
        default_args = {'random_state': 30}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)
        try:
            rf1.eval(X_test, y_test, metrics='r2')

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'cannot use R2 metric for classification problem!' )
    
        #   pandas.DataFrame
        data = load_iris(as_frame=True).frame
        data_cols = data.columns[:-1]
        target_col = 'target'
        X_train, X_test, y_train, y_test = train_test_split(data[data_cols], data[target_col], train_size=0.8, random_state=0)

        default_args = {'random_state': 19}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestClassifier(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)

        rf1.eval(X_test, y_test, metrics=['acc', 'precision', 'recall'])
        actual_acc = rf1.getMetric('acc')
        actual_prec = rf1.getMetric('prec')
        actual_rec = rf1.getMetric('rec')
        preds = rf2.predict(X_test)
        expected_acc = accuracy_score(y_test, preds)
        expected_prec = precision_score(y_test, preds, average='weighted')
        expected_rec = recall_score(y_test, preds, average='weighted')
        self.assertEqual( actual_acc, expected_acc )
        self.assertEqual( actual_prec, expected_prec )
        self.assertEqual( actual_rec, expected_rec )

        # Error if r-squared is wanted in classification problem
        # Recall
        default_args = {'random_state': 30}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='classification')
        rf2.fit(X_train, y_train)
        try:
            rf1.eval(X_test, y_test, metrics='r2')

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'cannot use R2 metric for classification problem!' )


        # Regression
        #  sklearn.Bunch
        data = load_boston()
        X_train, y_train = data.data[:120], data.target[:120]
        X_test, y_test = data.data[120:], data.target[120:]

        default_args = {'random_state': 30}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)

        rf1.eval(X_test, y_test, metrics=['r2'])
        actual_r2 = rf1.getMetric('r2')
        preds = rf2.predict(X_test)
        expected_r2 = r2_score(y_test, preds)
        self.assertEqual( actual_r2, expected_r2 )

        # Error if accuracy, recall, etc is wanted in a regression problem
        # Recall
        default_args = {'random_state': 30}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)
        try:
            rf1.eval(X_test, y_test, metrics='recall')

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'cannot use Recall metric for regression problem!' )
        
        # Accuracy
        default_args = {'random_state': 30}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)
        try:
            rf1.eval(X_test, y_test, metrics='accuracy')

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'cannot use Accuracy metric for regression problem!' )

        # Precision
        default_args = {'random_state': 30}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)
        try:
            rf1.eval(X_test, y_test, metrics='prec')

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'cannot use Precision metric for regression problem!' )


        #   pandas.DataFrame
        data = load_diabetes(as_frame=True).frame
        data_cols = data.columns[:-1]
        target_col = 'target'
        X_train, X_test, y_train, y_test = train_test_split(data[data_cols], data[target_col], train_size=0.8, random_state=0)

        default_args = {'random_state': 30}
        rf1 = RandomForest(default_args=default_args)
        rf2 = RandomForestRegressor(**default_args)

        rf1.train(X_train, y_train, exp_type='regression')
        rf2.fit(X_train, y_train)

        rf1.eval(X_test, y_test, metrics=['r2'])
        actual_r2 = rf1.getMetric('r2')
        preds = rf2.predict(X_test)
        expected_r2 = r2_score(y_test, preds)
        self.assertEqual( actual_r2, expected_r2 )


if __name__ == '__main__':
    unittest.main()