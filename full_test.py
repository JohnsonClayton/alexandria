from experiments import Experiments, Experiment
from utils import Helper
from model import Model
from metric import Metric, MetricsManager

from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.datasets import load_iris, load_boston, make_classification

import sys

import unittest

class TestExperiment(unittest.TestCase):
	def test_init(self):
		experiment = Experiment()
		self.assertEqual( experiment.getName(), 'unnamed_experiment' )
		self.assertEqual( experiment.getModels(), {} )

	def test_setRandomState(self):
		experiment = Experiment(name='experiment 1', models=['rf', 'dt'], exp_type='regression')
		models = experiment.getModels()
		self.assertEqual( models['rf'].getConstructorArgs()['random_state'], 0 )
		self.assertEqual( models['dt'].getConstructorArgs()['random_state'], 0 )

		experiment.setRandomState(1)

		models = experiment.getModels()
		self.assertEqual( models['rf'].getConstructorArgs()['random_state'], 1 )
		self.assertEqual( models['dt'].getConstructorArgs()['random_state'], 1 )



	def test_isValidExperimentType(self):
		experiment = Experiment()
		experiment_type = 'classification'
		self.assertEqual( experiment.isValidExperimentType(experiment_type), True )

		experiment_type = 'bloogus!'
		self.assertEqual( experiment.isValidExperimentType(experiment_type), False )

	def test_addModel(self):
		experiment = Experiment('experiment_1')
		experiment.addModel('rf')
		dt_model = Model('decision tree', DecisionTreeClassifier, {'random_state': 0, 'max_depth': 2})
		experiment.addModel(dt_model)

		models_dict = experiment.getModels()

		#print(models_dict)
		self.assertTrue( 'rf' in models_dict )

	def test_setExperimentType(self):
		experiment = Experiment()
		experiment_type = 'bloogus!'
		try:
			experiment.setExperimentType(experiment_type)

			# This should never run!
			self.assertEqual(0, 1)
		except ValueError as ve:
			self.assertEqual( str(ve), 'Experiment must be \'regression\' or \'classification\', cannot be bloogus!' )

		experiment_type = 'classification'
		experiment.setExperimentType(experiment_type)
	   
		self.assertEqual( experiment.getExperimentType(), experiment_type )

	def test_run(self):
		experiment = Experiment('exp1', models=['rf', 'dt'], exp_type='classification')
		iris = load_iris()

		experiment.train(iris.data[:120], iris.target[:120])

		try:
			experiment.train(iris.data[:120], iris.target[:12])

			# This should never be executed!
			self.assertEqual( 0, 1 )
		except ValueError as ve:
			self.assertEqual( str(ve), 'Data and target provided to \'exp1\' must be same length:\n\tlen of data: 120\n\tlen of target: 12' )

		experiment = Experiment(models=['rf', 'dt'], exp_type='classification')
		experiment.train(iris.data[:120], iris.target[:120])

		try:
			experiment.train(iris.data[:47], iris.target)

			# This should never be executed!
			self.assertEqual( 0, 1 )
		except ValueError as ve:
			self.assertEqual( str(ve), 'Data and target provided to \'unnamed_experiment\' must be same length:\n\tlen of data: 47\n\tlen of target: 150' )

	def test_predict(self):
		experiment = Experiment('exp1', models=['rf', 'dt'], exp_type='classification')
		iris = load_iris()

		experiment.train(iris.data[:120], iris.target[:120])
		predictions = experiment.predict(iris.data[120:])
		actual = {
				'rf':
					[2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 
				'dt':
					[2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}
		self.assertEqual( predictions, actual )

	def test_eval(self):
		iris = load_iris()
		X_train, y_train = iris.data[:120], iris.target[:120]
		X_test, y_test = iris.data[120:], iris.target[120:]

		experiment = Experiment('exp1', models=['rf', 'dt'], exp_type='classification')

		experiment.train(X_train, y_train)
		metrics = experiment.eval(X_test, y_test, metrics=['acc'])

		rf_metric = Metric('rf')
		rf_metric.addValue('acc', 0.7666666666666667)

		dt_metric = Metric('dt')
		dt_metric.addValue('acc', 0.8)

		expected_metrics = [rf_metric, dt_metric]

		self.assertEqual( metrics, expected_metrics )

		metrics = experiment.eval(X_test, y_test, metrics='acc')
		expected_metrics = [rf_metric, dt_metric]

		self.assertEqual( metrics, expected_metrics )

		# 10-fold cross validation test

		X = iris.data
		y = iris.target

		experiment = Experiment('exp1', models=['rf', 'dt'], exp_type='classification')

		experiment.train(X, y, cv=True, n_folds=10, shuffle=True, metrics=['acc', 'rec', 'prec'])
		experiment.setRandomState(0)

		rf_metric = Metric('rf')
		rf_metric.addValue('acc_avg', 0.9400000000000001)
		rf_metric.addValue('acc_std', 0.04666666666666665)
		rf_metric.addValue('rec_avg', 0.9400000000000001)
		rf_metric.addValue('rec_std', 0.04666666666666665)
		rf_metric.addValue('prec_avg', 0.9531746031746031)
		rf_metric.addValue('prec_std', 0.03412698412698411)

		dt_metric = Metric('dt')
		dt_metric.addValue('acc_avg', 0.9400000000000001)
		dt_metric.addValue('acc_std', 0.05537749241945382)
		dt_metric.addValue('rec_avg', 0.9400000000000001)
		dt_metric.addValue('rec_std', 0.05537749241945382)
		dt_metric.addValue('prec_avg', 0.9541666666666668)
		dt_metric.addValue('prec_std', 0.036244412085277455)

		expected_metrics = {'rf': rf_metric, 'dt': dt_metric}
		metrics = experiment.summarizeMetrics()

		self.assertEqual( metrics, expected_metrics )

	def test_calcAcc(self):
		model = Model()

		pred = [0, 0, 1, 0, 0, 1]
		act  = [1, 0, 1, 0, 1, 0]

		accuracy_expected = 0.5
		accuracy_calculated = model.calcAcc(act, pred)
		self.assertEqual( accuracy_calculated, accuracy_expected )

	def test_calcRec(self):
		model = Model()

		pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		act  = [1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]

		recall_expected = 0.75
		recall_calculated = model.calcRec(act, pred)
		self.assertEqual( recall_calculated, recall_expected )

	def test_calcPrec(self):
		model = Model()

		pred = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		act  = [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]

		precision_expected = 0.8235294117647058
		precision_calculated = model.calcPrec(act, pred)
		self.assertEqual( precision_calculated, precision_expected )

class TestMetric(unittest.TestCase):
	def test_init(self):
		metric = Metric('random forest')
		self.assertEqual( metric.getName(), 'random forest' )
		self.assertEqual( metric.getValues(), {} )

	def test_addValue(self):
		metric = Metric('random forest')
		metric.addValue('accuracy', 0.75)
		self.assertEqual( metric.getValue('accuracy'), 0.75 )

		metric.addValue('recall', 0.9)
		metric.addValue('precision', 0.2)

		self.assertEqual( metric.getValue('accuracy'), 0.75 )
		self.assertEqual( metric.getValue('recall'), 0.9 )
		self.assertEqual( metric.getValue('precision'), 0.2 )

		metric.addValue('accuracy', 0.99)
		metric.addValue('accuracy', 0.3)
		metric.addValue('accuracy', 0.25672837482)

		self.assertEqual( metric.getValue('accuracy'), 0.25672837482 )

		try:
			metric.addValue('accuracy', 'hello!')

			# This should never run
			self.assertEqual( 0, 1 )
		except ValueError as ve:
			self.assertEqual( str(ve), 'Metric.addValue must have \'m_type\' as string and \'value\' as integer or floating point number instead of type(m_type) => <class \'str\'> and type(value) => <class \'str\'>')

		try:
			metric.addValue(['accuracy'], 0.9)

			# This should never run
			self.assertEqual( 0, 1 )
		except ValueError as ve:
			self.assertEqual( str(ve), 'Metric.addValue must have \'m_type\' as string and \'value\' as integer or floating point number instead of type(m_type) => <class \'list\'> and type(value) => <class \'float\'>')

	def test_getMetricWithMeasure(self):
		metric = Metric('random forest')
		metric.addValue('accuracy', 0.75)
		metric.addValue('recall', 0.9)
		metric.addValue('precision', 0.2)
		metric.addValue('acc', 0.98)
		metric.addValue('test_value', 500)

		actual_result = metric.getMetricWithMeasure()
		expected_result = {'accuracy': 0.75, 'recall': 0.9, 'precision': 0.2, 'acc': 0.98, 'test_value': 500} 
		self.assertEqual( actual_result, expected_result )

		actual_result = metric.getMetricWithMeasure('acc')
		expected_result = {'acc': 0.98}
		self.assertEqual( actual_result, expected_result )

		actual_result = metric.getMetricWithMeasure(['recall', 'acc', 'test_value'])
		expected_result = {'recall': 0.9, 'acc': 0.98, 'test_value': 500} 
		self.assertEqual( actual_result, expected_result )

		try:
			actual_result = metric.getMetricWithMeasure( 480 )

			# This should never be run
			self.assertEqual( 0, 1 )
		except ValueError as ve:
			self.assertEqual( str(ve), 'Metric.getMetricWithMeasure must be given either a string of metric or array of strings of metrics desired.' )

	def test_equals(self):
		metric1 = Metric('some name')
		metric1.addValue('acc', 0.98)
		metric1.addValue('recall', 0.72)

		metric2 = Metric('some other name')
		metric2.addValue('acc', 0.98)
		metric2.addValue('recall', 0.72)

		self.assertEqual( metric1, metric2 )

		metric1 = Metric('some name')
		metric1.addValue('acc', 0.98)
		metric1.addValue('recall', 0.72)

		metric2 = Metric('some other name')
		metric2.addValue('acc', 0.20)
		metric2.addValue('recall', 0.5)

		self.assertNotEqual( metric1, metric2 )
		

class TestMetricsManager(unittest.TestCase):
	def test_init(self):
		metrics_manager = MetricsManager()
		self.assertListEqual( metrics_manager.metrics_list, [] )

	def test_getMetrics(self):
		metrics_manager = MetricsManager()

		metric1 = Metric('metric1')
		metric1.addValue('acc', 0.48)
		metric1.addValue('recall', 0.3)
		metric1.addValue('precision', 0.7)

		metric2 = Metric('metric2')
		metric2.addValue('acc', 0.98)
		metric2.addValue('precision', 0.42)

		metric3 = Metric('metric3')
		metric3.addValue('recall', 0.34)
		metric3.addValue('precision', 0)

		metric4 = Metric('metric4')
		metric4.addValue('acc', 0.8)
		metric4.addValue('recall', 0.35)

		metric5 = Metric('metric5')
		metric5.addValue('acc', 0.14)
		metric5.addValue('recall', 0.03)
		metric5.addValue('precision', 0.71)

		metric6 = Metric('metric6')
		metric6.addValue('acc', 0.92)
		metric6.addValue('recall', 1.0)
		metric6.addValue('precision', 0.63)

		metrics_manager.addMetric(metric1)
		metrics_manager.addMetric(metric2)
		metrics_manager.addMetric(metric3)
		metrics_manager.addMetric(metric4)
		metrics_manager.addMetric(metric5)
		metrics_manager.addMetric(metric6)

		actual_result_metrics = metrics_manager.getMetrics()
		expected_result_metrics = [metric1, metric2, metric3, metric4, metric5, metric6]
		self.assertListEqual( actual_result_metrics, expected_result_metrics )

		
		metrics_manager = MetricsManager()

		metric1 = Metric('metric1')
		metric1.addValue('acc', 0.48)
		metric1.addValue('recall', 0.3)
		metric1.addValue('precision', 0.7)

		metric2 = Metric('metric2')
		metric2.addValue('acc', 0.98)
		metric2.addValue('precision', 0.42)

		metric3 = Metric('metric3')
		metric3.addValue('recall', 0.34)
		metric3.addValue('precision', 0)

		metric4 = Metric('metric4')
		metric4.addValue('acc', 0.8)
		metric4.addValue('recall', 0.35)

		metric5 = Metric('metric5')
		metric5.addValue('acc', 0.14)
		metric5.addValue('recall', 0.03)
		metric5.addValue('precision', 0.71)

		metric6 = Metric('metric6')
		metric6.addValue('acc', 0.92)
		metric6.addValue('recall', 1.0)
		metric6.addValue('precision', 0.63)

		metrics_manager.addMetric(metric1)
		metrics_manager.addMetric(metric2)
		metrics_manager.addMetric(metric3)
		metrics_manager.addMetric(metric4)
		metrics_manager.addMetric(metric5)
		metrics_manager.addMetric(metric6)

		# Metric 3 will be different!
		metric3 = Metric('metric3')
		metric3.addValue('recall', 0.68)
		metric3.addValue('precision', 0.74)

		actual_result_metrics = metrics_manager.getMetrics()
		expected_result_metrics = [metric1, metric2, metric3, metric4, metric5, metric6]
		self.assertNotEqual( actual_result_metrics, expected_result_metrics )


class TestExperiments(unittest.TestCase):
	def test_init(self):
		experiments = Experiments()
		self.assertEqual( experiments.getNumOfExperiments(), 0 )
		self.assertEqual( experiments.getExperiments(), {} )

		try:
			experiments.runAllExperiments()

			# This should never be reached in the case of a successful execution
			self.assertEqual( 0, 1)
		except ValueError as ve:
			self.assertEqual( str(ve), 'Experiments object has no models to run!')

		try:
			experiments.addExperiment('random forest')

			# This should never be reached in the case of a successful execution
			self.assertEqual( 0, 1)
		except ValueError as ve:
			self.assertEqual( str(ve), 'Object must be Experiment object: random forest')

		try:
			experiments.addExperiment( Experiment(1) )

			
			# This should never be reached in the case of a successful execution
			self.assertEqual( 0, 1)
		except ValueError as ve:
			self.assertEqual( str(ve), 'Experiment name attribute must be string: 1' )
		self.assertEqual( experiments.getNumOfExperiments(), 0 )

		experiments.addExperiment( Experiment('1') )
		experiments.addExperiment( Experiment('2') )
		experiments.addExperiment( Experiment('3') )
		experiments.addExperiment( Experiment('4') )

		self.assertEqual( experiments.getNumOfExperiments(), 4 )
		self.assertEqual( experiments.getExperimentNames(), ['1', '2', '3', '4'] )


class TestHelper(unittest.TestCase):
	def test_init(self):
		helper = Helper()

		model_name = 'rf:classifier'
		model = helper.getDefaultModel(model_name)
		self.assertEqual( model, RandomForestClassifier ) 
		
		args = helper.getDefaultArgs(model_name)
		self.assertEqual( args, {'random_state':0} )
		
		model = model(**args)
		self.assertEqual( type(model), RandomForestClassifier )

		
		helper.setRandomState(1)

		model_name = 'dt:regressor'
		model = helper.getDefaultModel(model_name)
		self.assertEqual( model, DecisionTreeRegressor ) 
		
		args = helper.getDefaultArgs(model_name)
		self.assertEqual( args, {'random_state':1} )
		
		model = model(**args)
		self.assertEqual( type(model), DecisionTreeRegressor )

	def test_setRandomState(self):
		helper = Helper()

		self.assertEqual( helper.random_state, 0 )

		try:
			helper.setRandomState( 'cheeseburger' )

			# This should never run!
			self.assertEqual(0, 1)
		except ValueError as ve:
			self.assertEqual( str(ve), 'Random state must be an integer: cheeseburger' )

		helper.setRandomState(101)

		self.assertEqual( helper.random_state, 101 )

	def test_getBuiltModel(self):
		helper = Helper()

		model_name = 'rf:classifier'
		actual = helper.getBuiltModel(model_name)
		expected = RandomForestClassifier(random_state=0)
		self.assertEqual( type(actual), type(expected) )

		try:
			model_name = 'jonathan'
			model = helper.getBuiltModel(model_name)

			# This should never run
			self.assertTrue( False )
		except ValueError as ve:
			self.assertEqual( str(ve), 'No default model found for jonathan')


	def test_RandomForestClassifier(self):
		helper = Helper()

		# Load up data
		iris = load_iris()
		X_train, y_train, X_test, y_test = iris.data[:120], iris.target[:120], iris.data[120:], iris.target[120:]

		# Bring in the default RandomForestClassifier model
		model_name = 'rf:classifier'
		actual_model = helper.getBuiltModel(model_name)

		# Explicity create the model we expect to get from getBuiltModel call
		expected_model = RandomForestClassifier(random_state=0)

		# Make sure the models are the same type before continuing
		self.assertEqual( type(actual_model), type(expected_model) )

		# Train this default model on the iris dataset
		actual_model.fit(X_train, y_train)
		
		# Get default model accuracy on testing set
		actual_accuracy = actual_model.score(X_test, y_test)

		# Complete the same process, however we make the model explicitly
		expected_model.fit(X_train, y_train)
		expected_accuracy = expected_model.score(X_test, y_test)

		# Make sure that the accuracy reported from both models is the same
		self.assertEqual( actual_accuracy, expected_accuracy )

	def test_RandomForestRegressor(self):
		helper = Helper()

		# Load up data
		X, y = load_boston(return_X_y=True)
		X_train, y_train, X_test, y_test = X[:405], y[:405], X[405:], y[405:]

		# Bring in the default RandomForestClassifier model
		model_name = 'rf:regressor'
		actual_model = helper.getBuiltModel(model_name)

		# Explicity create the model we expect to get from getBuiltModel call
		expected_model = RandomForestRegressor(random_state=0)

		# Make sure the models are the same type before continuing
		self.assertEqual( type(actual_model), type(expected_model) )

		# Train this default model on the iris dataset
		actual_model.fit(X_train, y_train)
		
		# Get default model accuracy on testing set
		actual_accuracy = actual_model.score(X_test, y_test)

		# Complete the same process, however we make the model explicitly
		expected_model.fit(X_train, y_train)
		expected_accuracy = expected_model.score(X_test, y_test)

		# Make sure that the accuracy reported from both models is the same
		self.assertEqual( actual_accuracy, expected_accuracy )

	def test_DecisionTreeClassifier(self):
		helper = Helper()

		# Load up data
		iris = load_iris()
		X_train, y_train, X_test, y_test = iris.data[:120], iris.target[:120], iris.data[120:], iris.target[120:]

		# Bring in the default RandomForestClassifier model
		model_name = 'dt:classifier'
		actual_model = helper.getBuiltModel(model_name)

		# Explicity create the model we expect to get from getBuiltModel call
		expected_model = DecisionTreeClassifier(random_state=0)

		# Make sure the models are the same type before continuing
		self.assertEqual( type(actual_model), type(expected_model) )

		# Train this default model on the iris dataset
		actual_model.fit(X_train, y_train)
		
		# Get default model accuracy on testing set
		actual_accuracy = actual_model.score(X_test, y_test)

		# Complete the same process, however we make the model explicitly
		expected_model.fit(X_train, y_train)
		expected_accuracy = expected_model.score(X_test, y_test)

		# Make sure that the accuracy reported from both models is the same
		self.assertEqual( actual_accuracy, expected_accuracy )

	def test_DecisionTreeRegressor(self):
		helper = Helper()

		# Load up data
		X, y = load_boston(return_X_y=True)
		X_train, y_train, X_test, y_test = X[:405], y[:405], X[405:], y[405:]

		# Bring in the default RandomForestClassifier model
		model_name = 'dt:regressor'
		actual_model = helper.getBuiltModel(model_name)

		# Explicity create the model we expect to get from getBuiltModel call
		expected_model = DecisionTreeRegressor(random_state=0)

		# Make sure the models are the same type before continuing
		self.assertEqual( type(actual_model), type(expected_model) )

		# Train this default model on the iris dataset
		actual_model.fit(X_train, y_train)
		
		# Get default model accuracy on testing set
		actual_accuracy = actual_model.score(X_test, y_test)

		# Complete the same process, however we make the model explicitly
		expected_model.fit(X_train, y_train)
		expected_accuracy = expected_model.score(X_test, y_test)

		# Make sure that the accuracy reported from both models is the same
		self.assertEqual( actual_accuracy, expected_accuracy )

	def test_KNeighorsClassifier(self):
		helper = Helper()

		# Load up data
		iris = load_iris()
		X_train, y_train, X_test, y_test = iris.data[:120], iris.target[:120], iris.data[120:], iris.target[120:]

		# Bring in the default RandomForestClassifier model
		model_name = 'knn:classifier'
		actual_model = helper.getBuiltModel(model_name)

		# Explicity create the model we expect to get from getBuiltModel call
		expected_model = KNeighborsClassifier()

		# Make sure the models are the same type before continuing
		self.assertEqual( type(actual_model), type(expected_model) )

		# Train this default model on the iris dataset
		actual_model.fit(X_train, y_train)
		
		# Get default model accuracy on testing set
		actual_accuracy = actual_model.score(X_test, y_test)

		# Complete the same process, however we make the model explicitly
		expected_model.fit(X_train, y_train)
		expected_accuracy = expected_model.score(X_test, y_test)

		# Make sure that the accuracy reported from both models is the same
		self.assertEqual( actual_accuracy, expected_accuracy )

	def test_KNeighborsRegressor(self):
		helper = Helper()

		# Load up data
		X, y = load_boston(return_X_y=True)
		X_train, y_train, X_test, y_test = X[:405], y[:405], X[405:], y[405:]

		# Bring in the default RandomForestClassifier model
		model_name = 'knn:regressor'
		actual_model = helper.getBuiltModel(model_name)

		# Explicity create the model we expect to get from getBuiltModel call
		expected_model = KNeighborsRegressor()

		# Make sure the models are the same type before continuing
		self.assertEqual( type(actual_model), type(expected_model) )

		# Train this default model on the iris dataset
		actual_model.fit(X_train, y_train)
		
		# Get default model accuracy on testing set
		actual_accuracy = actual_model.score(X_test, y_test)

		# Complete the same process, however we make the model explicitly
		expected_model.fit(X_train, y_train)
		expected_accuracy = expected_model.score(X_test, y_test)

		# Make sure that the accuracy reported from both models is the same
		self.assertEqual( actual_accuracy, expected_accuracy )

	def test_AdaBoostClassifier(self):
		helper = Helper()

		# Load up data
		iris = load_iris()
		X_train, y_train, X_test, y_test = iris.data[:120], iris.target[:120], iris.data[120:], iris.target[120:]

		# Bring in the default RandomForestClassifier model
		model_name = 'ab:classifier'
		actual_model = helper.getBuiltModel(model_name)

		# Explicity create the model we expect to get from getBuiltModel call
		expected_model = AdaBoostClassifier(random_state=0)

		# Make sure the models are the same type before continuing
		self.assertEqual( type(actual_model), type(expected_model) )

		# Train this default model on the iris dataset
		actual_model.fit(X_train, y_train)
		
		# Get default model accuracy on testing set
		actual_accuracy = actual_model.score(X_test, y_test)

		# Complete the same process, however we make the model explicitly
		expected_model.fit(X_train, y_train)
		expected_accuracy = expected_model.score(X_test, y_test)

		# Make sure that the accuracy reported from both models is the same
		self.assertEqual( actual_accuracy, expected_accuracy )

	def test_AdaBoostRegressor(self):
		helper = Helper()

		# Load up data
		X, y = load_boston(return_X_y=True)
		X_train, y_train, X_test, y_test = X[:405], y[:405], X[405:], y[405:]

		# Bring in the default RandomForestClassifier model
		model_name = 'ab:regressor'
		actual_model = helper.getBuiltModel(model_name)

		# Explicity create the model we expect to get from getBuiltModel call
		expected_model = AdaBoostRegressor(random_state=0)

		# Make sure the models are the same type before continuing
		self.assertEqual( type(actual_model), type(expected_model) )

		# Train this default model on the iris dataset
		actual_model.fit(X_train, y_train)
		
		# Get default model accuracy on testing set
		actual_accuracy = actual_model.score(X_test, y_test)

		# Complete the same process, however we make the model explicitly
		expected_model.fit(X_train, y_train)
		expected_accuracy = expected_model.score(X_test, y_test)

		# Make sure that the accuracy reported from both models is the same
		self.assertEqual( actual_accuracy, expected_accuracy )


class TestModel(unittest.TestCase):
	def test_init(self):
		model = Model()
		self.assertEqual( model.getName(), None )

	def test_hasConstructorArg(self):
		model = Model('test model name', RandomForestClassifier, {'random_state': 0, 'n_estimators': 100})

		self.assertTrue( model.hasConstructorArg('random_state') )
		self.assertTrue( model.hasConstructorArg('n_estimators') )
		self.assertFalse( model.hasConstructorArg('max_depth') )

	def test_setConstructorArg(self):
		model = Model('test model name', RandomForestClassifier, {'random_state': 0, 'n_estimators': 100})

		args = model.getConstructorArgs()
		self.assertTrue( 'random_state' in args )
		self.assertEqual( args['random_state'], 0 )

		model.setConstructorArg('random_state', 1)
		args = model.getConstructorArgs()
		self.assertEqual( args['random_state'], 1 )


	def test_setConstructor(self):
		model = Model('best name ever!', lambda x : x**2, {'x':1})
		self.assertEqual( model.getConstructor()(5), 25 )
		self.assertEqual( model.getConstructor()(4), 16 )
		self.assertEqual( model.getConstructorArgs(), {'x':1} )

		model.setConstructor( lambda x : x**3 )
		self.assertEqual( model.getConstructor()(3), 27 )
		self.assertEqual( model.getConstructor()(1), 1 )
		self.assertEqual( model.getConstructor()(4), 64 )

		model.setConstructor()
		self.assertEqual( model.getConstructor(), None )
		model.setConstructor(None)
		self.assertEqual( model.getConstructor(), None )
		
		try:
			model.setConstructor('the lazy brown dog jumped over the fox')

			# This should never run!
			self.assertEqual( 0, 1 )
		except ValueError as ve:
			self.assertEqual( str(ve), 'Model \'best name ever!\' cannot set constructor as non-callable value: the lazy brown dog jumped over the fox')

	def test_run(self):
		model = Model(name='rf1', constr=RandomForestClassifier, constr_args={'max_depth': 2, 'random_state': 0})

		# Creating small dataset as per https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
		X, y = make_classification(n_samples=1000, n_features=4,
				n_informative=2, n_redundant=0,
				random_state=0, shuffle=False)
		model.run(X=X, y=y)
		self.assertListEqual( model.predict( [[0, 0, 0, 0]] ), [1] )

		model = Model('dt1', DecisionTreeClassifier, {'random_state':0})
		iris = load_iris()
		model.run( iris.data[:120], iris.target[:120] )
		self.assertListEqual( model.predict( iris.data[120:] ), [2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] )

		try:
			model.run( iris.data[:120], iris.target[:119] )

			# This should never execute!
			self.assertEqual( 0, 1 )
		except ValueError as ve:
			self.assertEqual( str(ve), 'Model dt1 cannot be trained when data and target are different lengths:\n\tlen of data: 120\n\tlen of target: 119' )

	def test_eval(self):
		iris = load_iris()
		X_train, y_train = iris.data[:120], iris.target[:120]
		X_test, y_test = iris.data[120:], iris.target[120:]

		rf = Model('rf', RandomForestClassifier, {'random_state':0})
		rf.run(X_train, y_train)
		metrics = rf.eval(X_test, y_test, metrics=['acc'])

		rf_metric = Metric('rf')
		rf_metric.addValue('acc', 0.7666666666666667)

		expected_metrics = rf_metric.getValues()

		self.assertEqual( metrics, expected_metrics )

if __name__ == '__main__':
	unittest.main()
