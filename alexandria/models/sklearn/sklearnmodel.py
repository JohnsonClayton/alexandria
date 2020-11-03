from alexandria.models import Model
from alexandria.metrics import Metrics, SklearnMetricsGenerator

from abc import abstractmethod

class SklearnModel(Model):
    def __init__(self, default_args={}, exp_type='', *args, **kwargs):
        super().__init__()
        self.lib = 'sklearn'
        self.model_name = 'na'

        # Make sure that the default arguments are valid (Ex. {'random_state':20})
        if type(default_args) == dict:
            # Ensure all the keys are strings
            for key in default_args.keys():
                if type(key) != str:
                    raise ValueError('All argument names must be strings!')
            self.default_args = default_args
        else:
            raise ValueError('Default arguments must be dictionary type, not {}'.format( str( type( default_args ) )))

        # Set up the experiment type
        if exp_type != '':
            self.setExperimentType(exp_type)
        else:
            self.exp_type = exp_type

        self.metrics = Metrics()
        self.mg = SklearnMetricsGenerator()

    def getArguments(self):
        return self.default_args
        

    def setExperimentType(self, exp_type):
        if type(exp_type) == str:
            if exp_type == 'regression':
                self.exp_type = exp_type
            elif exp_type == 'classification':
                self.exp_type = exp_type
            else:
                raise ValueError('Experiment type argument must be \'classification\' or \'regression\', not {}'.format( exp_type ))
        else:
            raise ValueError('Experiment type argument must be string type, not {}'.format( str( type(exp_type) ) ))
        

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def eval(self, X, y, metrics):
        if type(metrics) == str:
            preds = self.predict( X )
            actual = y

            metrics = self.mg.getStandardizedName(metrics)
            self.metrics.addPair(
                {
                    metrics: self.mg.getValue(actual, preds, mtype=metrics, exp_type=self.exp_type) 
                }
            )
        elif type(metrics) == list:
            preds = self.predict( X )
            actual = y

            for metric in metrics:
                metric = self.mg.getStandardizedName(metric)
                self.metrics.addPair(
                    {
                        metric: self.mg.getValue(actual, preds, mtype=metric, exp_type=self.exp_type) 
                    }
                )
        else:
            raise ValueError('i don\'t know, dude')

    def getMetric(self, metric):
        return self.metrics.getMetric(
            self.mg.getStandardizedName(metric)
            )

    def getMetrics(self):
        return self.metrics.getPairs()

    def trainCV(self, X, y, exp_type, metrics, nfolds=-1):
        # Reset the metrics
        self.metrics.reset()
        self.metrics.addPair('name', '{}.{}'.format( self.lib, self.model_name ))

        # Collect the new metrics
        self.setExperimentType(exp_type)
        metrics = self.mg.trainCV(
            model=self.buildReturnModel(), 
            X=X, 
            y=y, 
            exp_type=exp_type, 
            metrics=metrics, 
            nfolds=nfolds
            )
        for metric_name, vals in metrics.items():
            #print('{}:{}'.format(metric_name, vals))
            self.metrics.addPair(metric_name, vals)


    @abstractmethod
    def buildReturnModel(self):
        pass
