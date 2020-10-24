from alexandria.models import Model

class SklearnModel(Model):
    def __init__(self, default_args={}, exp_type='', *args, **kwargs):
        super().__init__()
        self.lib = 'sklearn'

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
