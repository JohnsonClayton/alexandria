from alexandria.models.sklearn import SklearnModel

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class RandomForest(SklearnModel):
    def __init__(self, model_name='random forest', *args, **kwargs):
        super().__init__(*args, **kwargs)    

        if type(model_name) == str:
            self.model_name = model_name
        else:
            raise ValueError('Model name must be string, not {}'.format( str( type(model_name) ) ))

        if not self.default_args:
            self.default_args = {
                'random_state': 0
            }
        
    def train(self, X, y, exp_type):
        if type(exp_type) == str:
            if exp_type == 'regression':
                self.model = RandomForestRegressor(**self.default_args)
            elif exp_type == 'classification':
                self.model = RandomForestClassifier(**self.default_args)
            else:
                raise ValueError('Experiment type must be \'classification\' or \'regression\', not {}'.format( experiment_type ))
        
        else:
            raise ValueError('Experiment type must be string type, not {}'.format( str( type(experiment_type) ) ))
        
        super().train(X, y)
