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
        
    def buildReturnModel(self):
        model = None

        if self.exp_type:
            if self.exp_type == 'regression':
                model = RandomForestRegressor(**self.default_args)
            elif self.exp_type == 'classification':
                model = RandomForestClassifier(**self.default_args)
            else:
                raise ValueError('Experiment type argument must be specified for RandomForest - sklearn!')
        else:
            raise ValueError('Experiment type argument must be specified for RandomForest - sklearn!')
        
        return model

    def train(self, X, y, exp_type=''):
        # if the experiment type is specified, then set it
        if exp_type:
            self.setExperimentType(exp_type)
        
        # Set up the model 
        self.model = self.buildReturnModel()
        super().train(X, y)
        
    def predict_proba(self, X):
        if self.exp_type:
            if self.exp_type == 'classification':
                return super().predict_proba(X)
            else:
                raise NotImplementedError('The \'predict_proba\' method is not implemented for regression problems (this is an scikit-learn issue, not an alexandria issue!)')
        else:
            raise ValueError('Experiment type argument must be specified for RandomForest - sklearn!')
