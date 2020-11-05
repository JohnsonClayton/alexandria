from alexandria.models.sklearn import SklearnModel

from sklearn.linear_model import LogisticRegression as LR

class LogisticRegression(SklearnModel):
    def __init__(self, model_name='logistic regression', *args, **kwargs):
        super().__init__(*args, **kwargs)

        if type(model_name) == str:
            self.model_name = model_name
        else:
            raise ValueError('Model name must be string, not {}'.format( str( type(model_name) ) ))

        if not self.default_args:
            self.default_args = {'random_state': 0}

        
    def buildReturnModel(self):
        model = None
        
        if self.exp_type == 'classification':
            model = LR(**self.default_args)
        else:
            raise ValueError('Logistic Regression can only be effectively used for classification problems!')
        return model

    def getBuiltModel(self):
        return self.buildReturnModel()

    def getName(self):
        return self.model_name

    def train(self, X, y, exp_type=''):
        # if the experiment type is specified, then set it
        if exp_type:
            self.setExperimentType(exp_type)

        # Set up the model 
        self.model = self.buildReturnModel()
        super().train(X, y)
        