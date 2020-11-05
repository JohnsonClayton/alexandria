from alexandria.models.sklearn import SklearnModel

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

class DiscriminantAnalysis(SklearnModel):
    def __init__(self, model_name='discriminant analysis', flavor='linear', *args, **kwargs):
        super().__init__(*args, **kwargs)

        if type(model_name) == str:
            self.model_name = model_name
        else:
            raise ValueError('Model name must be string, not {}'.format( str( type(model_name) ) ))

        if not self.default_args:
            self.default_args = {}

        # There are two options for Discriminant Analysis through scikit-learn: Linear and Quadratic
        self.flavor_alias_dict = {
            # Add all of the aliases for Quadratic Discriminant Analysis
            'linear': 'Linear',
            'LINEAR': 'Linear',
            'lin': 'Linear',
            'LIN': 'Linear',
            'Linear': True,

            # Add all of the aliases for Quadratic Discriminant Analysis
            'quadratic': 'Quadratic',
            'QUADRATIC': 'Quadratic',
            'quad': 'Quadratic',
            'QUAD': 'Quadratic',
            'Quadratic': True
        }

        self.flavor = None
        if type(flavor) != str:
            raise ValueError('flavor argument must be string type, not {}'.format( str( type(flavor) ) ))
        else:
            if flavor in self.flavor_alias_dict:
                if self.flavor_alias_dict[ flavor ] != True:
                    self.flavor = self.flavor_alias_dict[ flavor ]
                else:
                    self.flavor = flavor
            else:
                raise ValueError('unknown flavor of Discriminant Analysis: {}'.format(flavor))
        
    def buildReturnModel(self):
        model = None

        if self.flavor:
            if self.exp_type == 'classification':
                if self.flavor == 'Linear':
                    model = LinearDiscriminantAnalysis(**self.default_args)
                elif self.flavor == 'Quadratic':
                    model = QuadraticDiscriminantAnalysis(**self.default_args)
            else:
                raise ValueError('Discriminant Analysis can only be effectively used for classification problems!')
        else:
            raise ValueError('cannot build model because the flavor of Discriminant Analysis is unknown!')
        
        return model

    def getBuiltModel(self):
        return self.buildReturnModel()

    def getName(self):
        if self.model_name and self.flavor:
            return '.'.join([self.model_name, self.flavor])
        elif self.model_name:
            return self.model_name
        else:
            return self.flavor

    def train(self, X, y, exp_type=''):
        # if the experiment type is specified, then set it
        if exp_type:
            self.setExperimentType(exp_type)

        # Set up the model 
        self.model = self.buildReturnModel()
        super().train(X, y)
        