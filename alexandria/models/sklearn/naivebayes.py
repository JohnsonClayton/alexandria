from alexandria.models.sklearn import SklearnModel

from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB

class NaiveBayes(SklearnModel):
    def __init__(self, model_name='naive bayes', flavor='gaussian', *args, **kwargs):
        super().__init__(*args, **kwargs)    

        if type(model_name) == str:
            self.model_name = model_name
        else:
            raise ValueError('Model name must be string, not {}'.format( str( type(model_name) ) ))

        if not self.default_args:
            self.default_args = {}

        # Since there's multiple types of Naive Bayes implement by Scikit-Learn, we are
        #  implementing them as flavors you can choose from the constructor
        self.flavor_alias_dict = {
            # Add all of the aliases for Bernoulli Naive Bayes
            'bernoulli': 'Bernoulli',
            'bern': 'Bernoulli',
            'Bernoulli': True,

            # Add all of the aliases for Categorical Naive Bayes
            'cat': 'Categorical',
            'categorical': 'Categorical',
            'Categorical': True,

            # Add all of the aliases for Complement Naive Bayes
            'comp': 'Complement',
            'complement': 'Complement',
            'compliment': 'Complement',
            'Complement': True,

            # Add all of the aliases for Gaussian Naive Bayes
            'gauss': 'Gaussian',
            'gaussian': 'Gaussian', 
            'Gaussian': True, 

            # Add all of the aliases for Multinomial Naive Bayes
            'multi': 'Multinomial',
            'multinomial': 'Multinomial',
            'Multinomial': True
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
                raise ValueError('unknown flavor of Naive Bayes: {}'.format(flavor))
        
    def buildReturnModel(self):
        model = None

        if self.flavor:
            if self.exp_type == 'classification':
                if self.flavor == 'Bernoulli':
                    model = BernoulliNB(**self.default_args)
                elif self.flavor == 'Categorical':
                    model = CategoricalNB(**self.default_args)
                elif self.flavor == 'Complement':
                    model = ComplementNB(**self.default_args)
                elif self.flavor == 'Gaussian':
                    model = GaussianNB(**self.default_args)
                elif self.flavor == 'Multinomial':
                    model = MultinomialNB(**self.default_args)
            else:
                raise ValueError('Naive bayes can only be used for classification problems!')
        else:
            raise ValueError('cannot build model because the flavor of Naive Bayes is unknown!')
        
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