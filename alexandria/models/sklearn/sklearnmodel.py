from alexandria.models import Model

class SklearnModel(Model):
    def __init__(self, default_args={}, *args, **kwargs):
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


    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
