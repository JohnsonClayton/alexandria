from alexandria.models import sklearn

import random
class ModelsManager:
    def __init__(self):
        self.models = {}

        # Seed the random number generator, we just need random number for ids
        random.seed(0)

    def getModelID(self):
        return int( 10000*random.random())

    def addModel(self, model, lib='', args={}):
        # Need to generate ID for the model and avoid repeats
        model_id = self.getModelID()
        while model_id in self.models:
            model_id = self.getModelID()

        # Get a model object, then instantiate it
        # Then add the model object to our dictionary {id => obj}
        self.models[ model_id ] = ( self.getObjectFromName(model=model, lib=lib) )(id=model_id, default_args=args)
        
    # This method will find the specific object we need to use for a given string
    def getObjectFromName(self, model, lib=''):
        if type(lib) == str:
            if lib != '':
                if lib == 'sklearn' or lib == 'scikit-learn':
                    # It is a scikit-learn model
                    if model == 'random forest' or model == 'rf':
                        obj = sklearn.RandomForest
            else:
                # We need to try to figure out which one the user wants
                #  we should output all of the ones we may think match up
                obj = None

        return obj


