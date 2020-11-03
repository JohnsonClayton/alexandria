from alexandria.dataset import DatasetManager
from alexandria.models import ModelsManager

from tabulate import tabulate

class Experiment:
    def __init__(self, name, dataset=None, xlabels=None, ylabels=None, libs=None, models=None, modellibdict=None):
        if type(name) == str:
            self.name = name
        else:
            raise ValueError('Experiment \'name\' argument must be string, not {}'.format(  str( type( name ) ) ))
        
        # Initialize the dataset manager object
        self.dm = DatasetManager()
        if type(dataset) != type(None):
            if type(xlabels) == type(None) or type(ylabels) == type(None):
                self.dm.setData(dataset)
            else:
                self.dm.setData(dataset=dataset, xlabels=xlabels, ylabels=ylabels)

        # Initialize the models object
        self.mm = ModelsManager()

        # Add the provided model information
        self.addModels(libs=libs, models=models, modellibdict=modellibdict)

    # This method only exists to allow for users to feel more comfortable with the API  
    def addModel(self, lib=None, model=None, modellibdict=None):
        self.addModels(libs=lib, models=model, modellibdict=modellibdict)

    def addModels(self, modellibdict=None, libs=None, models=None):
        # We can hand this data over
        if modellibdict != None:
            self.mm.addModels( modellibdict )

        # If model values were specified, we must translate them into a dictionary
        #   for the model manager
        else:
            if models != None:
                modellibdict = self.createForModelsManager(libs=libs, models=models)
                self.mm.addModels( modellibdict )


    def createForModelsManager(self, libs=[], models=[]):
        return_dict = dict()
        
        # If libs is None, then we will assume the default library on the other end
        if libs == None:
            libs = ''

        # If the libs argument is a string, then add all the models with this library
        if type(libs) == str:
            lib = libs
            return_dict[ lib ] = models
        elif type(libs) == list:
            if type(models) == list:
                if len(models) == len(libs):
                    for lib, model in zip(libs, models):
                        if lib in return_dict:
                            return_dict[ lib ].append(model)
                        else:
                            return_dict[ lib ] = [ model ]

        return return_dict                            

        

    def getName(self):
        return self.name
    def setName(self, name):
        if type(name) == str:
            self.name = name
        else:
            raise ValueError('Experiment \'name\' argument must be string, not {}'.format(  str( type( name ) ) ))
       
    def getModels(self, aslist=False):
        return self.mm.getModels(aslist)

    def getNumModels(self):
        return len(self.mm.getNumModels())

    def train(self, X=None, y=None, *args, **kwargs):
        X = X
        y = y
        exp_type = ''
        if type(X) == type(None) or type(y) == type(None):
            self.dm.splitData(*args, **kwargs)
            X, y = self.dm.getXtrain(), self.dm.getytrain()
            exp_type = self.dm.target_type
        else:
            # If user provides data, we need to figure out what type of experiment it is
            exp_type = self.dm.getExperimentTypeOf(y)

        # Train the models on the provided data
        self.mm.trainModelsOnXy(X, y, exp_type)

    def trainCV(self, X=None, y=None, nfolds=-1, metrics=''):
        if type(metrics) == type(None):
            raise ValueError('Metrics must be defined for cross validation!')

        X=X
        y=y
        exp_type = ''
        if type(X) == type(None) or type(y) == type(None):
            X = self.dm.getX()
            y = self.dm.gety()
            exp_type = self.dm.getTargetType()
        else:
            # If user provides data, we need to figure out what type of experiment it is
            exp_type = self.dm.getExperimentTypeOf(y)
        
        self.mm.trainCV(X, y, metrics=metrics, nfolds=nfolds, exp_type=exp_type)

    def predict(self, X=None):
        if type(X) != type(None):
            return self.mm.generateModelPredictions(X)
        else:
            return self.mm.generateModelPredictions( self.dm.getXtest() )

    def getMetrics(self):
        return self.mm.getMetrics()

    def summarizeMetrics(self):
        # TO-DO: Make this output a lot smarter and more customizable 
        print('\n' + self.name)

        metrics = self.mm.getMetrics()

        # List will hold all the rows for tabulate
        rows = []
        headers = []

        # Go through all of the metrics in the dictionary
        for model_metrics in metrics.values():
            if type(model_metrics) == dict:
                # Set up the headers
                if headers == []:
                    headers = list( model_metrics.keys() )

                # Initialize the row
                row = []

                for name, value in model_metrics.items():
                    # If there is an average and standard deviation, then let's output both
                    if type(value) == dict and 'avg' in value and 'std' in value:
                        row.append('{:.4f}\u00B1{:.4f}'.format( value['avg'], value['std'] ))
                    else:
                        row.append(value)

                rows.append(row)
        print( tabulate( rows, headers=headers ) )
