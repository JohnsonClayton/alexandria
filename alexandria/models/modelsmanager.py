from alexandria.models import Model, sklearn

import random
import warnings
from tabulate import tabulate
from mlxtend.evaluate import paired_ttest_5x2cv

# Define a custom warning format to prevent output of the line raising the warning
def msg_only(msg, *args, **kwargs):
    return str(msg) + '\n'

class ModelsManager:
    def __init__(self, default_library='scikit-learn'):
        # Set the default warning format to the custom one
        warnings.formatwarning = msg_only

        self.models = {}

        
        # Set up dictionaries to allow for aliasing on library and model names
        #   In this set up, if the key 'x' points to a string, then it is an alias.
        #   If the provided key points to True, then the key is a legit library/model name
        self.lib_aliases = {
            # Enter all the aliases for scikit-learn
            'sklearn':'scikit-learn',
            'scikit-learn': True
        }
        self.model_aliases = {
            # Enter all the aliases for Random Forest
            'rf': 'Random Forest',
            'random forest': 'Random Forest',
            'Random Forest': True,

            # Enter all the aliases for Decision Tree
            'dt': 'Decision Tree',
            'decision tree': 'Decision Tree',
            'Decision Tree': True,

            # Enter all the aliases for kNearest Neighbor
            'knn': 'K Nearest Neighbor',
            'knearest': 'K Nearest Neighbor',
            'knearestneighbor': 'K Nearest Neighbor',
            'kneighbor': 'K Nearest Neighbor',
            'KNN': 'K Nearest Neighbor',
            'kNN': 'K Nearest Neighbor',
            'KNearestNeighbor': 'K Nearest Neighbor',
            'kNearestNeighbor': 'K Nearest Neighbor',
            'K Nearest Neighbor': True,

            # Enter all the aliases for Naive Bayes
            'nb': 'Naive Bayes',
            'NB': 'Naive Bayes',
            'naive bayes': 'Naive Bayes',
            'naivebayes': 'Naive Bayes',
            'NaiveBayes': 'Naive Bayes',
            'Naivebayes': 'Naive Bayes',
            'naiveBayes': 'Naive Bayes',
            'Naive Bayes': True,

            # Enter all the aliases for Discriminant Analysis
            'da': 'Discriminant Analysis',
            'discriminant analysis': 'Discriminant Analysis',
            'discriminantanalysis': 'Discriminant Analysis',
            'DiscriminantAnalysis': 'Discriminant Analysis',
            'DA': 'Discriminant Analysis',
            'Discriminant Analysis': True
        }

        # Seed the random number generator, we just need random number for ids
        random.seed(0)

        # Set the default library used to scikit-learn
        if type(default_library) == str:
            self.default_library = self.findLibraryNameWithAlias(default_library)
        else:
            raise ValueError('default library argument must be string type, not {}'.format( str( type(default_library) ) ))

    def getModels(self, aslist=False):
        if aslist:
            return list( self.models.values() )
        return self.models

    def getNumModels(self):
        return len(self.models)

    def createModelID(self):
        return int( 10000*random.random())

    def addModel(self, model, lib='', **args):
        # Filter the lib and args values to account for aliases
        #   Ex. sklearn -> scikit-learn
        #   Ex. rf      -> Random Forest

        if type(model) != str:
            raise ValueError('model argument must be string type, not {}'.format( str(type(model)) ) )
        if type(lib) != str:
            raise ValueError('lib argument must be string type, not {}'.format( str(type(model)) ) )
        
        lib = self.findLibraryNameWithAlias(lib)
        model = self.findModelNameWithAlias(model)

        # Get a model object, then instantiate it
        # Then add the model object to our dictionary {id => obj}
        model_obj = self.getObjectFromName(model=model, lib=lib)
        if model_obj and issubclass( model_obj, Model ):
            # Need to generate ID for the model and avoid repeats
            model_id = self.createModelID()
            while model_id in self.models:
                model_id = self.createModelID()

            self.models[ model_id ] = model_obj(id=model_id, **args)

    def addModels(self, modelslibsdict):
        # We are expecting a dictionary where each entry is library1 => [model1, model2], library2 => [model3], etc
        '''
        Below is an example of a structure that would be valid here
        modelslibdict = {
            'sklearn': [
                {
                    'model': 'random forest',
                    'args': {
                        'random_state': 512,
                        'n_estimators': 250
                    }
                },
                {
                    'model': 'dt',
                    'args': {
                        'random_state': 219,
                        'max_depth': 4
                    }
                },
                'knn',
                'mlp'
            ],
            'library2': 'classifier',
            etc...
        }

        Please note that this method wasn't built to be user-friendly. If you want user-friendly, add the
          models through the Experiment or Experiments classes! They exist to act as intermediaries!

        '''
        
        if type(modelslibsdict) == dict:
            # All of these libraries must be strings
            for lib in modelslibsdict.keys():
                if type(lib) != str:
                    raise ValueError('library names must be strings!')

            for lib, models in modelslibsdict.items():
                if type(models) == str:
                    self.addModel(lib=lib, model=models)
                elif type(models) == list:
                    for model in models:
                        if type(model) == str:
                            self.addModel(lib=lib, model=model)
                        elif type(model) == dict and 'model' in model and 'args' in model:
                            if 'flavor' in model:
                                self.addModel(lib=lib, model=model['model'], flavor=model['flavor'], default_args=model['args'])
                            else:
                                self.addModel(lib=lib, model=model['model'], default_args=model['args'])
                        elif type(model) == dict and 'model' in model:
                            if 'flavor' in model:
                                self.addModel(lib=lib, model=model['model'], flavor=model['flavor'])
                            else:
                                self.addModel(lib=lib, model=model['model'])
                elif type(models) == dict and 'model' in models and 'args' in models:
                            self.addModel(lib=lib, model=models['model'], default_args=models['args'])
                else:
                    raise ValueError('models in dictionary must be in string, list (of string), or dictionary types! If in dictionary types, then the \'model\' attribute must be present! Cannot be {}'.format(str(type(models))))
        else:
            raise NotImplementedError('Cannot add models from non-dictionary type. Use Experiment class as intermediary or change to dictionary')    

    # This method will find the specific object we need to use for a given string
    def getObjectFromName(self, model, lib=''):
        obj = None
        if type(lib) != str:
            raise ValueError('lib argument must be string type, not {}'.format( str( type(lib) ) ) )
        if type(model) != str:
            raise ValueError('model argument must be string type, not {}'.format( str( type(lib) ) ) )
        
        if lib != '':
            if lib == 'scikit-learn':
                # It is a scikit-learn model
                if model == 'Random Forest':
                   obj = sklearn.RandomForest
                elif model == 'Decision Tree':
                    obj = sklearn.DecisionTree
                elif model == 'K Nearest Neighbor':
                    obj = sklearn.KNeighbors
                elif model == 'Naive Bayes':
                    obj = sklearn.NaiveBayes
                elif model == 'Discriminant Analysis':
                    obj = sklearn.DiscriminantAnalysis
        else:
            # TO-DO: We need to try to figure out which one the user wants
            #  we should output all of the ones we may think match up
            obj = None

        return obj

    def setDefaultLibrary(self, lib):
        pass

    def findLibraryNameWithAlias(self, lib=''):
        if type(lib) == str:
            if lib:
                # Go through the alias dictionary to check if the lib exists
                if lib in self.lib_aliases:
                    value = self.lib_aliases[ lib ]
                    
                    # If the value at this location is a string, then it is the library name
                    if type(value) == str:
                        lib = value
                    elif type(value) == bool: 
                        if not value:
                            # This will only occur if we have a library name pointing to False, which is when we
                            #   are explicitly stating it is not supported
                            raise ValueError('Library \'{}\' not supported. If you believe this is in error, please create an issue at https://github.com/JohnsonClayton/alexandria/issues'.format(lib))
                    else:
                        raise RuntimeError('Illegal state of library alias dictionary: {} => {}'.format( lib, str( value ) ))
                else:
                    raise ValueError('Library \'{}\' not found or supported. If you believe this is in error, please create an issue at https://github.com/JohnsonClayton/alexandria/issues'.format(lib))
            else:
                # We need to try to figure out which one the user wants
                #  we should output all of the ones we may think match up
                warnings.warn('Library unspecified, using default library: \'{}\''.format(self.default_library) )
                
                # Default to the default library
                lib = self.default_library

            return lib
        else:
            raise ValueError('Library name must be string type, not {}'.format(  str( type( lib ) )))


    def findModelNameWithAlias(self, model):
        if type(model) == str:
            if model != '':
                # Go through the alias dictionary to check if the lib exists
                if model in self.model_aliases:
                    value = self.model_aliases[ model ]
                    
                    # If the value at this location is a string, then it is the library name
                    if type(value) == str:
                        model = value
                    elif type(value) == bool and not value:
                        # This will only occur if we have a library name pointing to False, which is when we
                        #   are explicitly stating it is not supported
                        raise ValueError('Model \'{}\' not supported. If you believe this is in error, please create an issue at https://github.com/JohnsonClayton/alexandria/issues'.format(model))
                    else:
                        raise RuntimeError('Illegal state of model alias dictionary: {} => {}'.format( model, str( value ) ))
                else:
                    raise ValueError('alexandria does not recognize model name or alias \'{}\', or this is unsupported in alexandria. If you believe this is in error, please create an issue at https://github.com/JohnsonClayton/alexandria/issues'.format( str(model) ))
            else:
                raise ValueError('model argument must be non-empty string!')

            return model
        else:
            raise ValueError('Model name must be string type, not {}'.format(  str( type(model) )))

    def isSklearn(self, model):
        if isinstance(model, sklearn.SklearnModel):
            return True
        else:
            return False

    def trainModelsOnXy(self, X, y, exp_type):
        for model in self.models.values():
            if self.isSklearn( model ):
                model.train(X, y, exp_type)

    def trainCV(self, X, y, exp_type, metrics, nfolds):
        for model in self.models.values():
            if self.isSklearn( model ):
                model.trainCV(X, y, exp_type, metrics, nfolds)

    def generateModelPredictions(self, X):
        preds = dict()
        for id, model in self.models.items():
            if self.isSklearn( model ):
                preds[ id ] = {
                    'name': model.lib + '.' + model.model_name,
                    'predictions': ( model.predict( X ) ).tolist()
                }

        return preds

    def getMetrics(self):
        metrics = dict()
        for id, model in self.models.items():
            metrics[id] = model.getMetrics()

        return metrics
            
    def compareModels_tTest(self, X, y, exp_type, a=0.05):
        # Initialize list objects and set up the header row
        rows = []
        headers = ['model pairs', 'model1', 'mean1', 'std1', 'model2', 'mean2', 'std2', 'sig/notsig']
        model_ids = list(self.models.keys())
        
        # Initialize bestModel and respective values to the first one we have
        bestModel = self.models[ model_ids[0] ]
        bestModel.trainCV(X, y, exp_type=exp_type, nfolds=10, nrepeats=2, metrics='accuracy')
        bestScores = bestModel.getMetrics()
        bestAvg = bestScores['Accuracy']['avg']
        bestStd = bestScores['Accuracy']['std']

        for model_id in model_ids[1:]:
            # Set up the current model
            model = self.models[ model_id ]

            # Set the row's first element
            row = ['{} vs {}'.format( bestModel.getName(), model.getName() )]

            # Train and get the results for the current model
            model.trainCV(X, y, exp_type=exp_type, nfolds=10, nrepeats=2, metrics='accuracy')
            modelScores = model.getMetrics()
            modelAvg = modelScores['Accuracy']['avg']
            modelStd = modelScores['Accuracy']['std']

            # Compare the scores with the best one so far
            swap = False
            if modelAvg > bestAvg:
                # This is the best classifier so far

                # Add the previous best model's information to the row
                row.append( bestModel.getName() )
                row.append( '{:.4f}'.format(bestAvg) )
                row.append( '{:.4f}'.format(bestStd) )

                # Add the current model information to the row
                row.append( model.getName() + '*' )
                row.append( '{:.4f}'.format(modelAvg) )
                row.append( '{:.4f}'.format(modelStd) )
            
                # Set the best model as the current one
                swap = True
            else:
                # This model performed worse than the best we've seen so far
            
                # Add the previous best model's information to the row
                row.append( bestModel.getName() + '*')
                row.append( '{:.4f}'.format(bestAvg) )
                row.append( '{:.4f}'.format(bestStd) )

                # Add the current model information to the row
                row.append( model.getName() )
                row.append( '{:.4f}'.format(modelAvg) )
                row.append( '{:.4f}'.format(modelStd) )

            # Determine whether the difference in performance is significant
            t, p = paired_ttest_5x2cv(
                estimator1=model.getBuiltModel(), 
                estimator2=bestModel.getBuiltModel(), 
                X=X, 
                y=y, 
                scoring='accuracy',
                random_seed=0
                )

            # Add the t, p values to the row
            #row.append('{:.3f}'.format(t))
            #row.append('{:.3f}'.format(p))  

            # Add the significance determination to the row
            if p <= a:    
                row.append('sig')
            else:
                row.append('notsig')

            # Add the completed row to the list of rows
            rows.append(row)

            if swap:
                bestModel = model
                bestAvg = modelAvg
                bestStd = modelStd

        # Print the table
        print( tabulate(rows, headers=headers) )
