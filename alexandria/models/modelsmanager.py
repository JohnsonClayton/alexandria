from alexandria.models import Model, sklearn

import random
import warnings
class ModelsManager:
    def __init__(self, default_library='scikit-learn'):
        self.models = {}

        
        # Set up dictionaries to allow for aliasing on library and model names
        #   In this set up, if the key 'x' points to a string, then it is an alias.
        #   If the provided key points to True, then the key is a legit library/model name
        self.lib_aliases = {
            'sklearn':'scikit-learn',
            'scikit-learn': True
        }
        self.model_aliases = {
            'rf': 'Random Forest',
            'random forest': 'Random Forest',
            'Random Forest': True,
            'dt': 'Decision Tree',
            'decision tree': 'Decision Tree',
            'Decision Tree': True
        }

        # Seed the random number generator, we just need random number for ids
        random.seed(0)

        # Set the default library used to scikit-learn
        if type(default_library) == str:
            self.default_library = self.findLibraryNameWithAlias(default_library)
        else:
            raise ValueError('default library argument must be string type, not {}'.format( str( type(default_library) ) ))


    def getModelID(self):
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
            model_id = self.getModelID()
            while model_id in self.models:
                model_id = self.getModelID()

            self.models[ model_id ] = model_obj(id=model_id, **args)
        
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
        else:
            # We need to try to figure out which one the user wants
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

