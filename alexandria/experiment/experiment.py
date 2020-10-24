from alexandria.dataset import DatasetManager
from alexandria.models import ModelsManager

class Experiment:
    def __init__(self, name):
        if type(name) == str:
            self.name = name
        else:
            raise ValueError('Experiment \'name\' argument must be string, not {}'.format(  str( type( name ) ) ))
        
        # Initialize the dataset manager object
        self.dm = DatasetManager()

        # Initialize the models object
        self.models_manager = ModelsManager()

    def getName(self):
        return self.name
    def setName(self, name):
        if type(name) == str:
            self.name = name
        else:
            raise ValueError('Experiment \'name\' argument must be string, not {}'.format(  str( type( name ) ) ))
       