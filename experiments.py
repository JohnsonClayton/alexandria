from sklearn.tree import DecisionTreeClassifier

from utils import Helper

class Experiments:
    '''
    Experiments object keeps track of all experiment objects provided
    '''
    def __init__(self, experiments=[]):
        # Keep track of all of the experiment objects for this experiment
        self.experiment_dict = {}

        if type(experiments) == list and len(experiments) > 0:
            # TO-DO: Should we check if these are legitimate experiment objects? - Probably
            for exp in experiments:
                self.addExperiment(exp)

        # Keep constants around
        self.default_experiment_name = 'experiment_'
        self.num_of_experiments = len(self.experiment_dict)

    def getDefaultExperimentName(self):
        return self.default_experiment_name + str(self.num_of_experiments)

    def addExperiment(self, experiment):
        if type(experiment) == Experiment:
            exp_name = experiment.getName()
            if exp_name == 'unnamed experiment':
                # Set its name to the default name. Do NOT change the experiment object's name
                exp_name = self.getDefaultExperimentName() 

            # Set the experiment as the value to it's own name
            self.experiment_dict[exp_name] = experiment
        else:
            raise ValueError('Object must be Experiment object: {}'.format(str(experiment)))

    def runAllExperiments(self):
        if len(self.experiment_dict.keys()) > 0:
            for experiment in self.experiment_dict.keys():
                experiment.run()
        else:
            raise ValueError('Experiments object has no models to run!')

    def getNumOfExperiments(self):
        return len(self.experiment_dict)

    def getExperiments(self):
        return self.experiment_dict

    def getExperimentNames(self):
        return list(self.experiment_dict.keys())


class Experiment:
    def __init__(self, name='', models=[], exp_type=None):
        self.helper = Helper()
    
        self.type_of_experiment = None
        if exp_type:
            if self.isValidExperimentType(exp_type):
                self.type_of_experiment = exp_type

        if type(name) is str:
            if name != '':
                self.name = name
            else:
                self.name = 'unnamed_experiment'
        else:
            raise ValueError('Experiment name attribute must be string: {}'.format(str(name)))

        # initialize the dictionary of models within the experiment
        self.models_dict = {}

        # Add the models (if any) that the user wants to use
        if type(models) is list:
            if len(models) > 1:
                # Then add all of the model objects to the experiment
                for model in models:
                    self.addModel(model)
            elif len(models) == 1:
                # Add the only model to the experiment list
                self.addModel(models)
        elif type(models) is dict:
            # Add all of the models with their corresponding names provided from the user
            for name in models.keys():
                self.addModel(model, name)
        elif type(models) is str:
            self.addModel(models)
        else:
            raise ValueError('Models must be in list format if more than one is provided. Ex. models=[\'rf\', \'Decision Tree\', RandomForestClassifer()... ]')

    def getDefaultModelName(self):
        return 'exp_' + str(len(self.models_dict))

    def isValidExperimentType(self, exp_type):
        if type(exp_type) == str:
            if exp_type == 'regression' or exp_type == 'classification':
                return True
            else:
                return False
        else:
            raise ValueError('Experiment type must be string: {}'.format(str(exp_type)))

    def setExperimentType(self, exp_type):
        if self.isValidExperimentType(exp_type):
            self.type_of_experiment = exp_type
        else:
            raise ValueError('Experiment must be \'regression\' or \'classification\', cannot be {}'.format(str(exp_type)))

    def getDefaultVersionOf(self, model_name):
        # TO-DO: Contact state object to get the object!
        # Do we know what type of experiment we're dealing with here?
        if self.type_of_experiment:
            # Figure out what type of model we need
            if self.type_of_experiment == 'regression':
                model_name = model_name + ':regressor'
            elif self.type_of_experiment == 'classification':
                model_name = model_name + ':classifier'
            return Model( model_name, self.helper.getDefaultModel(model_name), self.helper.getDefaultArgs(model_name) ) 
        else:
            return Model( model_name )

    def addModel(self, model, name=''):
        # TO-DO: What are valid objects to pass here? String? Yes, but gives default. Actual model object (such as RandomForestClassifier)? Yes!
        if type(model) is str:
            # Add the default version of the model they are requesting
            model = self.getDefaultVersionOf(model)
        elif type(model) is object:
            print('model is an object!')
            
            # TO-DO: What type of object is it? Scikit learn?
            
        # TO-DO: Find a way to generate automatic names for these models in a way that sounds smarter than model_1, model_2, ...
        if type(name) is str:
            if name == '':
                # Default name will be assigned
                name = self.getDefaultModelName()
            elif self.hasModelByName(name):
                raise ValueError('Experiment object already has model by name: {}'.format(name))
        else:
            raise ValueError('Model name must be string: {}'.format(str(name)))

        self.models_dict[name] = model

    def run(self):
        self.completed = True

    def setRandomState(self, rand_state):
        if type(rand_state) == int:
            self.random_state = rand_state
        else:
            raise ValueError('random state must be integer: {}'.format(rand_state))

    def getName(self):
        return self.name
    def getModels(self):
        return self.models_dict
