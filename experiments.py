from sklearn.tree import DecisionTreeClassifier

class Experiments:
    '''
    Experiments object keeps track of all experiment objects provided
    '''
    def __init__(self, experiments=[]):
        # Keep track of all of the experiment objects for this experiment
        self.experiment_list = []

        if type(experiments) == list and len(experiments) > 0:
            # TO-DO: Should we check if these are legitimate experiment objects? - Probably
            self.experiment_list = experiments 

        # Keep constants around
        self.default_experiment_name = 'experiment_'
        self.num_of_experiments = len(self.experiment_list)

    def getDefaultExperimentName():
        return self.default_experiment_name + str(self.num_of_experiments)

    def addExperiment(self, experiment):
        if type(experiment) == Experiment:
            self.experiment_list.append(experiment)
        else:
            raise ValueError('Object must be Experiment object: {}'.format(str(experiment)))


class Experiment:
    def __init__(self, name='', models=[]):
        if type(name) is str:
            if name != '':
                self.name = name
            else:
                self.name = None
        else:
            raise ValueError('Experiment name attribute must be string: {}'.format(str(name)))

        # initialize the dictionary of models within the experiment
        self.model_list = {}

        # Add the models (if any) that the user wants to use
        if type(models) is list:
            if len(models) > 1:
                # Then add all of the model objects to the experiment
                for model in models:
                    self.addModel(model)
            else:
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
        return 'exp_' + str(len(self.model_list))

    def getDefaultVersionOf(self, model):
        # TO-DO: Contact state object to get the object!
        return DecisionTreeClassifier()

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

        self.model_list[name] = model

    def setRandomState(self, rand_state):
        if type(rand_state) == int:
            self.random_state = rand_state
        else:
            raise ValueError('random state must be integer: {}'.format(rand_state))
