import sklearn as sk

class Helper:
    def __init__(self):
        self.random_state = 0
        self.initializeDefaultModels()

    def initializeDefaultModels(self):
        self.default_model_objs = {
                'rf:regressor': [
                        sk.ensemble.RandomForestRegressor,
                        {'random_state':self.random_state}],
                'rf:classifier': [
                        sk.ensemble.RandomForestClassifier,
                        {'random_state':self.random_state}],
                'dt:regressor': [
                        sk.tree.DecisionTreeRegressor,
                        {'random_state':self.random_state}],
                'dt:classifier': [
                        sk.tree.DecisionTreeClassifier,
                        {'random_state':self.random_state}]
        }

    def getDefaultModel(self, model):
        if type(model) == str and model in self.default_model_objs:
            return self.default_model_objs[model][0]
        else:
            raise ValueError('No default model found for {}'.format(str(model)))

    def getDefaultArgs(self, model):
        if type(model) == str and model in self.default_model_objs:
            return self.default_model_objs[model][1]
        else:
            raise ValueError('No default model found for {}'.format(str(model)))

    def setRandomState(self, rand_state):
        if type(rand_state) == int:
            self.random_state = rand_state
            self.initializeDefaultModels()
        else:
            raise ValueError('Random state must be an integer: {}'.format(str(rand_state)))
