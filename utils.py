# Models
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


class Helper:
    def __init__(self):
        self.random_state = 0
        self.initializeDefaultModels()

    def initializeDefaultModels(self):
        self.default_model_objs = {
                'rf:regressor': [
                        RandomForestRegressor,
                        {'random_state':self.random_state}],
                'rf:classifier': [
                        RandomForestClassifier,
                        {'random_state':self.random_state}],
                'dt:regressor': [
                        DecisionTreeRegressor,
                        {'random_state':self.random_state}],
                'dt:classifier': [
                        DecisionTreeClassifier,
                        {'random_state':self.random_state}],
                'ab:classifier': [
                        AdaBoostClassifier,
                        {'random_state':self.random_state}],
                'ab:regressor': [
                        AdaBoostRegressor,
                        {'random_state':self.random_state}],
                'knn:classifier': [
                        KNeighborsClassifier,
                        {}],
                'knn:regressor': [
                        KNeighborsRegressor,
                        {}]
                
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

    def getBuiltModel(self, model_name):
        model = self.getDefaultModel(model_name)
        args = self.getDefaultArgs(model_name)
        return model(**args)