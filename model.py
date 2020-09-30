from metric import MetricsManager, Metric

from sklearn.metrics import accuracy_score, recall_score, precision_score

import numpy as np

class Model:
    def __init__(self, name='', constr=None, constr_args=None):
        self.name = None
        self.constructor = None
        self.constructor_args = None

        self.setName(name)
        self.setConstructor(constr)
        self.setConstructorArgs(constr_args)    

    def getConstructor(self):
        return self.constructor
    def setConstructor(self, constr=None):
        if constr:
            if callable(constr):
                self.constructor = constr
            else:
                if self.name:
                    raise ValueError('Model \'{}\' cannot set constructor as non-callable value: {}'.format(self.name, str(constr)))
                else:
                    raise ValueError('Model cannot set constructor as non-callable value: {}'.format(str(constr)))
        else:
            self.constructor = None

    def getConstructorArgs(self):
        return self.constructor_args
    def setConstructorArgs(self, constr_args):
        if constr_args:
            if type(constr_args) == dict:
                self.constructor_args = constr_args
            else:
                raise ValueError('Constructor arguments must be dictionary type: {}'.format(str(constr_args)))
        else:
            self.constructor_args = None

    def getName(self):
        return self.name
    def setName(self, name):
        if type(name) == str:
            if name != '':
                self.name = name
        else:
            raise ValueError('Name must be string type: {}'.format(name))

    def run(self, X, y, metrics=[]):
        if len(X) == len(y):
            self.model = self.constructor(**self.constructor_args)
            self.model.fit(X, y)
        else:
            if self.name:
                raise ValueError('Model {} cannot be trained when data and target are different lengths:\n\tlen of data: {}\n\tlen of target: {}'.format(self.name, str(len(X)), str(len(y))))
            else:
                raise ValueError('Model cannot be trained when data and target are different lengths:\n\tlen of data: {}\n\tlen of target: {}'.format(str(len(X)), str(len(y))))

    def predict(self, X):
        if self.model:
            return list(self.model.predict( X ))

    def calcAcc(self, y, y_pred):
        return accuracy_score(y, y_pred)

    def calcRec(self, y, y_pred):
        return recall_score(y, y_pred, average='weighted')

    def calcPrec(self, y, y_pred):
        return precision_score(y, y_pred, average='weighted')

    def calcMetrics(self, y, y_pred, metrics=[]):
        metrics_dict = {}
        for m in metrics:
            if m == 'acc' or m == 'accuracy':
                metrics_dict[m] = self.calcAcc(y, y_pred) 
            elif m == 'rec' or m == 'recall':
                metrics_dict[m] = self.calcRec(y, y_pred)
            elif m == 'prec' or m == 'precision':
                metrics_dict[m] = self.calcPrec(y, y_pred)
        return metrics_dict

    def eval(self, X, y, metrics=[]):
        # Collect all the predictions from the model
        if (type(X) == list and type(y) == list) or (type(X) == np.ndarray and type(y) == np.ndarray):
            if len(X) == len(y):
                y_pred = self.predict(X)
                #print('predicted y: {}'.format(y_pred))
                #print('actual y: {}'.format(list(y)))
                
                #acc = accuracy_score(y, y_pred)
                #rec = recall_score(y, y_pred, average='weighted')
                #prec = precision_score(y, y_pred, average='weighted')

                return self.calcMetrics(list(y), y_pred, metrics)

            else:
                raise ValueError('Model cannot be evaluated when data and target are different lengths:\n\tlen of data: {}\n\tlen of target: {}'.format(str(len(X)), str(len(y))))
        else:
            raise ValueError('Model.eval method must be provided data and target in list format. Instead, it got type(X) = {} and type(y) = {}'.format(str(type(X)), str(type(y))))

