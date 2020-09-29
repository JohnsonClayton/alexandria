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
        self.model = self.constructor(**self.constructor_args)
        self.model.fit(X, y)

    def predict(self, X):
        if self.model:
            return self.model.predict( X )
