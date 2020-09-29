class Model:
    def __init__(self, name='', constructor=None, constructor_args=None):
        if type(name) == str:
            if name != '':
                self.name = name
        else:
            raise ValueError('Name must be string type: {}'.format(name))

        self.constructor = None
        self.constructor_args = None

        if constructor:
            self.constructor = constructor
            
            if type(constructor_args) == dict:
                self.constructor_args = constructor_args
