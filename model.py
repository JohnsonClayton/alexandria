class Model:
    def __init__(self, name='', constructor=None):
        if type(name) == str:
            if name != '':
                self.name = name
        else:
            raise ValueError('Name must be string type: {}'.format(name))

        if constructor:
            self.constructor = constructor
