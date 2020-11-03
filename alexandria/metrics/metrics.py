import numpy as np

class Metrics:
    def __init__(self):
        self.metrics = {}

    def addPair(self, key=None, value=None, keyvalue=None):
        # If they are providing a dictionary, it should be figured out
        if type(key) == dict:
            if len(key) == 1:
                key, value = list(key.items())[0]
            elif len(key) > 1:
                raise ValueError('provided pair contains more than one pairing, please use \'addPairs\' method instead.')
            else:
                raise ValueError('provided pair is an empty dictionary!')
        if type(keyvalue) == dict:
            if len(keyvalue) == 1:
                key, value = list(keyvalue.items())[0]
            elif len(key) > 1:
                raise ValueError('provided pair contains more than one pairing, please use \'addPairs\' method instead.')
            else:
                raise ValueError('provided pair is an empty dictionary!')

        # Check the types of the values and add it to the dictionary
        if type(key) == str:
            if type(value) in [str, int, float, np.float64, dict]:
                self.metrics[key] = value
            else:
                raise ValueError('value must be string, integer, float, or dict type, not {}'.format( str(type(value)) ))
        else:
            raise ValueError('key value must be string type, not {}'.format( str( type( key ) ) ))

    def addPairs(self, pairs):
        if type(pairs) == dict:
            for key, value in pairs.items():
                self.addPair(key, value)
        else:
            raise ValueError('provided pairs must be dictionary type, not {}'.format( str(type(pairs)) ) )


    def getPair(self, key):
        if type(key) == str:
            if key in self.metrics:
                return { key : self.metrics[key] }
            else:
                raise ValueError('key value \'{}\' not found'.format( str( type(key) ) ) )
        else:
            raise ValueError('key value must be string type, not {}'.format( str(type(key)) ))

    def getPairs(self):
        return self.metrics

    def getMetric(self, key):
        if type(key) == str:
            if key in self.metrics:
                return self.metrics[key]
            else:
                raise ValueError('key value \'{}\' not found'.format( str( key ) ) )
        else:
            raise ValueError('key value must be string type, not {}'.format( str(type(key)) ))

    def reset(self):
        self.__init__()