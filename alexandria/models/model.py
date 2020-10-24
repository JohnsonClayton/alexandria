from abc import abstractmethod

class Model:
    def __init__(self, id=''):
        self.model = None
        self.id = id

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass