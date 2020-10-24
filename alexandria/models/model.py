from abc import abstractmethod

class Model:
    def __init__(self):
        self.model = None

    @abstractmethod
    def train(self):
        pass