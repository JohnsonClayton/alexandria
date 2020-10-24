from alexandria.models import SklearnModel

class RandomForest(SklearnModel):
    def __init__(self, args=None):
        print('RandomForest init reached!')
        super(args)

        self.name = 'random forest'