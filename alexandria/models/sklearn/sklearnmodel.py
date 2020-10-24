from alexandria.models import Model

class SklearnModel(Model):
    def __init__(self, args=None):
        print('SklearnModel init reached!')
        super(args)
        self.lib = 'sklearn'
