from sklearn.utils import Bunch
class DatasetManager:
    def __init__(self, dataset=None, xlabels=[], ylabels=''):
        # Initialize values
        self.target_type = None
        self.datatype = None

        # Make sure that the provided labels are the correct data types (string or list of strings)
        # TO-DO: Would ints be valid as well?
        self.xlabels = None
        if xlabels:
            if type(xlabels) == str:
                self.xlabels = xlabels
            elif type(xlabels) == list:
                for label in xlabels:
                    if type(label) != str:
                        raise ValueError('xlabels list must only contain strings, not {}'.format( str( type(label) ) ))
                self.xlabels = xlabels
            else:
                raise ValueError('xlabels argument must be string or list of strings, not {}'.format( str( type( xlabels ) ) ))
        
        # Usually there is only one target column
        self.ylabels = None
        if ylabels:
            if type(ylabels) == str:
                self.ylabels = ylabels
            else:
                raise ValueError('ylabels argument must be string, not {}'.format( str( type( ylabels ) ) ))
        

        self.num_classes = None

        if dataset:
            self.datatype = type(dataset)
        self.dataset = dataset

        if self.datatype == Bunch:
            if hasattr(self.dataset, 'target_names'):
                self.target_type = 'classification'
                self.classes = list(self.dataset.target_names)
                self.num_classes = len(self.classes)
            else:
                self.target_type = 'regression'
                self.classes = None
                self.num_classes = None

        # Check to make sure provided dataset, xlabels, and ylabels all make sense together

        