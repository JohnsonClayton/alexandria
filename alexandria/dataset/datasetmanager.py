import warnings

from sklearn.utils import Bunch

from pandas import DataFrame
class DatasetManager:
    def __init__(self, dataset=None, xlabels=[], ylabels='', target_type='', num_classes=None):
        # Initialize values
        self.reset()

        self.setData( dataset, xlabels, ylabels )

        self.calcDatasetData()
        
        self.setTargetType( target_type )
        self.setNumClasses( num_classes )

    def reset(self):
        self.target_type = None
        self.classes = None
        self.num_classes = None
        self.dataset = None
        self.datatype = None
        self.xlabels = None
        self.ylabels = None
        
    def setData(self, dataset, xlabels=[], ylabels=''):
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
            elif type(ylabels) == list:
                raise ValueError('Multi-column target is not supported!')
            else:
                raise ValueError('ylabels argument must be string, not {}'.format( str( type( ylabels ) ) ))

        # Figure out what datatype the dataset is
        try:
            if dataset:
                self.datatype = type(dataset)
        except ValueError:
            # The dataset is defined as something, just not 'None'
            self.datatype = type(dataset)

        self.dataset = dataset

    def validateBunchLabels(self):
        # Check to make sure the provided labels are actually within the sklearn.Bunch object
        if self.xlabels:
            if type(self.xlabels) == list:
                for xlabel in self.xlabels:
                    if not hasattr(self.dataset, xlabel):
                        raise ValueError('{} is not an attribute of the provided dataset!'.format( self.xlabels ))
            elif type(self.xlabels) == str and not hasattr(self.dataset, self.xlabels):
                raise ValueError('{} is not an attribute of the provided dataset!'.format( self.xlabels ))
        if self.ylabels and type(self.ylabels) == str and not hasattr(self.dataset, self.ylabels):
            raise ValueError('{} is not an attribute of the provided dataset!'.format( self.ylabels ))
           
    def validatePandasLabels(self):
        # Make sure all of the columns exist in the dataframe!
        not_found = []
        for xlabel in self.xlabels:
            if xlabel not in self.dataset.columns:
                not_found.append(xlabel)
        if self.ylabels not in self.dataset.columns:
            not_found.append(self.ylabels)

        if len(not_found) != 0:
            raise ValueError('These columns don\'t exist in the dataset: {}'.format(not_found))

    def calcDatasetData(self):
        # Try to figure out what type of dataset this is and how many classes there are (if classification)
        #   Note that these values can be overwritten by the user if our logic is wrong
        self.classes = None
        self.num_classes = None
        if self.datatype == Bunch:
            # Make sure the labels make sense!
            self.validateBunchLabels()

            # Check if the 'target_names' attribute exists because that will be a clear indicator!
            if hasattr(self.dataset, 'target_names'):
                self.target_type = 'classification'
                self.classes = list(self.dataset.target_names)
                self.num_classes = len(self.classes)
            else:
                self.target_type = 'regression'
                self.classes = None
                self.num_classes = None
        elif self.datatype == DataFrame:
            # Make sure the labels make sense!
            self.validatePandasLabels()

            # If the data series is an object, then it is classification type
            if self.dataset[ self.ylabels ].dtype == 'O' or (self.dataset[ self.ylabels ].nunique() < 0.25*len(self.dataset[ self.ylabels ])):
                # The series is an object or there are enough unique values that it makes sense to not be regression
                self.target_type = 'classification'
                self.classes = list(self.dataset[ self.ylabels ].unique())
                self.num_classes = len(self.classes)
            else:
            # If there are a max of 4 samples per unique value, assume it is a regression dataset
                self.target_type = 'regression'
                self.classes = None
                self.num_classes = None
        
    def setTargetType(self, target_type):
        # If the user defined these variables, then give them a heads up that this is kind of messed up compared 
        #   to what we just calculated
        if target_type and self.target_type != target_type:
            if type(target_type) == str:
                if target_type == 'regression' or target_type == 'classification':
                    warnings.warn('User specified {} target type, but alexandria found {} target type. Assuming the user is correct...'.format(target_type, self.target_type))
                    self.target_type = target_type
                else:
                    raise ValueError('target_type argument must be \'regression\' or \'classification\', not {}'.format( target_type ))
            else:
                raise ValueError('target type must be of string type, not {}'.format( str( type(target_type) ) ))
       
    def setNumClasses(self, num_classes):
        # If the user defined these variables, then give them a heads up that this is kind of messed up compared 
        #   to what we just calculated
        if num_classes and self.target_type == 'classification' and num_classes != self.num_classes:
            if type(num_classes) == int:
                warnings.warn('User specified {} classes, but alexandria found {} classes. Assuming user is correct...'.format(num_classes, self.num_classes))
                self.num_classes = num_classes
            else:
                raise ValueError('num_classes argument must be integer, not {}'.format( str( type( num_classes ) ) ))
        