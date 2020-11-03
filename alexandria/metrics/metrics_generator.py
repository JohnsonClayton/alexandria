from abc import abstractmethod

class MetricsGenerator:
    def __init__(self):
        # This dictionary contains the acceptable aliases for metric names
        self.metric_name_aliases = {
            # Add all the aliases for accuracy
            'acc': 'Accuracy',
            'accuracy': 'Accuracy',
            'Accuracy': True,

            # Add all the aliases for r squared
            'r squared': 'R2',
            'rsquared': 'R2',
            'Rsquared': 'R2',
            'RSquared': 'R2',
            'r2': 'R2',
            'R2': True,

            # Add all the aliases for recall
            'rec': 'Recall',
            'recall': 'Recall',
            'Recall': True,

            # Add all the aliases for precision
            'prec': 'Precision',
            'precision': 'Precision',
            'Precision': True,

            # Add all the aliases for area under the ROC curve
            'auc': 'AUC',
            'AuC': 'AUC',
            'AUC': True
        }

    def getStandardizedName(self, name):
        new_name = ''
        if type(name) == str:
            if name in self.metric_name_aliases:
                if self.metric_name_aliases == True:
                    new_name = name
                else:
                    new_name = self.metric_name_aliases[name]
            else:
                new_name = 'no standardized name for {}'.format(name)
        else:
            raise ValueError('name argument must be string type, not {}'.format(str(type(name))))

        return new_name

    @abstractmethod
    def getValue(self):
        pass