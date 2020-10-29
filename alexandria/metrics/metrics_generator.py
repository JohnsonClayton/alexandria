from abc import abstractmethod

class MetricsGenerator:
    def __init__(self):
        # This dictionary contains the acceptable aliases for metric names
        self.metric_name_aliases = {
            'acc': 'Accuracy',
            'accuracy': 'Accuracy',
            'Accuracy': True,
            'rec': 'Recall',
            'recall': 'Recall',
            'Recall': True,
            'prec': 'Precision',
            'precision': 'Precision',
            'Precision': True
        }

    @abstractmethod
    def getValue(self):
        pass