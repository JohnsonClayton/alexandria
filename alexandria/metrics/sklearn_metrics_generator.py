from alexandria.metrics import MetricsGenerator

from sklearn.metrics import accuracy_score, recall_score, precision_score

class SklearnMetricsGenerator(MetricsGenerator):
    def __init__(self):
        super().__init__()
        self.lib = 'scikit-learn'

    def getValue(self, y_true, y_pred, mtype, *args, **kwargs):
        if mtype in self.metric_name_aliases and self.metric_name_aliases[ mtype ] != True:
            # If the provided name is not the standardized name, then it's value at that alias
            #   is equal to the standardized version
            mtype = self.metric_name_aliases[ mtype ]
        
        value = -1

        # sklearn implementation for acquiring accuracy
        if mtype == 'Accuracy':
            value =  accuracy_score(y_true, y_pred, *args, **kwargs)
        # sklearn implementation for acquiring recall
        elif mtype == 'Recall':
            value = recall_score(y_true, y_pred, average='weighted', *args, **kwargs)
        # sklearn implementation for acquiring precision
        elif mtype == 'Precision':
            value = precision_score(y_true, y_pred, average='weighted', *args, **kwargs)

        return value