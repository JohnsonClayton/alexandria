from alexandria.metrics import MetricsGenerator

from sklearn.metrics import accuracy_score, recall_score, precision_score, r2_score

class SklearnMetricsGenerator(MetricsGenerator):
    def __init__(self):
        super().__init__()
        self.lib = 'scikit-learn'

    def getValue(self, y_true, y_pred, mtype, exp_type, *args, **kwargs):
        if mtype in self.metric_name_aliases and self.metric_name_aliases[ mtype ] != True:
            # If the provided name is not the standardized name, then it's value at that alias
            #   is equal to the standardized version
            mtype = self.metric_name_aliases[ mtype ]
        
        if type(exp_type) != str:
            raise ValueError('exp_type argument must be string type, not {}'.format(str(type(exp_type))))

        value = -1
        if exp_type == 'classification':
            # sklearn implementation for acquiring accuracy
            if mtype == 'Accuracy':
                value =  accuracy_score(y_true, y_pred, *args, **kwargs)
            # sklearn implementation for acquiring recall
            elif mtype == 'Recall':
                value = recall_score(y_true, y_pred, average='weighted', *args, **kwargs)
            # sklearn implementation for acquiring precision
            elif mtype == 'Precision':
                value = precision_score(y_true, y_pred, average='weighted', *args, **kwargs)
        elif exp_type == 'regression':
            # sklearn implementation for r squared
            if mtype == 'R2':
                value = r2_score(y_true, y_pred, *args, **kwargs)

        if value == -1:
            raise ValueError('cannot use {} metric for {} problem!'.format(mtype, exp_type))

        return value