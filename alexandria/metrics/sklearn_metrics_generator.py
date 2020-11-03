from alexandria.metrics import MetricsGenerator

from sklearn.metrics import accuracy_score, recall_score, precision_score, r2_score, make_scorer
from sklearn.model_selection import cross_validate

import numpy as np

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

    def getScorerObject(self, metric, exp_type):
        scorer = None
        if exp_type == 'classification':
            if metric == 'Accuracy':
                scorer = make_scorer(accuracy_score)
            elif metric == 'Recall':
                scorer = make_scorer(recall_score, average='weighted')
            elif metric == 'Precision':
                scorer = make_scorer(precision_score, average='weighted')
        elif exp_type == 'regression':
            if metric == 'R2':
                scorer = make_scorer(r2_score)
        else:
            raise ValueError('invalid experiment type, must be \'classification\' or \'regression\', not {}'.format(str(exp_type)))
        
        if not scorer:
            raise ValueError('cannot use {} metric for {} problem!'.format(metric, exp_type))

        return scorer

    def trainCV(self, model, X, y, exp_type, metrics, nfolds=-1):
        # Standardize the metric names and create scorer object
        scorer = {}
        if type(metrics) == str:
            metric = self.getStandardizedName(metrics)
            scorer[metric] = self.getScorerObject(metric, exp_type)
        elif type(metrics) == list:
            for metric in metrics:
                if type(metric) == str:
                    metric = self.getStandardizedName(metric)
                    scorer[metric] = self.getScorerObject(metric, exp_type)
                else:
                    raise ValueError('metrics argument must be string or list of strings type, not {}'.format(str(type(metric))))
        else:
            raise ValueError('metrics argument must be string or list of strings type, not {}'.format(str(type(metrics))))


        # Run cross_validate
        scores = cross_validate( model, X, y, scoring=scorer, cv=nfolds )

        return_metrics = {}
        if type(metrics) == str:
            metrics = self.getStandardizedName(metrics)
            vals = scores[ 'test_{}'.format(metrics) ]
            return_metrics[metrics] = {
                'avg': round( np.mean(vals), 4),
                'std': round( np.std(vals), 4)
            }
        else:
            for metric in metrics:
                metric = self.getStandardizedName(metric)
                vals = scores[ 'test_{}'.format(metric) ]
                return_metrics[metric] = {
                    'avg': round( np.mean(vals), 4), 
                    'std': round( np.std(vals), 4)
                }      

        return return_metrics
