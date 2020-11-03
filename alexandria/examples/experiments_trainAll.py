from alexandria.experiment import Experiments, Experiment

from sklearn.datasets import load_iris
from sklearn.feature_selection import chi2

import pandas as pd

if __name__ == '__main__':
    # Create the experiments object
    exps = Experiments()

    models = ['rf', 'dt', 'knn']
    metrics = ['accuracy', 'rec', 'prec']

    # Load in the dataset
    iris_df = load_iris(as_frame=True).frame
    data_cols = iris_df.columns[:-1]
    target_col = iris_df.columns[-1] # 'target'

    # Let's figure out the order in which these features are the most
    #  useful using the chi-squared technique

    X = iris_df[data_cols]
    y = iris_df[target_col]
    chi_scores = chi2(X, y)
    p_vals = pd.Series( chi_scores[1], index=data_cols )
    p_vals.sort_values(ascending=False, inplace=True)

    # p_vals now has the ordered list of features from most significant 
    #  to least significant
    print('feature\t\t\tp value (higher=more significant)')
    print(p_vals.head())

    data_cols = list( p_vals.index )

    for i in range(1, len(data_cols)+1 ): # The last column is the target column
        # We will slowly add one more feature to the usable list
        features = data_cols[:i]
        exp = Experiment(
            name='Iris with features: {}'.format(', '.join(features)),
            dataset=iris_df,
            xlabels=features,
            ylabels=target_col,
            models=models
        )
        exps.addExperiment(exp)

    # Use 10-fold CV for all the experiments
    exps.trainCV(metrics=metrics, nfolds=10)

    # Get the results
    exps.summarizeMetrics()
