# alexandria
This is a high-level machine learning framework that allows for the users to easily run multiple types of machine learning experiments at the drop of a hat. I'm currently working on developing this project, along with the [wiki pages](https://github.com/JohnsonClayton/alexandria/wiki) further.


### Build
To build from source (which is currently the only way to build this), use the Makefile:  
```
$ make
```  
This will call the `setup.py` script and will attempt to install the package onto your system. If you find any issues, please create one and I'll get on to it. I haven't done these sorts of things before, so bugs are expected.   

### Example

An example for the API is below:

```python
# examples/demo.py - DataBunch and DataFrame demonstrations
# Data preprocessing
from sklearn.datasets import load_iris, load_diabetes

from alexandria.experiment import Experiment

if __name__ == '__main__':
	# Data preprocessing
	iris = load_iris()

	experiment = Experiment(
		name='Cross Validation Example #1',
		dataset=iris,
		xlabels='data',
		ylabels='target',
		models=['rf', 'dt', 'knn']
	)
	experiment.trainCV(nfolds=10, metrics=['accuracy', 'rec', 'prec'])
	experiment.summarizeMetrics()

	# Data preprocessing for dataframe object
	diabetes_df = load_diabetes(as_frame=True).frame
	data_cols = diabetes_df.columns[:-1] # All columns, but the last one is the target
	target_col = diabetes_df.columns[-1] # 'target'

	experiment = Experiment(
		name='Cross Validation Example #2',
		dataset=diabetes_df,
		xlabels=data_cols,
		ylabels=target_col,
		models=['rf', 'dt', 'knn']
	)
	experiment.trainCV(nfolds=10, metrics='r2')
	experiment.summarizeMetrics()
```
```
Cross Validation Example #1
name                   Accuracy       Recall         Precision
---------------------  -------------  -------------  -------------
sklearn.random forest  0.9600±0.0442  0.9600±0.0442  0.9644±0.0418
sklearn.decision tree  0.9600±0.0442  0.9600±0.0442  0.9644±0.0418
sklearn.k neighbors    0.9667±0.0447  0.9667±0.0447  0.9738±0.0339

Cross Validation Example #2
name                   R2
---------------------  --------------
sklearn.random forest  0.3963±0.1006
sklearn.decision tree  -0.2044±0.2989
sklearn.k neighbors    0.3329±0.1247
```
