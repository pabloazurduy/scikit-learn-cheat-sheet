# scikit-learn-cheat-sheet
A compilation of main commands for scikit-learn with examples. Inspired by https://inria.github.io/scikit-learn-mooc/index.html.

- [scikit-learn-cheat-sheet](#scikit-learn-cheat-sheet)
  - [1. Numerical data preprocessing](#1-numerical-data-preprocessing)
  - [1.1 Text data preprocessing](#11-text-data-preprocessing)
  - [2. Encoding](#2-encoding)
  - [3. Column selection and transformation](#3-column-selection-and-transformation)
  - [4. Pipelines](#4-pipelines)
  - [5. Model training](#5-model-training)
  - [6. Metrics](#6-metrics)
  - [7. Parameter tuning](#7-parameter-tuning)
  - [8. Model selection](#8-model-selection)
  - [9. Dummy models](#9-dummy-models)
  - [10. Linear models](#10-linear-models)
  - [11. kNN](#11-knn)
  - [12. Tree models](#12-tree-models)
  - [12. Ensemble models](#12-ensemble-models)

## 1. Numerical data preprocessing

### [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

Standardizes data by removing the mean and scaling to unit variance.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)
scaler.transform(data)
```

### [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

Transforms the data so that it values appear in the given range.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data)
scaler.transform(data)
```

### [Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)

Each sample (i.e. each row of the data matrix) with at least one non zero component is rescaled independently of other samples so that its norm (l1, l2 or inf) equals one.

```python
from sklearn.preprocessing import Normalizer
transformer = Normalizer()
transformer.fit(data)
transformer.transform(data)
```
### [KBinsDiscretizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html)
Bin continuous data into intervals.
```python
from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=3, 
                       encode='onehot', # onehot (ordinal too)
                       strategy='uniform' # quantile
                       )
```


### [Binarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html)

Binarizes data (set feature values to 0 or 1) according to a threshold.

```python
from sklearn.preprocessing import Binarizer
transformer = Binarizer().fit(data)
transformer.transform(data)
```

### [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

Replaces missing values using a descriptive statistic (e.g. mean, median, or most frequent) along each column, or using a constant value.
Parameters: `missing_values` specifies what we assume as a missing value, `strategy` specifies what we will replace the missing values with.

```python
import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(data)
imputer.transform(data)
```


### [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer)

Multivariate imputer that estimates each feature from all the others.
A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.

```python
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
imp_mean = IterativeImputer(estimator=None, # by default is BayesianRidge()
                            random_state=0)
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])
```

### [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)

Generates polynomial and interaction features.
Parameters: `degree` specifies the maximal degree of the polynomial features

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
poly.fit_transform(data)
```

## 1.1 Text data preprocessing

### [MultiLabelBinarizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html)
Transform between iterable of iterables and a multilabel format.

```python
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit_transform([(1, 2), (3,)])
```

### [CountVectorizer](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

We call vectorization the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the Bag of Words or “Bag of n-grams” representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.


```python 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(1, 2), # do words and pairs of words
                             token_pattern=r'\b\w+\b', 
                             lowercase=True, 
                             stop_words=None, #'english' or list of words
                             min_df=1 #float:[0,1] When building the vocabulary ignor terms that have a document frequency strictly lower than the given threshold.
                             max_df=0.9, #ignore terms that have a document frequency strictly higher than the given threshold  (to infer stop words)
                             )

X = vectorizer.fit_transform(corpus) # create a sparse matrix
# get vocabulary 
vectorizer.vocabulary_.get('document')
```
### [TfidfTransformer](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)


$$\text{tf-idf(t,d)}=\text{tf(t,d)} \times \text{idf(t)}$$
$$\text{idf}(t) = \log{\frac{1 + n}{1+\text{df}(t)}} + 1$$

```python
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer 
# the vectorizer is a merge between the CountVectorizer and the TfidfTransformer
transformer = TfidfTransformer(norm='l2', 
                               use_idf=True, 
                               smooth_idf=True, #adding one to document frequencies, Prevents zero divisions. 
                               sublinear_tf=False)
transformer

```



## 2. Encoding

### [OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)

`OrdinalEncoder` will encode each category with a different number.

```python
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(data)
```

### [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

For a given feature, `OneHotEncoder` will create as many new columns as there are possible categories. For a given sample, the value of the column corresponding to the category will be set to 1 while all the columns of the other categories will be set to 0.

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False,max_categories=None )
data_encoded = encoder.fit_transform(data)
```

## 3. Column selection and transformation

### [make_column_selector](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector)

Selects columns based on datatype or the columns name.

### [pandas column selector](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html)

Select columns based on their datatype, returns a DataFrame
```python 
import pandas as pd 
numeric_columns = df.select_dtypes(include='number').columns
str_columns = df.select_dtypes(include='object').columns
dt_columns = df.select_dtypes(include='datetime').columns 

```

### [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer)

Applies specific transformations to the subset of columns in the data.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

categorical_preprocessor = OrdinalEncoder(handle_unknown="use_encoded_value",
                                          unknown_value=-1)
preprocessor = ColumnTransformer([
    ('cat_preprocessor', categorical_preprocessor, categorical_columns)],
    remainder='passthrough', sparse_threshold=0)
```

## 4. Pipelines

### [make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)

Allows to construct a pipeline – a set of commands/models/etc. which will be executed consequently.

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=500)
)
```

### [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
Applies transformers to columns of an array or pandas DataFrame.

```python
from sklearn.compose import ColumnTransformer 

ct = ColumnTransformer(transformers = [('child_pipeline', child_pipeline, 'children'),
                                       ('age_pipeline', age_pipeline, 'age'),
                                       ('cat', cat_pipeline, ['sex', 'smoker', 'region']),
                                       ('num', num_pipeline, ['bmi']),
                                       ('np_array_transform', 'passthrough', ['numpy_array']) # creates a non transformed col
                                      ], 
                       remainder='drop', 
                       sparse_threshold=0
                       )
#remember that any pipeline or transformer can be converted in pd dataframe 
X_out= ct.fit_transform(X_df
X_df_out = pd.DataFrame(X_out), columns=ct.get_feature_names_out())
```


### [FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html)
Create a Custom Transformation of a column

```python 
from sklearn.preprocessing import  FunctionTransformer
FunctionTransformer(func=lambda x:np.where(x<0, np.nan, x).reshape(-1, 1), # vectorial function 
                    inverse_func = None, # optional 
                    feature_names_out='one-to-one' #: keeps the same names or alternatively 
                    feature_names_out=lambda ft, fn:fn + '_pos' # args: self FunctionTransformer[self], feat_name upstream
                    # feature_names_out always return a list ex:['cluster_id']
                    )
```
### adding kmeans to a pipeline 


```python
ct = ColumnTransformer(transformers=[('bins', KBinsDiscretizer(n_bins=20, strategy='uniform', encode='onehot'), ['subscriber_id']),
                                     ('one_hot', OneHotEncoder(drop='first', sparse_output=False,max_categories=None ), ['age_group']),
                                     ('scaler', StandardScaler(), ['engagement_time', 'engagement_frequency']),
                                    ],
                      remainder='drop', 
                      sparse_threshold=0)
ct.fit_transform(X_train)
feat_idx = list(range(len(ct.get_feature_names_out())))
# this is not very correct given that the function will be called twice even in 
kmeansf = FunctionTransformer(func= lambda X:KMeans(n_clusters=5).fit_predict(X).reshape(-1,1),
                              feature_names_out=lambda ft, fn:['cluster_id']
                             )
ct_kmeans = ColumnTransformer(transformers=[('cluster_id', kmeansf ,feat_names),
                                            ('np_array_transform', 'passthrough', feat_names)
                                           ],
                      remainder='passthrough')
                                            
```

### [get_feature_names_out](https://scikit-learn.org/stable/modules/compose.html#tracking-feature-names-in-a-pipeline)
To enable model inspection, Pipeline has a `get_feature_names_out()` method, just like all transformers. You can use pipeline slicing to get the feature names going into each step:

```python 
X = pipe.fit_transform(X)
pipe.get_feature_names_out()
pipe[:-1].get_feature_names_out()
```

### [set_config](https://scikit-learn.org/stable/modules/generated/sklearn.set_config.html#sklearn.set_config)

Allows to vizualize the pipelines in Jupyter, needs to be set once at the beginning of your notebook.

```python
from sklearn import set_config
set_config(display="diagram")
```

## 5. Model training

### [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)

Split arrays or matrices into random train and test subsets.

```python
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42)
```

### [learning_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)

Allows to see how the model performance changes when choosing different train/test split size.

```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()

from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=30, test_size=0.2)

from sklearn.model_selection import learning_curve

train_sizes=[0.3, 0.6, 0.9]
results = learning_curve(
    regressor, data, target, train_sizes=train_sizes, cv=cv,
    scoring="neg_mean_absolute_error", n_jobs=2)
```

## 6. Metrics

### [metrics_module](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)

```python
'neg_mean_squared_error'
'neg_root_mean_squared_error'
'balanced_accuracy'

```
### [mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred)
```

### [precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

`average` parameter is required for multiclass/multilabel targets.

```python
from sklearn.metrics import precision_score
precision_score(y_true, y_pred, average='macro')
```

### [recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

`average` parameter is required for multiclass/multilabel targets.

```python
from sklearn.metrics import recall_score
recall_score(y_true, y_pred, average='macro')
```

### [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)

The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.

```python
from sklearn.metrics import balanced_accuracy_score
balanced_accuracy_score(y_true, y_pred)
```

### [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)

```python
from sklearn.metrics import confusion_matrix
labels=["a", "b", "c"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
```

### [ConfusionMatrixDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix)

Confusion Matrix visualization.

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels=["a", "b", "c"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()
```
### [roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve)

`pos_label` parameter defines the label of the positive class.

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
```

### [auc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc)

Compute Area Under the Curve (AUC).

```python
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
roc_auc = auc(fpr, tpr)
```

### [RocCurveDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html)

ROC Curve visualization.

```python
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
roc_auc = auc(fpr, tpr), estimator_name='example estimator')
disp.plot()
plt.show()
```

### [precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve)

```python
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
```

### [PrecisionRecallDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html#sklearn.metrics.PrecisionRecallDisplay)

Precision-Recall visualization.

```python
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.show()
```

### [Feature Importance](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py)

```python 
# feature importance by RF/Tree
feature_importance = model.feature_importances_
# plot
import matplotlib.pyplot as plt
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, feat_names[sorted_idx])
plt.title("Feature Importance (MDI)") # based on impurity
```
### [permutation_importance](https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance)

based on randomization order of the feature and then testing the effect on the score/accuracy of the model 

```python 
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X, y, n_repeats=10,random_state=42)

sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=feat_names[sorted_idx],
)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()
```

## 7. Parameter tuning

### [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

Greedy search over specified parameter values for an estimator.

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'parameter_A': (0.01, 0.1, 1, 10),
    'parameter_B': (3, 10, 30)}
model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                 n_jobs=2, cv=2)
model_grid_search.fit(data, target)
```

### [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

In contrast to `GridSearchCV`, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by `n_iter`.

```python
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'parameter_A': (0.01, 0.1, 1, 10),
    'parameter_B': (3, 10, 30)}
model_random_search = RandomizedSearchCV(
    model, param_distributions=param_grid, n_iter=10,
    cv=5, verbose=1,
)
model_random_search.fit(data, target)
```

## 8. Model selection

### [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html)

Evaluate metric(s) by cross-validation and also record fit/score times. `scoring` parameters is used to define which metric(s) will be computed during each fold. In the `cv` parameter, one can pass any type of splitting strategy: k-fold, stratified and etc.

```python
from sklearn.model_selection import cross_validate
cv_results = cross_validate(
    model, data, target, cv=5, scoring="neg_mean_absolute_error")
```

### [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)

Identical to calling the `cross_validate` function and to select the test score only.

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, data, target)
```

### [validation_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html)

Determine training and test scores for varying parameter values.

```python
from sklearn.model_selection import validation_curve
param_A = [1, 5, 10, 15, 20, 25]
train_scores, test_scores = validation_curve(
    model, data, target, param_name="param_A", param_range=param_A,
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=2)
```

### [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

K-Folds cross-validator.

```python
from sklearn.model_selection import KFold
cv = KFold(n_splits=2)
cv..get_n_splits(data)
```
### [ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html)

Random permutation cross-validator.

```python
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, random_state=0)
cv.get_n_splits(data)
```

### [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

Stratified K-Folds cross-validator, generates test sets such that all contain the same distribution of classes, or as close as possible.

```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=2)
cv.get_n_splits(data, target)
```

### [GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)

K-fold iterator variant with non-overlapping groups, which makes each group appear exactly once in the test set across all folds. `groups` should be an array of the same length of data. For each row `groups` should indicate which group it belongs to.

```python
from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=2)
cv.get_n_splits(data, target, groups=groups)
```

### [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

Time Series cross-validator, provides train/test indices to split time series data samples that are observed at fixed time intervals, in train/test sets.

```python
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=2)
cv.get_n_splits(data, target)
```

### [LeaveOneGroupOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html)

Leave One Group Out cross-validator, provides train/test indices to split data such that each training set is comprised of all samples except ones belonging to one specific group. `groups` should be an array of the same length of data. For each row `groups` should indicate which group it belongs to.

```python
from sklearn.model_selection import LeaveOneGroupOut
cv = LeaveOneGroupOut()
cv.get_n_splits(data, target, groups=groups)
```

## 9. Dummy models

### [DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)

Predicts the same value based on a (simple) rule without using training features. `strategy` can be `{“mean”, “median”, “quantile”, “constant”}`.

```python
from sklearn.dummy import DummyRegressor
model = DummyRegressor(strategy="mean")
model.fit(data, target)
```

### [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)

Predicts the same class based on a (simple) rule without using training features. `strategy` can be `{“most_frequent”, “prior”, “stratified”, “uniform”, “constant”}`.

```python
from sklearn.dummy import DummyClassifier
model = DummyClassifier(strategy="most_frequent")
model.fit(data, target)
```

## 10. Linear models

### [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

Ordinary least squares Linear Regression.

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(data, target)
```

### [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

Linear least squares with l2 regularization. `alpha` parameter defines the l2 multiplier coefficient.

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(data, target)
```

### [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)

Ridge regression with built-in cross-validation. `alphas` defines the array of alpha values to try.

```python
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
model.fit(data, target)
```

### [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

Logistic Regression classifier. `penalty` parameter is by default `l2`, `C` defines inverse of regularization strength, must be a positive float.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C = 1.0)
model.fit(data, target)
```

## 11. kNN

### [KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor)

Regression based on k-nearest neighbors. `n_neighbors` defines number of neighbors to use.

```python
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=2)
model.fit(data, target)
```

### [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

Regression based on k-nearest neighbors. `n_neighbors` defines number of neighbors to use.

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsRegressor(n_neighbors=2)
model.fit(data, target)
```

## 12. Tree models

### [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)

A decision tree regressor. `max_depth` defines the maximum depth of a tree.

```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=2)
model.fit(data, target)
```

### [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)

A decision tree classifier. `max_depth` defines the maximum depth of a tree.

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=2)
model.fit(data, target)
```

## 12. Ensemble models

### [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

Gradient Boosting trees for regression. `max_depth` defines the maximum depth of a tree, `learning_rate` defines the "contribution" of each tree, `n_estimators` controls the number of trained trees.

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(max_depth=2, learning_rate=1.0, n_estimators=100)
model.fit(data, target)
```

### [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

Gradient Boosting for classification. `max_depth` defines the maximum depth of a tree, `learning_rate` defines the "contribution" of each tree, `n_estimators` controls the number of trained trees.

```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingRegressor(max_depth=2, learning_rate=1.0, n_estimators=100)
model.fit(data, target)
```

### [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

A random forest regressor. `max_depth` defines the maximum depth of a tree, `n_estimators` controls the number of trained trees.

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=2, n_estimators=100)
model.fit(data, target)
```

### [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

A random forest classifier. `max_depth` defines the maximum depth of a tree, `n_estimators` controls the number of trained trees.

```
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=2, n_estimators=100)
model.fit(data, target)
```
