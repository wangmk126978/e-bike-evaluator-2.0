from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np 
import bike_evaluator as be


def merge_train_set():
    k=12
    num_for_each_k=350
    train_x_set,train_y_set,test_x,test_y=be.creat_random_cross_valida_sets(feature_vectors,labels,k,num_for_each_k)
    train_x=[]
    train_y=[]
    for i in range(len(train_x_set)):
        for j in range(len(train_x_set[i])):
            train_x.append(train_x_set[i][j])
            train_y.append(train_y_set[i][j])
    return train_x,train_y,test_x,test_y


def get_mse(feature_vectors,labels):
    regr = RandomForestRegressor(n_estimators=200,oob_score=True)
    regr.fit(feature_vectors, labels)  
    oob=1 - regr.oob_score_ 
    return oob

#展示预测曲线
def show_model_performance(test_x,test_y):
    pred=regr.predict(test_x) 
    mix=[]
    for i in range(len(pred)):
        mix.append([test_y[i][0],pred[i]])
    mix=sorted(mix, key=lambda x:x[0],reverse = True)  
    mix=np.array(mix)
    X=list(range(len(mix)))
    plt.figure()
    plt.plot(mix[:,0],mix[:,0],color='r',label='real score')
    plt.scatter(mix[:,0],mix[:,1],marker='x',color='g',label='predict score')
    plt.title('random forest performance on the entire samples')
    plt.legend()
    plt.show()

start_date='20181001'
end_date='20190710'
source='elektrischefietsen'
selected_column,labels_source,en_dic,uncode_feature_vectors,feature_vectors,labels,bike_names=be.creating_time_series_data(start_date,end_date,source)

train_x,train_y,test_x,test_y=merge_train_set()
regr = RandomForestRegressor(n_estimators=200,oob_score=False,bootstrap=True)
regr.fit(feature_vectors, labels)
show_model_performance(test_x,test_y)
show_model_performance(feature_vectors,labels)
mse=mean_squared_error(regr.predict(test_x),test_y)
#print(1 - regr.oob_score_)
feature_importances=regr.feature_importances_
#oob=regr.oob_score_ 

def test_performance(train_x,train_y,test_x,test_y,n):
    regr = RandomForestRegressor(n_estimators=n,oob_score=False,bootstrap=True)
    regr.fit(train_x, train_y)
    mse=mean_squared_error(regr.predict(test_x),test_y)
    return mse

    

#保存模型
#joblib.dump(regr, "./RF_saver/RF_model.m")

#知道了feature importance了，计算每一类组件的重要性
attributes=['brand','frame color','wheel color','frame type','battery position','engine position','front carrier','bike type','saddle color','gears','engine brand','price']
components_importance=[]
index=0
for i in range(len(en_dic)):
    sub_com=[]
    sum_importance=[]
    for j in range(len(en_dic[i])):
        sub_com.append([en_dic[i][j][0],feature_importances[index]])
        sum_importance.append(feature_importances[index])
        index+=1
    sub_com=sorted(sub_com, key=lambda x:x[1],reverse = True)
    components_importance.append([[attributes[i],np.mean(sum_importance)],sub_com])
components_importance=sorted(components_importance, key=lambda x:x[0][1],reverse = True)

#大类的排序
a=[]
sum_a=0
for j in range(len(components_importance)):
    a.append(components_importance[j][0])
    sum_a+=components_importance[j][0][1]
for i in range(len(a)):
    a[i][1]=round(a[i][1]/sum_a,2)

#小类的排序
a=[]
sum_a=0
com=9
for i in range(len(components_importance[com][1])):
    a.append(components_importance[com][1][i])
    sum_a+=components_importance[com][1][i][1]
for i in range(len(a)):
    a[i][1]=round(a[i][1]/sum_a,2)
    

    
#计算n_estimators对结果的影响曲线
n_estimators=[]
mse=[]
for i in range(200):
    n_estimators.append(i+1)
    regr = RandomForestRegressor(n_estimators=i+1,oob_score=True)
    regr.fit(feature_vectors, labels)  
    oob=1 - regr.oob_score_
    mse.append(oob)
    
plt.figure()
plt.plot(n_estimators,mse)
plt.title('The relationship between the decision trees number and the MSE of RF')
plt.xlabel('decision trees number')       #x轴的标签
plt.ylabel('MSE of RF') 
plt.show()




#调回模型
#regr = joblib.load("./RF_saver/RF_model.m")

#为了benchmark RF, ANN, KNN做的测试,首先要import ANN_import.spydata来看test_x有哪些
removed_train_x=[]
removed_train_y=[]
for i in range(len(feature_vectors)):
    for j in range(len(test_x)):
        if list(test_x[j]) == list(feature_vectors[i]):
            if test_y[j][0] == labels[i][0]:
                break
        if j == len(test_x)-1:
            removed_train_x.append(feature_vectors[i])
            removed_train_y.append(labels[i])
regr = RandomForestRegressor(n_estimators=200,oob_score=False,bootstrap=True)
regr.fit(removed_train_x, removed_train_y)
RF_pred=regr.predict(test_x) 

RF_mse=mean_squared_error(RF_pred,test_y)



###############################################################################
'''
帮助
Help on class RandomForestRegressor in module sklearn.ensemble.forest:

class RandomForestRegressor(ForestRegressor)
 |  RandomForestRegressor(n_estimators='warn', criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)
 |  
 |  A random forest regressor.
 |  
 |  A random forest is a meta estimator that fits a number of classifying
 |  decision trees on various sub-samples of the dataset and uses averaging
 |  to improve the predictive accuracy and control over-fitting.
 |  The sub-sample size is always the same as the original
 |  input sample size but the samples are drawn with replacement if
 |  `bootstrap=True` (default).
 |  
 |  Read more in the :ref:`User Guide <forest>`.
 |  
 |  Parameters
 |  ----------
 |  n_estimators : integer, optional (default=10)
 |      The number of trees in the forest.
 |  
 |      .. versionchanged:: 0.20
 |         The default value of ``n_estimators`` will change from 10 in
 |         version 0.20 to 100 in version 0.22.
 |  
 |  criterion : string, optional (default="mse")
 |      The function to measure the quality of a split. Supported criteria
 |      are "mse" for the mean squared error, which is equal to variance
 |      reduction as feature selection criterion, and "mae" for the mean
 |      absolute error.
 |  
 |      .. versionadded:: 0.18
 |         Mean Absolute Error (MAE) criterion.
 |  
 |  max_depth : integer or None, optional (default=None)
 |      The maximum depth of the tree. If None, then nodes are expanded until
 |      all leaves are pure or until all leaves contain less than
 |      min_samples_split samples.
 |  
 |  min_samples_split : int, float, optional (default=2)
 |      The minimum number of samples required to split an internal node:
 |  
 |      - If int, then consider `min_samples_split` as the minimum number.
 |      - If float, then `min_samples_split` is a fraction and
 |        `ceil(min_samples_split * n_samples)` are the minimum
 |        number of samples for each split.
 |  
 |      .. versionchanged:: 0.18
 |         Added float values for fractions.
 |  
 |  min_samples_leaf : int, float, optional (default=1)
 |      The minimum number of samples required to be at a leaf node.
 |      A split point at any depth will only be considered if it leaves at
 |      least ``min_samples_leaf`` training samples in each of the left and
 |      right branches.  This may have the effect of smoothing the model,
 |      especially in regression.
 |  
 |      - If int, then consider `min_samples_leaf` as the minimum number.
 |      - If float, then `min_samples_leaf` is a fraction and
 |        `ceil(min_samples_leaf * n_samples)` are the minimum
 |        number of samples for each node.
 |  
 |      .. versionchanged:: 0.18
 |         Added float values for fractions.
 |  
 |  min_weight_fraction_leaf : float, optional (default=0.)
 |      The minimum weighted fraction of the sum total of weights (of all
 |      the input samples) required to be at a leaf node. Samples have
 |      equal weight when sample_weight is not provided.
 |  
 |  max_features : int, float, string or None, optional (default="auto")
 |      The number of features to consider when looking for the best split:
 |  
 |      - If int, then consider `max_features` features at each split.
 |      - If float, then `max_features` is a fraction and
 |        `int(max_features * n_features)` features are considered at each
 |        split.
 |      - If "auto", then `max_features=n_features`.
 |      - If "sqrt", then `max_features=sqrt(n_features)`.
 |      - If "log2", then `max_features=log2(n_features)`.
 |      - If None, then `max_features=n_features`.
 |  
 |      Note: the search for a split does not stop until at least one
 |      valid partition of the node samples is found, even if it requires to
 |      effectively inspect more than ``max_features`` features.
 |  
 |  max_leaf_nodes : int or None, optional (default=None)
 |      Grow trees with ``max_leaf_nodes`` in best-first fashion.
 |      Best nodes are defined as relative reduction in impurity.
 |      If None then unlimited number of leaf nodes.
 |  
 |  min_impurity_decrease : float, optional (default=0.)
 |      A node will be split if this split induces a decrease of the impurity
 |      greater than or equal to this value.
 |  
 |      The weighted impurity decrease equation is the following::
 |  
 |          N_t / N * (impurity - N_t_R / N_t * right_impurity
 |                              - N_t_L / N_t * left_impurity)
 |  
 |      where ``N`` is the total number of samples, ``N_t`` is the number of
 |      samples at the current node, ``N_t_L`` is the number of samples in the
 |      left child, and ``N_t_R`` is the number of samples in the right child.
 |  
 |      ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
 |      if ``sample_weight`` is passed.
 |  
 |      .. versionadded:: 0.19
 |  
 |  min_impurity_split : float, (default=1e-7)
 |      Threshold for early stopping in tree growth. A node will split
 |      if its impurity is above the threshold, otherwise it is a leaf.
 |  
 |      .. deprecated:: 0.19
 |         ``min_impurity_split`` has been deprecated in favor of
 |         ``min_impurity_decrease`` in 0.19. The default value of
 |         ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
 |         will be removed in 0.25. Use ``min_impurity_decrease`` instead.
 |  
 |  bootstrap : boolean, optional (default=True)
 |      Whether bootstrap samples are used when building trees. If False, the
 |      whole datset is used to build each tree.
 |  
 |  oob_score : bool, optional (default=False)
 |      whether to use out-of-bag samples to estimate
 |      the R^2 on unseen data.
 |  
 |  n_jobs : int or None, optional (default=None)
 |      The number of jobs to run in parallel for both `fit` and `predict`.
 |      `None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
 |      ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
 |      for more details.
 |  
 |  random_state : int, RandomState instance or None, optional (default=None)
 |      If int, random_state is the seed used by the random number generator;
 |      If RandomState instance, random_state is the random number generator;
 |      If None, the random number generator is the RandomState instance used
 |      by `np.random`.
 |  
 |  verbose : int, optional (default=0)
 |      Controls the verbosity when fitting and predicting.
 |  
 |  warm_start : bool, optional (default=False)
 |      When set to ``True``, reuse the solution of the previous call to fit
 |      and add more estimators to the ensemble, otherwise, just fit a whole
 |      new forest. See :term:`the Glossary <warm_start>`.
 |  
 |  Attributes
 |  ----------
 |  estimators_ : list of DecisionTreeRegressor
 |      The collection of fitted sub-estimators.
 |  
 |  feature_importances_ : array of shape = [n_features]
 |      The feature importances (the higher, the more important the feature).
 |  
 |  n_features_ : int
 |      The number of features when ``fit`` is performed.
 |  
 |  n_outputs_ : int
 |      The number of outputs when ``fit`` is performed.
 |  
 |  oob_score_ : float
 |      Score of the training dataset obtained using an out-of-bag estimate.
 |  
 |  oob_prediction_ : array of shape = [n_samples]
 |      Prediction computed with out-of-bag estimate on the training set.
 |  
 |  Examples
 |  --------
 |  >>> from sklearn.ensemble import RandomForestRegressor
 |  >>> from sklearn.datasets import make_regression
 |  
 |  >>> X, y = make_regression(n_features=4, n_informative=2,
 |  ...                        random_state=0, shuffle=False)
 |  >>> regr = RandomForestRegressor(max_depth=2, random_state=0,
 |  ...                              n_estimators=100)
 |  >>> regr.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
 |  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
 |             max_features='auto', max_leaf_nodes=None,
 |             min_impurity_decrease=0.0, min_impurity_split=None,
 |             min_samples_leaf=1, min_samples_split=2,
 |             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
 |             oob_score=False, random_state=0, verbose=0, warm_start=False)
 |  >>> print(regr.feature_importances_)
 |  [0.18146984 0.81473937 0.00145312 0.00233767]
 |  >>> print(regr.predict([[0, 0, 0, 0]]))
 |  [-8.32987858]
 |  
 |  Notes
 |  -----
 |  The default values for the parameters controlling the size of the trees
 |  (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
 |  unpruned trees which can potentially be very large on some data sets. To
 |  reduce memory consumption, the complexity and size of the trees should be
 |  controlled by setting those parameter values.
 |  
 |  The features are always randomly permuted at each split. Therefore,
 |  the best found split may vary, even with the same training data,
 |  ``max_features=n_features`` and ``bootstrap=False``, if the improvement
 |  of the criterion is identical for several splits enumerated during the
 |  search of the best split. To obtain a deterministic behaviour during
 |  fitting, ``random_state`` has to be fixed.
 |  
 |  The default value ``max_features="auto"`` uses ``n_features``
 |  rather than ``n_features / 3``. The latter was originally suggested in
 |  [1], whereas the former was more recently justified empirically in [2].
 |  
 |  References
 |  ----------
 |  
 |  .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.
 |  
 |  .. [2] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
 |         trees", Machine Learning, 63(1), 3-42, 2006.
 |  
 |  See also
 |  --------
 |  DecisionTreeRegressor, ExtraTreesRegressor
 |  
 |  Method resolution order:
 |      RandomForestRegressor
 |      ForestRegressor
 |      BaseForest
 |      sklearn.ensemble.base.BaseEnsemble
 |      sklearn.base.BaseEstimator
 |      sklearn.base.MetaEstimatorMixin
 |      sklearn.base.MultiOutputMixin
 |      sklearn.base.RegressorMixin
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __init__(self, n_estimators='warn', criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)
 |      Initialize self.  See help(type(self)) for accurate signature.
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  __abstractmethods__ = frozenset()
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from ForestRegressor:
 |  
 |  predict(self, X)
 |      Predict regression target for X.
 |      
 |      The predicted regression target of an input sample is computed as the
 |      mean predicted regression targets of the trees in the forest.
 |      
 |      Parameters
 |      ----------
 |      X : array-like or sparse matrix of shape = [n_samples, n_features]
 |          The input samples. Internally, its dtype will be converted to
 |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
 |          converted into a sparse ``csr_matrix``.
 |      
 |      Returns
 |      -------
 |      y : array of shape = [n_samples] or [n_samples, n_outputs]
 |          The predicted values.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from BaseForest:
 |  
 |  apply(self, X)
 |      Apply trees in the forest to X, return leaf indices.
 |      
 |      Parameters
 |      ----------
 |      X : array-like or sparse matrix, shape = [n_samples, n_features]
 |          The input samples. Internally, its dtype will be converted to
 |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
 |          converted into a sparse ``csr_matrix``.
 |      
 |      Returns
 |      -------
 |      X_leaves : array_like, shape = [n_samples, n_estimators]
 |          For each datapoint x in X and for each tree in the forest,
 |          return the index of the leaf x ends up in.
 |  
 |  decision_path(self, X)
 |      Return the decision path in the forest
 |      
 |      .. versionadded:: 0.18
 |      
 |      Parameters
 |      ----------
 |      X : array-like or sparse matrix, shape = [n_samples, n_features]
 |          The input samples. Internally, its dtype will be converted to
 |          ``dtype=np.float32``. If a sparse matrix is provided, it will be
 |          converted into a sparse ``csr_matrix``.
 |      
 |      Returns
 |      -------
 |      indicator : sparse csr array, shape = [n_samples, n_nodes]
 |          Return a node indicator matrix where non zero elements
 |          indicates that the samples goes through the nodes.
 |      
 |      n_nodes_ptr : array of size (n_estimators + 1, )
 |          The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
 |          gives the indicator value for the i-th estimator.
 |  
 |  fit(self, X, y, sample_weight=None)
 |      Build a forest of trees from the training set (X, y).
 |      
 |      Parameters
 |      ----------
 |      X : array-like or sparse matrix of shape = [n_samples, n_features]
 |          The training input samples. Internally, its dtype will be converted
 |          to ``dtype=np.float32``. If a sparse matrix is provided, it will be
 |          converted into a sparse ``csc_matrix``.
 |      
 |      y : array-like, shape = [n_samples] or [n_samples, n_outputs]
 |          The target values (class labels in classification, real numbers in
 |          regression).
 |      
 |      sample_weight : array-like, shape = [n_samples] or None
 |          Sample weights. If None, then samples are equally weighted. Splits
 |          that would create child nodes with net zero or negative weight are
 |          ignored while searching for a split in each node. In the case of
 |          classification, splits are also ignored if they would result in any
 |          single class carrying a negative weight in either child node.
 |      
 |      Returns
 |      -------
 |      self : object
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from BaseForest:
 |  
 |  feature_importances_
 |      Return the feature importances (the higher, the more important the
 |         feature).
 |      
 |      Returns
 |      -------
 |      feature_importances_ : array, shape = [n_features]
 |          The values of this array sum to 1, unless all trees are single node
 |          trees consisting of only the root node, in which case it will be an
 |          array of zeros.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from sklearn.ensemble.base.BaseEnsemble:
 |  
 |  __getitem__(self, index)
 |      Returns the index'th estimator in the ensemble.
 |  
 |  __iter__(self)
 |      Returns iterator over estimators in the ensemble.
 |  
 |  __len__(self)
 |      Returns the number of estimators in the ensemble.
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from sklearn.base.BaseEstimator:
 |  
 |  __getstate__(self)
 |  
 |  __repr__(self, N_CHAR_MAX=700)
 |      Return repr(self).
 |  
 |  __setstate__(self, state)
 |  
 |  get_params(self, deep=True)
 |      Get parameters for this estimator.
 |      
 |      Parameters
 |      ----------
 |      deep : boolean, optional
 |          If True, will return the parameters for this estimator and
 |          contained subobjects that are estimators.
 |      
 |      Returns
 |      -------
 |      params : mapping of string to any
 |          Parameter names mapped to their values.
 |  
 |  set_params(self, **params)
 |      Set the parameters of this estimator.
 |      
 |      The method works on simple estimators as well as on nested objects
 |      (such as pipelines). The latter have parameters of the form
 |      ``<component>__<parameter>`` so that it's possible to update each
 |      component of a nested object.
 |      
 |      Returns
 |      -------
 |      self
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from sklearn.base.BaseEstimator:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Methods inherited from sklearn.base.RegressorMixin:
 |  
 |  score(self, X, y, sample_weight=None)
 |      Returns the coefficient of determination R^2 of the prediction.
 |      
 |      The coefficient R^2 is defined as (1 - u/v), where u is the residual
 |      sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
 |      sum of squares ((y_true - y_true.mean()) ** 2).sum().
 |      The best possible score is 1.0 and it can be negative (because the
 |      model can be arbitrarily worse). A constant model that always
 |      predicts the expected value of y, disregarding the input features,
 |      would get a R^2 score of 0.0.
 |      
 |      Parameters
 |      ----------
 |      X : array-like, shape = (n_samples, n_features)
 |          Test samples. For some estimators this may be a
 |          precomputed kernel matrix instead, shape = (n_samples,
 |          n_samples_fitted], where n_samples_fitted is the number of
 |          samples used in the fitting for the estimator.
 |      
 |      y : array-like, shape = (n_samples) or (n_samples, n_outputs)
 |          True values for X.
 |      
 |      sample_weight : array-like, shape = [n_samples], optional
 |          Sample weights.
 |      
 |      Returns
 |      -------
 |      score : float
 |          R^2 of self.predict(X) wrt. y.
 |      
 |      Notes
 |      -----
 |      The R2 score used when calling ``score`` on a regressor will use
 |      ``multioutput='uniform_average'`` from version 0.23 to keep consistent
 |      with `metrics.r2_score`. This will influence the ``score`` method of
 |      all the multioutput regressors (except for
 |      `multioutput.MultiOutputRegressor`). To specify the default value
 |      manually and avoid the warning, please either call `metrics.r2_score`
 |      directly or make a custom scorer with `metrics.make_scorer` (the
 |      built-in scorer ``'r2'`` uses ``multioutput='uniform_average'``).
'''