#!/usr/bin/env python
# coding: utf-8

# In[7]:


from gc import enable
import mlflow.pyfunc
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler , MinMaxScaler
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor , RandomForestRegressor , HistGradientBoostingRegressor
from sklearn.svm import SVR

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import VotingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import RegressorChain

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, VotingRegressor
import lightgbm as lgb


# In[8]:


# Define your ensemble model outside the class
# et_opt_params = {'n_estimators': 600, 'bootstrap': True, 'max_depth': 20, 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_jobs': -1}
# regressors = [
#     ('Bagged Linear Reg', BaggingRegressor(LinearRegression(), n_estimators=20, bootstrap=True)),
#     ('Extra Trees Reg', ExtraTreesRegressor(**et_opt_params))
# ]
# model = MultiOutputRegressor(VotingRegressor(estimators=regressors))
# model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
# model = MultiOutputRegressor(HistGradientBoostingRegressor(max_depth=18, max_iter=100))
lg_regressor = lgb.LGBMRegressor(max_depth = 5, learning_rate = 0.09)
model = MultiOutputRegressor(estimator=lg_regressor)


# In[9]:


class Model(mlflow.pyfunc.PythonModel):
    def __init__(self):
        """Create the Model.

        All stateful objects built from preprocessing or training should be properties of the class
        to ensure they are serialized and saved when the model is persisted to disk.
        # """
#         self.gb = GradientBoostingRegressor(random_state=42)
#         self.rf = RandomForestRegressor(random_state=42)
#         self.hg = HistGradientBoostingRegressor(random_state=42)
#         self.vr = VotingRegressor(estimators=[('gb',self.gb),('rf',self.rf),('hg',self.hg)])
       
        self.model = model
        # craete our final ensemble model
        self.imputer = KNNImputer()
        # self.model = RegressorChain(GradientBoostingRegressor(random_state=42))
        # self.model = MultiOutputRegressor(self.vr)
        # self.model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
        # self.model = LinearRegression()
        self.scale = StandardScaler()
        

    def preprocess(self, input):
        """Preprocess the scan data collected for a single core.

        Features may be engineered on a row by row basis or create features from aggregate data within
        the entire core, to find predictive power in the trends of scan data. Any number of rows may
        be returned and will subsequently be passed into the `train` method.

        Args:
            input (pandas.DataFrame): All the 10cm scan data collected for a single core.

        Returns:
            pandas.DataFrame: Any number of rows representing the preprocessed training set for a single core.
                Targets will be joined against each row returned from this function.
        """

        # Sample feature that aggregates the mean of 'Feature 2' across all 10cm rows.
        #input['Feature 2'] = input['Feature 2'].mean()
        
        
        df_input = input.copy()
        df_input = df_input.iloc[:,3:]
        df = df_input.median()
        df = df.to_frame().T
        df['from'] = input['from'].min()
        df['HolNum'] = input['HolNum'].mean()
        df['to'] = input['to'].max()
        
        return df
    
    def train(self, X_train, y_train):
        """Train the model.

        Args:
            X_train (pandas.DataFrame): A combined dataset with the results of `preprocess` for all cores, with forbidden columns dropped.
            y_train (pandas.DataFrame): The targets associated with each row in X_train.
        """
        
        # Deal with missing values by dropping any rows that are null in the target set.
        idx = ~np.isnan(y_train).any(axis=1)
        y_train = y_train[idx]
        X_train = X_train[idx]

        # Impute missing values in feature set.
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_imputed = self.scale.fit_transform(X_train_imputed)
        self.model.fit(X_train_imputed, y_train)

    def predict(self, context, X_test, params=None):
        """Create predictions for a single core.

        Args:
            X_test (pandas.DataFrame): Data from the public/private set, after being preprocessed, with forbidden columns dropped.

        Returns:
            array: Predictions made by the model, at any level of granularity, which will later be aggregated in `postprocess`.
        """
        X_test_imputed = self.imputer.transform(X_test)
        X_test_imputed = self.scale.transform(X_test_imputed)
                                            
        predictions = self.model.predict(X_test_imputed)
        
        return predictions

    def postprocess(self, predictions, preprocessed_core):
        """Post process predictions for a single core.

        Args:
            predictions (array): The predictions produced by `predict` for the given core.
            preprocessed_core (pandas.DataFrame): The preprocessed core for which predictions are being made.

        Returns:
            array: A 1d array containing 1 set of predictions for a single core.
        """
        
        return pd.DataFrame(predictions).agg('mean')


# 
# class Model(mlflow.pyfunc.PythonModel):
#     def __init__(self):
#         """Create the Model.
# 
#         All stateful objects built from preprocessing or training should be properties of the class
#         to ensure they are serialized and saved when the model is persisted to disk.
#         """
#         self.imputer = SimpleImputer(strategy="mean")
#         self.model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
# 
#     def preprocess(self, input):
#         """Preprocess the scan data collected for a single core.
# 
#         Features may be engineered on a row by row basis or create features from aggregate data within
#         the entire core, to find predictive power in the trends of scan data. Any number of rows may
#         be returned and will subsequently be passed into the `train` method.
# 
#         Args:
#             input (pandas.DataFrame): All the 10cm scan data collected for a single core.
# 
#         Returns:
#             pandas.DataFrame: Any number of rows representing the preprocessed training set for a single core.
#                 Targets will be joined against each row returned from this function.
#         """
# 
#         # Sample feature that aggregates the mean of 'Feature 2' across all 10cm rows.
#         #input['Feature 2'] = input['Feature 2'].mean()
#         for i in range(1, 50):
#             column_name = f'Feature {i}'  # Generate the column name
#             input[column_name] = input[column_name].mean()
# 
#         return input
# 
#     def train(self, X_train, y_train):
#         """Train the model.
# 
#         Args:
#             X_train (pandas.DataFrame): A combined dataset with the results of `preprocess` for all cores, with forbidden columns dropped.
#             y_train (pandas.DataFrame): The targets associated with each row in X_train.
#         """
#         
#         # Deal with missing values by dropping any rows that are null in the target set.
#         idx = ~np.isnan(y_train).any(axis=1)
#         y_train = y_train[idx]
#         X_train = X_train[idx]
# 
#         # Impute missing values in feature set.
#         X_train_imputed = self.imputer.fit_transform(X_train)
# 
#         self.model.fit(X_train_imputed, y_train)
# 
#     def predict(self, context, X_test, params=None):
#         """Create predictions for a single core.
# 
#         Args:
#             X_test (pandas.DataFrame): Data from the public/private set, after being preprocessed, with forbidden columns dropped.
# 
#         Returns:
#             array: Predictions made by the model, at any level of granularity, which will later be aggregated in `postprocess`.
#         """
#         X_test_imputed = self.imputer.transform(X_test)
#         predictions = self.model.predict(X_test_imputed)
# 
#         return predictions
# 
#     def postprocess(self, predictions, preprocessed_core):
#         """Post process predictions for a single core.
# 
#         Args:
#             predictions (array): The predictions produced by `predict` for the given core.
#             preprocessed_core (pandas.DataFrame): The preprocessed core for which predictions are being made.
# 
#         Returns:
#             array: A 1d array containing 1 set of predictions for a single core.
#         """
#         return pd.DataFrame(predictions).agg("mean")
# 

# In[ ]:




