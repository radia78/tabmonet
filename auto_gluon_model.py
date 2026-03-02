import numpy as np
import pandas as pd

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

from sklearn.preprocessing import (
    QuantileTransformer, 
    StandardScaler,
)

class TabularMONet(AbstractModel):
    def __init__(self, **kwargs):
        # Simply pass along kwargs to parent, and init our internal `_feature_generator` variable to None
        super().__init__(**kwargs)
        self._target_encoder = None
        self._cat_encoder = None
        self._cont_encoder = None
        self.cat_columns = None
        self.cont_columns = None
        self.max_cat = None

    def _preprocess_train(self, X, y, X_val=None, y_val=None):
        X = self.preprocess(X, fit=True)
        
        if X_val is not None:
            X_val = self.preprocess(X_val, fit=False)
        
        if self.problem_type in ["binary", "multiclass"]:
            self._target_encoder = LabelEncoderFeatureGenerator(verbosity=0)
            y
        else:
            self._target_encoder = StandardScaler()

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs):
        X = super()._preprocess(X, **kwargs)

        if is_train:
            self.cont_columns = X.select_dtypes(include="number").columns.tolist()
            self.cat_columns = X.select_dtypes(exclude="number").columns.tolist()

            self._cont_encoder = QuantileTransformer(output_distribution='normal')
            self._cat_encoder = LabelEncoderFeatureGenerator(verbosity=0)

            # Fit them separately and retain the max-class inforamtion automatically
            if self.cat_columns:
                X_cat = self._cat_encoder.fit_transform(X[self.cat_columns])
                X_stats = X[self.cat_columns].describe(include="all")["unique"]
                # TODO: Abandon any feature with a cardinality of 10000
                self.max_cat = X_stats.max()

            if self.cont_columns:
                X_cont = self._cont_encoder.fit_transform(X[self.cont_columns])

        if self.cat_columns:
            X_cat = self._cat_encoder.transform(X[self.cat_columns])
            
            if self.cont_columns:
                 
            
        if self.cont_columns:
            X_cont = self._cont_encoder.transform(X[self.cont_columns])

        

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self,
             X: pd.DataFrame,  # training data
             y: pd.Series,  # training labels
             # X_val=None,  # val data (unused in RF model)
             # y_val=None,  # val labels (unused in RF model)
             # time_limit=None,  # time limit in seconds (ignored in tutorial)
             **kwargs):  # kwargs includes many other potential inputs, refer to AbstractModel documentation for details
        print('Entering the `_fit` method')

        # First we import the required dependencies for the model. Note that we do not import them outside of the method.
        # This enables AutoGluon to be highly extensible and modular.
        # For an example of best practices when importing model dependencies, refer to LGBModel.
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Valid self.problem_type values include ['binary', 'multiclass', 'regression', 'quantile', 'softclass']
        if self.problem_type in ['regression', 'softclass']:
            model_cls = RandomForestRegressor
        else:
            model_cls = RandomForestClassifier

        # Make sure to call preprocess on X near the start of `_fit`.
        # This is necessary because the data is converted via preprocess during predict, and needs to be in the same format as during fit.
        X = self.preprocess(X, y=y, is_train=True)
        # This fetches the user-specified (and default) hyperparameters for the model.
        params = self._get_model_params()
        print(f'Hyperparameters: {params}')
        # self.model should be set to the trained inner model, so that internally during predict we can call `self.model.predict(...)`
        self.model = model_cls(**params)
        self.model.fit(X, y)
        print('Exiting the `_fit` method')

    # The `_set_default_params` method defines the default hyperparameters of the model.
    # User-specified parameters will override these values on a key-by-key basis.
    def _set_default_params(self):
        default_params = {
            'n_estimators': 300,
            'n_jobs': -1,
            'random_state': 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # The `_get_default_auxiliary_params` method defines various model-agnostic parameters such as maximum memory usage and valid input column dtypes.
    # For most users who build custom models, they will only need to specify the valid/invalid dtypes to the model here.
    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            # the total set of raw dtypes are: ['int', 'float', 'category', 'object', 'datetime']
            # object feature dtypes include raw text and image paths, which should only be handled by specialized models
            # datetime raw dtypes are generally converted to int in upstream pre-processing,
            # so models generally shouldn't need to explicitly support datetime dtypes.
            valid_raw_types=['int', 'float', 'category'],
            # Other options include `valid_special_types`, `ignored_type_group_raw`, and `ignored_type_group_special`.
            # Refer to AbstractModel for more details on available options.
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params