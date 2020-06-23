import numpy as np
import warnings
import os
import pickle as pkl
from sklearn.metrics.scorer import _BaseScorer
from torch.utils.data import DataLoader

from solnml.components.ensemble.base_ensemble import BaseImgEnsembleModel
from solnml.components.utils.constants import CLS_TASKS
from solnml.components.evaluators.base_dl_evaluator import get_estimator_with_parameters


class Blending(BaseImgEnsembleModel):
    def __init__(self, stats,
                 ensemble_size: int,
                 task_type: int,
                 metric: _BaseScorer,
                 output_dir=None,
                 device='cpu',
                 meta_learner='lightgbm'):
        super().__init__(stats=stats,
                         ensemble_method='blending',
                         ensemble_size=ensemble_size,
                         task_type=task_type,
                         metric=metric,
                         output_dir=output_dir,
                         device=device)
        try:
            from lightgbm import LGBMClassifier
        except:
            warnings.warn("Lightgbm is not imported! Blending will use linear model instead!")
            meta_learner = 'linear'
        self.meta_method = meta_learner
        # We use Xgboost as default meta-learner
        if self.task_type in CLS_TASKS:
            if meta_learner == 'linear':
                from sklearn.linear_model.logistic import LogisticRegression
                self.meta_learner = LogisticRegression(max_iter=1000)
            elif meta_learner == 'gb':
                from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
                self.meta_learner = GradientBoostingClassifier(learning_rate=0.05, subsample=0.7, max_depth=4,
                                                               n_estimators=250)
            elif meta_learner == 'lightgbm':
                from lightgbm import LGBMClassifier
                self.meta_learner = LGBMClassifier(max_depth=4, learning_rate=0.05, n_estimators=150)
        else:
            if meta_learner == 'linear':
                from sklearn.linear_model import LinearRegression
                self.meta_learner = LinearRegression()
            elif meta_learner == 'lightgbm':
                from lightgbm import LGBMRegressor
                self.meta_learner = LGBMRegressor(max_depth=4, learning_rate=0.05, n_estimators=70)

    def fit(self, train_data):
        # Train basic models using a part of training data
        model_cnt = 0
        feature_p2 = None

        if hasattr(train_data, 'val_dataset'):
            loader = DataLoader(train_data.val_dataset)
        else:
            loader = DataLoader(train_data.train_dataset, sampler=train_data.val_sampler)
        num_samples = 0
        y_p2 = list()
        for sample in loader:
            num_samples += 1
            y_p2.extend(sample[1].detach().numpy())
        y_p2 = np.array(y_p2)

        for algo_id in self.stats["include_algorithms"]:
            model_configs = self.stats[algo_id]['model_configs']
            for idx, (node, config) in enumerate(model_configs):
                estimator = get_estimator_with_parameters(config, self.output_dir)
                if self.task_type in CLS_TASKS:
                    if hasattr(train_data, 'val_dataset'):
                        pred = estimator.predict_proba(train_data.val_dataset)
                    else:
                        pred = estimator.predict_proba(train_data.train_dataset, sampler=train_data.val_sampler)
                    n_dim = np.array(pred).shape[1]
                    if n_dim == 2:
                        # Binary classificaion
                        n_dim = 1
                    # Initialize training matrix for phase 2
                    if feature_p2 is None:
                        feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                    if n_dim == 1:
                        feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] = pred[:, 1:2]
                    else:
                        feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] = pred
                else:
                    if hasattr(train_data, 'val_dataset'):
                        pred = estimator.predict(train_data.val_dataset)
                    else:
                        pred = estimator.predict(train_data.train_dataset, sampler=train_data.val_sampler)
                    pred = pred.reshape(-1, 1)
                    n_dim = 1
                    # Initialize training matrix for phase 2
                    if feature_p2 is None:
                        feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                    feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] = pred
                model_cnt += 1
        self.meta_learner.fit(feature_p2, y_p2)

        return self

    def get_feature(self, data, sampler=None):
        # Predict the labels via blending
        feature_p2 = None
        model_cnt = 0

        loader = DataLoader(data, sampler=sampler)
        num_samples = len(list(loader))

        for algo_id in self.stats["include_algorithms"]:
            model_configs = self.stats[algo_id]['model_configs']
            for idx, (node, config) in enumerate(model_configs):
                estimator = get_estimator_with_parameters(config, device=self.device, model_dir=self.output_dir)
                if self.task_type in CLS_TASKS:
                    pred = estimator.predict_proba(data, sampler=sampler)
                    n_dim = np.array(pred).shape[1]
                    if n_dim == 2:
                        # Binary classificaion
                        n_dim = 1
                    # Initialize training matrix for phase 2
                    if feature_p2 is None:
                        feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                    if n_dim == 1:
                        feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] = pred[:, 1:2]
                    else:
                        feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] = pred
                else:
                    pred = estimator.predict_proba(data, sampler=sampler)
                    pred = pred.reshape(-1, 1)
                    n_dim = 1
                    # Initialize training matrix for phase 2
                    if feature_p2 is None:
                        feature_p2 = np.zeros((num_samples, self.ensemble_size * n_dim))
                    feature_p2[:, model_cnt * n_dim:(model_cnt + 1) * n_dim] = pred
                model_cnt += 1

        return feature_p2

    def predict(self, data, sampler=None):
        feature_p2 = self.get_feature(data, sampler=sampler)
        # Get predictions from meta-learner
        if self.task_type in CLS_TASKS:
            final_pred = self.meta_learner.predict_proba(feature_p2)
        else:
            final_pred = self.meta_learner.predict(feature_p2)
        return final_pred

    def get_ens_model_info(self):
        raise NotImplementedError