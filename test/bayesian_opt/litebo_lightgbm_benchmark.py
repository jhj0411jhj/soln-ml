"""
example cmdline:

python test/bayesian_opt/litebo_lightgbm_benchmark.py --mths lite_bo,smac,tpe --datasets wind --n 10 --rep 2 --n_jobs 4

python test/bayesian_opt/litebo_lightgbm_benchmark.py --plot_mode 1 --mths lite_bo,smac,tpe --datasets wind --n 10

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pk

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from lightgbm import LGBMClassifier

sys.path.append(".")
sys.path.insert(0, "../lite-bo")
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from solnml.components.utils.constants import MULTICLASS_CLS
from litebo.optimizer.smbo import SMBO


default_datasets = 'optdigits,satimage,wind,delta_ailerons,puma8NH,kin8nm,cpu_small,puma32H,cpu_act,bank32nh'
default_mths = 'lite_bo,smac,tpe'   # 'lite_bo,smac,tpe,sync-8,async-8'

parser = argparse.ArgumentParser()
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--plot_mode', type=int, default=0)
parser.add_argument('--datasets', type=str, default=default_datasets)
parser.add_argument('--n_jobs', type=int, default=2)
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--rep', type=int, default=1)

args = parser.parse_args()
test_datasets = args.datasets.split(',')
print("datasets num=", len(test_datasets))
mths = args.mths.split(',')
plot_mode = args.plot_mode
max_runs = args.n


class LightGBM:
    def __init__(self, n_estimators, learning_rate, num_leaves, max_depth, min_child_samples,
                 subsample, colsample_bytree, random_state=None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree

        self.n_jobs = 2
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        self.estimator = LGBMClassifier(num_leaves=self.num_leaves,
                                        max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators,
                                        min_child_samples=self.min_child_samples,
                                        subsample=self.subsample,
                                        colsample_bytree=self.colsample_bytree,
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            n_estimators = UniformFloatHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
            num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
            max_depth = Constant('max_depth', 15)
            learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
            min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 30, default_value=20)
            subsample = UniformFloatHyperparameter("subsample", 0.7, 1, default_value=1, q=0.1)
            colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
            cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample,
                                    colsample_bytree])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'n_estimators': (hp.randint('lgb_n_estimators', 19) + 2) * 50,
                     'num_leaves': hp.randint('lgb_num_leaves', 2017) + 31,
                     'max_depth': 15,
                     'learning_rate': hp.loguniform('lgb_learning_rate', np.log(1e-3), np.log(0.3)),
                     'min_child_samples': hp.randint('lgb_min_child_samples', 26) + 5,
                     'subsample': (hp.randint('lgb_subsample', 4) + 7) * 0.1,
                     'colsample_bytree': (hp.randint('lgb_colsample_bytree', 4) + 7) * 0.1,
                     }
            return space
        else:
            raise ValueError('Unknown optimizer %s when getting cs' % optimizer)


def get_configspace(optimizer='smac'):
    return LightGBM.get_hyperparameter_search_space(optimizer=optimizer)


def get_estimator(config):
    config['random_state'] = 1
    estimator = LightGBM(**config)
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', n_jobs)
    return estimator


def evaluate(mth, dataset, run_id):
    print(mth, dataset, run_id)
    train_data, test_data = load_train_test_data(dataset, test_size=0.3, task_type=MULTICLASS_CLS)

    def objective_function(config):
        metric = get_metric('bal_acc')
        estimator = get_estimator(config.get_dictionary())
        X_train, y_train = train_data.data
        X_test, y_test = test_data.data
        estimator.fit(X_train, y_train)
        return -metric(estimator, X_test, y_test)

    def tpe_objective_function(config):
        metric = get_metric('bal_acc')
        estimator = get_estimator(config)
        X_train, y_train = train_data.data
        X_test, y_test = test_data.data
        estimator.fit(X_train, y_train)
        return -metric(estimator, X_test, y_test)

    config_space = get_configspace()

    if mth == 'lite_bo':
        bo = SMBO(objective_function, config_space, max_runs=max_runs, time_limit_per_trial=600,
                  random_state=np.random.randint(10000))
        bo.run()
        perfs = bo.benchmark_perfs
    elif mth == 'smac':
        from smac.scenario.scenario import Scenario
        from smac.facade.smac_facade import SMAC
        # Scenario object
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": max_runs,
                             "cs": config_space,
                             "deterministic": "true"
                             })
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=objective_function)
        # incumbent = smac.optimize()
        # perf_bo = objective_function(incumbent)
        smac.optimize()
        # keys = [k.config_id for k in smac.runhistory.data.keys()]
        # print("key len=", len(keys), "unique=", len(set(keys)))
        perfs = [v.cost for v in smac.runhistory.data.values()]
    elif mth == 'tpe':
        config_space = get_configspace('tpe')
        from hyperopt import tpe, fmin, Trials
        trials = Trials()
        fmin(tpe_objective_function, config_space, tpe.suggest, max_runs, trials=trials)
        perfs = [trial['result']['loss'] for trial in trials.trials]
    else:
        raise ValueError('Invalid method.')
    return perfs


def check_datasets(datasets, task_type=MULTICLASS_CLS):
    for _dataset in datasets:
        try:
            _, _ = load_train_test_data(_dataset, random_state=1, task_type=task_type)
        except Exception as e:
            raise ValueError('Dataset - %s does not exist!' % _dataset)


class Timer:
    def __init__(self, name=''):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        print("[%s]Start." % self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        m, s = divmod(self.end - self.start, 60)
        h, m = divmod(m, 60)
        print("[%s]Total time = %d hours, %d minutes, %d seconds." % (self.name, h, m, s))


def descending(x):
    def _descending(x):
        y = [x[0]]
        for i in range(1, len(x)):
            y.append(min(y[-1], x[i]))
        return y

    if isinstance(x[0], list):
        y = []
        for xi in x:
            y.append(_descending(xi))
        return y
    else:
        return _descending(x)


if plot_mode != 1:
    n_jobs = args.n_jobs
    rep = args.rep

    with Timer('All'):
        check_datasets(test_datasets)
        for dataset in test_datasets:
            for mth in mths:
                for i in range(rep):
                    random_id = np.random.randint(10000)
                    with Timer('%s-%s-%d-%d' % (mth, dataset, i, random_id)):
                        perfs = evaluate(mth, dataset, random_id)
                        print("len=", len(perfs), "unique=", len(set(perfs)))

                        timestamp = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
                        dir_path = 'logs/litebo_benchmark_lightgbm_%d/%s/' % (max_runs, mth)
                        file = 'benchmark_%s_%s_%s_%04d.pkl' % (mth, dataset, timestamp, random_id)
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        with open(os.path.join(dir_path, file), 'wb') as f:
                            pk.dump(perfs, f)
                        print(dir_path, file, 'saved!')

else:
    import matplotlib.pyplot as plt
    for dataset in test_datasets:
        plot_list = []
        legend_list = []
        for mth in mths:
            result = []
            dir_path = 'logs/litebo_benchmark_lightgbm_%d/%s/' % (max_runs, mth)
            for file in os.listdir(dir_path):
                if file.startswith('benchmark_%s_%s_' % (mth, dataset)) and file.endswith('.pkl'):
                    with open(os.path.join(dir_path, file), 'rb') as f:
                        perfs = pk.load(f)
                    if len(perfs) != max_runs:
                        print('Error len: ', file, len(perfs), type(perfs))
                        continue
                    result.append(descending(perfs))    # descent curve
            print('result rep=', len(result))
            mean_res = np.mean(result, axis=0)
            std_res = np.std(result, axis=0)

            # todo plot std figsize
            x = np.arange(len(mean_res)) + 1
            # p, = plt.plot(mean_res)
            p = plt.errorbar(x, mean_res, yerr=std_res*0.2, fmt='', capthick=0.5, capsize=3, errorevery=max_runs//10)
            plot_list.append(p)
            legend_list.append(mth)
        plt.legend(plot_list, legend_list, loc='upper right')
        plt.title(dataset)
        plt.xlabel("Iteration")
        plt.ylabel("Negative Balanced Accuracy Score")
        plt.show()
