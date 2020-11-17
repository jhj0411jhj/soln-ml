"""
example cmdline:

python test/bayesian_opt/litebo_random_forest_benchmark.py --mths lite_bo,smac,tpe --datasets wind --n 10 --rep 2 --n_jobs 4

python test/bayesian_opt/litebo_random_forest_benchmark.py --plot_mode 1 --mths lite_bo,smac,tpe --datasets wind --n 10

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

sys.path.append(".")
sys.path.insert(0, "../lite-bo")
from solnml.datasets.utils import load_train_test_data
from solnml.components.metrics.metric import get_metric
from solnml.components.utils.constants import MULTICLASS_CLS
from solnml.components.utils.configspace_utils import check_none, check_for_bool
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
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
test_datasets = args.datasets.split(',')
print("datasets num=", len(test_datasets))
mths = args.mths.split(',')
plot_mode = args.plot_mode
max_runs = args.n
start_id = args.start_id

seeds = [4465, 3822, 4531, 8459, 6295, 2854, 7820, 4050, 280, 6983,
         5497, 83, 9801, 8760, 5765, 6142, 4158, 9599, 1776, 1656]


class RandomForest:
    def __init__(self, criterion, max_features,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, bootstrap, max_leaf_nodes,
                 min_impurity_decrease, random_state=None, n_jobs=1,
                 class_weight=None):
        self.n_estimators = self.get_max_iter()
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.bootstrap = bootstrap
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.estimator = None

    @staticmethod
    def get_max_iter():
        return 100

    def get_current_iter(self):
        return self.estimator.n_estimators

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import RandomForestClassifier

        if self.estimator is None:
            self.n_estimators = int(self.n_estimators)
            if check_none(self.max_depth):
                self.max_depth = None
            else:
                self.max_depth = int(self.max_depth)

            self.min_samples_split = int(self.min_samples_split)
            self.min_samples_leaf = int(self.min_samples_leaf)
            self.min_weight_fraction_leaf = float(self.min_weight_fraction_leaf)

            if self.max_features not in ("sqrt", "log2", "auto"):
                max_features = int(X.shape[1] ** float(self.max_features))
            else:
                max_features = self.max_features

            self.bootstrap = check_for_bool(self.bootstrap)

            if check_none(self.max_leaf_nodes):
                self.max_leaf_nodes = None
            else:
                self.max_leaf_nodes = int(self.max_leaf_nodes)

            self.min_impurity_decrease = float(self.min_impurity_decrease)

            # initial fit of only increment trees
            self.estimator = RandomForestClassifier(
                n_estimators=self.get_max_iter(),
                criterion=self.criterion,
                max_features=max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                bootstrap=self.bootstrap,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight=self.class_weight,
                warm_start=True)

        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_hyperparameter_search_space(optimizer='smac'):
        if optimizer == 'smac':
            cs = ConfigurationSpace()
            criterion = CategoricalHyperparameter(
                "criterion", ["gini", "entropy"], default_value="gini")

            # The maximum number of features used in the forest is calculated as m^max_features, where
            # m is the total number of features, and max_features is the hyperparameter specified below.
            # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
            # corresponds with Geurts' heuristic.
            max_features = UniformFloatHyperparameter(
                "max_features", 0., 1., default_value=0.5)

            max_depth = UnParametrizedHyperparameter("max_depth", "None")
            min_samples_split = UniformIntegerHyperparameter(
                "min_samples_split", 2, 20, default_value=2)
            min_samples_leaf = UniformIntegerHyperparameter(
                "min_samples_leaf", 1, 20, default_value=1)
            min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
            max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
            min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
            bootstrap = CategoricalHyperparameter(
                "bootstrap", ["True", "False"], default_value="True")
            cs.add_hyperparameters([criterion, max_features,
                                    max_depth, min_samples_split, min_samples_leaf,
                                    min_weight_fraction_leaf, max_leaf_nodes,
                                    bootstrap, min_impurity_decrease])
            return cs
        elif optimizer == 'tpe':
            from hyperopt import hp
            space = {'criterion': hp.choice('rf_criterion', ["gini", "entropy"]),
                     'max_features': hp.uniform('rf_max_features', 0, 1),
                     'max_depth': hp.choice('rf_max_depth', [None]),
                     'min_samples_split': hp.randint('rf_min_samples_split', 19) + 2,
                     'min_samples_leaf': hp.randint('rf_min_samples_leaf', 20) + 1,
                     'min_weight_fraction_leaf': hp.choice('rf_min_weight_fraction_leaf', [0]),
                     'max_leaf_nodes': hp.choice('rf_max_leaf_nodes', [None]),
                     'min_impurity_decrease': hp.choice('rf_min_impurity_decrease', [0]),
                     'bootstrap': hp.choice('rf_bootstrap', ["True", "False"]),
                     }
            return space
        else:
            raise ValueError('Unknown optimizer %s when getting cs' % optimizer)


def get_configspace(optimizer='smac'):
    return RandomForest.get_hyperparameter_search_space(optimizer=optimizer)


def get_estimator(config):
    config['random_state'] = 1
    estimator = RandomForest(**config)
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', n_jobs)
    return estimator


def evaluate(mth, dataset, seed):
    print(mth, dataset, seed)
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
                  random_state=seed)
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
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(seed),
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
        fmin(tpe_objective_function, config_space, tpe.suggest, max_runs, trials=trials,
             rstate=np.random.RandomState(seed))
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


if plot_mode == 0:
    n_jobs = args.n_jobs
    rep = args.rep

    with Timer('All'):
        check_datasets(test_datasets)
        for dataset in test_datasets:
            for mth in mths:
                for i in range(start_id, start_id + rep):
                    seed = seeds[i]
                    with Timer('%s-%s-%d-%d' % (mth, dataset, i, seed)):
                        perfs = evaluate(mth, dataset, seed)
                        print("len=", len(perfs), "unique=", len(set(perfs)))

                        timestamp = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
                        dir_path = 'logs/litebo_benchmark_random_forest_%d/%s/' % (max_runs, mth)
                        file = 'benchmark_%s_%s_%s_%04d.pkl' % (mth, dataset, timestamp, seed)
                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        with open(os.path.join(dir_path, file), 'wb') as f:
                            pk.dump(perfs, f)
                        print(dir_path, file, 'saved!')

elif plot_mode == 1:
    import tabulate
    headers = ['dataset']
    headers.extend(mths)
    tbl_data = []

    for dataset in test_datasets:
        row_data = [dataset]
        mean_list = []
        for mth in mths:
            incumbent = []
            dir_path = 'logs/litebo_benchmark_random_forest_%d/%s/' % (max_runs, mth)
            for file in os.listdir(dir_path):
                if file.startswith('benchmark_%s_%s_' % (mth, dataset)) and file.endswith('.pkl'):
                    with open(os.path.join(dir_path, file), 'rb') as f:
                        perfs = pk.load(f)
                    if len(perfs) != max_runs:
                        print('Error len: ', file, len(perfs), type(perfs))
                        continue
                    incumbent.append(-min(perfs)*100)   # positive score
            print('result rep=', len(incumbent))
            mean_res = np.mean(incumbent)
            std_res = np.std(incumbent)

            mean_list.append(mean_res)
            data_point = u'%.3f\u00B1%.3f' % (mean_res.item(), std_res.item())
            row_data.append(data_point)
        row_data[np.argsort(mean_list)[-1] + 1] += '**'  # mark best mth for each dataset
        row_data[np.argsort(mean_list)[-2] + 1] += '*'   # mark second best mth for each dataset
        tbl_data.append(row_data)
    print(tabulate.tabulate(tbl_data, headers, tablefmt='github'))

else:
    import matplotlib.pyplot as plt
    for dataset in test_datasets:
        plot_list = []
        legend_list = []
        for mth in mths:
            result = []
            dir_path = 'logs/litebo_benchmark_random_forest_%d/%s/' % (max_runs, mth)
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

            # plot
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


