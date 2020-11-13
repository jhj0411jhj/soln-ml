"""
example cmdline:

python test/bayesian_opt/litebo_parallel_lightgbm_benchmark.py \
--mths sync --datasets wind --n 10 --rep 1 --n_jobs 1 --batch_size 8

"""

import os
import sys
import time
import argparse
import numpy as np
import pickle as pk
from multiprocessing import Process, Manager

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
from litebo.optimizer.message_queue_smbo import mqSMBO
from litebo.core.message_queue.worker import Worker


default_datasets = 'optdigits,satimage,wind,delta_ailerons,puma8NH,kin8nm,cpu_small,puma32H,cpu_act,bank32nh'
default_mths = 'sync,async'

parser = argparse.ArgumentParser()
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--datasets', type=str, default=default_datasets)
parser.add_argument('--n_jobs', type=int, default=2)
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=0)

args = parser.parse_args()
test_datasets = args.datasets.split(',')
print("datasets num=", len(test_datasets))
mths = args.mths.split(',')
max_runs = args.n
n_jobs = args.n_jobs
rep = args.rep
start_id = args.start_id

batch_size = args.batch_size
ip = args.ip
port = args.port

seeds = [4465, 3822, 4531, 8459, 6295, 2854, 7820, 4050, 280, 6983,
         5497, 83, 9801, 8760, 5765, 6142, 4158, 9599, 1776, 1656]


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


def evaluate_parallel(mth, batch_size, dataset, seed, ip, port):
    assert mth in ['sync', 'async']
    print(mth, batch_size, dataset, seed)
    if port == 0:
        port = 13579 + np.random.randint(1000)
    print('ip=', ip, 'port=', port)

    train_data, test_data = load_train_test_data(dataset, test_size=0.3, task_type=MULTICLASS_CLS)

    def objective_function(config):
        metric = get_metric('bal_acc')
        estimator = get_estimator(config.get_dictionary())
        X_train, y_train = train_data.data
        X_test, y_test = test_data.data
        estimator.fit(X_train, y_train)
        return -metric(estimator, X_test, y_test)

    config_space = get_configspace()

    def master_run(return_list):
        bo = mqSMBO(None, config_space, max_runs=max_runs, time_limit_per_trial=600, parallel_strategy=mth,
                    batch_size=batch_size, ip='', port=port, random_state=seed)
        bo.run()
        return_list.extend(bo.benchmark_perfs[:max_runs])   # send to return list. may exceed max_runs in sync

    def worker_run(i):
        worker = Worker(objective_function, ip, port)
        worker.run()
        print("Worker %d exit." % (i))

    manager = Manager()
    perfs = manager.list()  # shared list
    master = Process(target=master_run, args=(perfs,))
    master.start()

    time.sleep(15)  # wait for master init
    worker_pool = []
    for i in range(batch_size):
        worker = Process(target=worker_run, args=(i,))
        worker_pool.append(worker)
        worker.start()

    master.join()   # wait for master to gen result
    for w in worker_pool:   # optional if repeat=1
        w.join()

    return list(perfs)  # covert to list


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
        if exc_type == SystemExit:  # worker exit, don't output
            print("this is", self.name)
            return
        self.end = time.time()
        m, s = divmod(self.end - self.start, 60)
        h, m = divmod(m, 60)
        print("[%s]Total time = %d hours, %d minutes, %d seconds." % (self.name, h, m, s))


with Timer('All'):
    check_datasets(test_datasets)
    for dataset in test_datasets:
        for mth in mths:
            for i in range(start_id, start_id + rep):
                seed = seeds[i]
                with Timer('%s-%d-%s-%d-%d' % (mth, batch_size, dataset, i, seed)):
                    perfs = evaluate_parallel(mth, batch_size, dataset, seed, ip, port)
                    print("len=", len(perfs), "unique=", len(set(perfs)))

                    mth_str = mth + '-' + str(batch_size)
                    timestamp = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
                    dir_path = 'logs/litebo_benchmark_lightgbm_%d/%s/' % (max_runs, mth_str)
                    file = 'benchmark_%s_%s_%s_%04d.pkl' % (mth_str, dataset, timestamp, seed)
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    with open(os.path.join(dir_path, file), 'wb') as f:
                        pk.dump(perfs, f)
                    print(dir_path, file, 'saved!')
