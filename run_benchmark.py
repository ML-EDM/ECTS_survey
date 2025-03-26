import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ['KMP_WARNINGS'] = 'off'

import json
import copy
import argparse 
import warnings

import numpy as np
import pickle as pkl
from sktime.classification.kernel_based import RocketClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifierCV
from aeon.classification.dictionary_based import WEASEL_V2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.extmath import softmax
from inspect import isclass, getmembers
from joblib import Parallel, delayed

from ml_edm.early_classifier import EarlyClassifier
from ml_edm.classification.classifiers_collection import ClassifiersCollection
from ml_edm.deep.deep_classifiers import *
from ml_edm.cost_matrices import CostMatrices
from ml_edm.trigger import *

warnings.filterwarnings("ignore")

PARAMSPATH = "config_params.json"

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save", default=False, action=argparse.BooleanOptionalAction,
                    help="Whether or not to save classifiers and trigger models")
parser.add_argument("-pl", "--preload_clf", default=False, action=argparse.BooleanOptionalAction,
                    help="Whether or not to preload the trained classifiers")
parser.add_argument("-base", "--baseline", default=True, action=argparse.BooleanOptionalAction,
                    help="Whether or not fit baselines trigger models")
args = parser.parse_args()

# Ridge classifier with a predict_proba 
# method to test calibration effect
class RidgeClassifierCVP(RidgeClassifierCV):
    
    def predict_proba(self, X):
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            scores = np.c_[-scores, scores]
        return softmax(scores)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def load_dataset(dataset_name, split, z_normalize=False):
    
    data_train = np.loadtxt(f'{dataset_name}/{dataset_name}_TRAIN.txt')
    data_test = np.loadtxt(f'{dataset_name}/{dataset_name}_TEST.txt')
    
    X_train, X_test = (data_train[:, 1:], data_test[:, 1:])
    y_train, y_test = (data_train[:, 0], data_test[:, 0])

    if split != 'default':
        X_ = np.concatenate((X_train, X_test), axis=0)
        y_ = np.concatenate((y_train, y_test), axis=0)
        
        X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=split, random_state=44, stratify=y_)
    
    if z_normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    lb = LabelEncoder()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    data_dict  = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }
    os.chdir(os.path.expanduser('~'))
    return data_dict

def extract_and_save_features(X, chrono_clf, path):

    if not os.path.isdir(path):
            os.makedirs(path)

    for i, t in enumerate(chrono_clf.timestamps):
        Xt = X[:, :t]
        Xt = chrono_clf.extractors[i].transform(Xt)

        save_path = os.path.join(path, f"features_{i}.npy")
        np.save(save_path, Xt)

    return path

def _fit_chrono_clf(X, y, name, base_classifier, feature_extraction, params, alpha=None):

    chrono_clf = None
    X_clf, X_trigger, y_clf, y_trigger = train_test_split(
        X, y, test_size=params['val_ratio_trigger'], 
        random_state=params['random_state'],
        stratify=y
    )
    if params['LOADPATH']: # load already fitted classifiers
        if feature_extraction:
            path = os.path.join(params['LOADPATH'], name, f"{type(base_classifier).__name__}", 
                                feature_extraction['method'], 'classifiers.pkl')
        else:
            path = os.path.join(params['LOADPATH'], name, f"{type(base_classifier).__name__}", 'classifiers.pkl')
        with open(path, "rb") as clf_file:
            chrono_clf = pkl.load(clf_file)
    else:
        calib = bool(params['calib'])
        chrono_clf = ClassifiersCollection(
            base_classifier=base_classifier,
            sampling_ratio=params['sampling_ratio'],
            min_length=params['min_length'],
            feature_extraction=feature_extraction, 
            calibration=calib
        )
        chrono_clf.fit(X_clf, y_clf)

    if params['SAVEPATH_clf'] and not params['LOADPATH']:
        clf_name = type(base_classifier).__name__ if isinstance(chrono_clf, ClassifiersCollection) \
            else chrono_clf.model._get_name()
        if feature_extraction:
            path = os.path.join(params['SAVEPATH_clf'], name, f"{clf_name}", feature_extraction['method'])
        else:
            path = os.path.join(params['SAVEPATH_clf'], name, f"{clf_name}")
        if not os.path.isdir(path):
            os.makedirs(path)
                     
        path = os.path.join(path, "classifiers.pkl")
        with open(path, "wb") as out_file:
            pkl.dump(chrono_clf, out_file)

    return chrono_clf, X_trigger, y_trigger

def _fit_early_classifier(X, y, name, chrono_clf, trigger_model, trigger_class, alpha, n_classes, params, features):

    def delay_cost(t):
        return np.exp((t/X.shape[1]) * np.log(100))
    
    #small_values = (n_classes / (n_classes+9))
    small_values = 1
    misclf_cost = small_values - np.eye(n_classes) * small_values

    classes, counts = np.unique(y, return_counts=True)
    idx_min_class = classes[counts.argmin()]
    misclf_cost[:, idx_min_class] *= 100

    cost_matrices = CostMatrices(chrono_clf.timestamps, n_classes, alpha=alpha, 
                                 delay_cost=delay_cost, misclf_cost=misclf_cost)
    
    trigger = copy.deepcopy(trigger_class)
    trigger.timestamps = chrono_clf.timestamps
    if trigger_model == "ecdire":
        trigger.chronological_classifiers = chrono_clf

    e_clf = EarlyClassifier(
        chrono_clf, trigger, cost_matrices, prefit_classifiers=True
    )
    e_clf.fit(X, y) 
    
    if params['SAVEPATH_early_clf']:
        trigg_name = type(chrono_clf.base_classifier).__name__ if isinstance(chrono_clf, ClassifiersCollection) \
            else chrono_clf.model._get_name()
        if features:
            pp = os.path.join(params['SAVEPATH_early_clf'], name, f"{trigg_name}", features['method'], f"alpha_{str(alpha)}")
        else:
            pp = os.path.join(params['SAVEPATH_early_clf'], name, f"{trigg_name}", f"alpha_{str(alpha)}")

        if not os.path.isdir(pp):
            os.mkdir(pp)

        path = os.path.join(pp, f"early_classifier_{trigger_model}.pkl")
        with open(path, "wb") as out_file:
            pkl.dump(e_clf, out_file)
    
    return e_clf

def get_output_metrics(early_clf, X, y, compute_post=False, dict_post_cache=None):
    
    metrics = early_clf.score(X, y, return_metrics=True)

    if compute_post:
        dict_post = early_clf.get_post(X, y, use_probas=False, return_metrics=True)
        metrics_post = {k+"_pred": v for k, v in dict_post.items()}
    else:
        metrics_post = dict_post_cache

    metrics_additional = {"mean_t_star": np.mean(metrics['t_star']),
                          "mean_t_post_pred": np.mean(metrics_post['t_post_pred']),
                          "median_t_star": np.median(metrics['t_star']),
                          "median_t_post_pred": np.median(metrics_post['t_post_pred']),
                          "std_t_star": np.std(metrics['t_star']),
                          "std_t_post_pred": np.std(metrics_post['t_post_pred']),
                          "std_f_star": np.std(metrics['f_star']),
                          "std_f_post_pred": np.std(metrics_post["f_post_pred"])}
        
    metrics_additional.update({"mean_diff_t_post_pred_star": np.mean(metrics_post['t_post_pred'] - metrics['t_star']),
                               "std_diff_t_post_pred_star": np.std(metrics_post['t_post_pred'] - metrics['t_star']),
                               "mean_diff_f_star_post_pred": np.mean(metrics['f_star'] - metrics_post['f_post_pred']),
                               "std_diff_f_star_post_pred": np.std(metrics['f_star'] - metrics_post['f_post_pred'])})

    if early_clf.trigger_model.require_classifiers: # if use probabilistic classifiers collection

        if compute_post:
            dict_post_proba = early_clf.get_post(X, y, use_probas=True, return_metrics=True)
            metrics_post.update({k+"_proba": v for k, v in dict_post_proba.items()})

        metrics_additional.update({"mean_t_post_proba": np.mean(metrics_post['t_post_proba']),
                                   "median_t_post_proba": np.median(metrics_post['t_post_proba']),
                                   "std_t_post_proba": np.std(metrics_post['t_post_proba']),
                                   "std_f_post_proba": np.std(metrics_post["f_post_proba"])})
            
        metrics_additional.update({"mean_diff_t_post_proba_star": np.mean(metrics_post['t_post_proba'] - metrics['t_star']),
                                   "std_diff_t_post_proba_star": np.std(metrics_post['t_post_proba'] - metrics['t_star']),
                                   "mean_diff_f_star_post_proba": np.mean(metrics['f_star'] - metrics_post['f_post_proba']),
                                   "std_diff_f_star_post_proba": np.std(metrics['f_star'] - metrics_post['f_post_proba'])})
        
    return metrics, metrics_post, metrics_additional 

def train_for_one_alpha(alpha, params, trigger_models, prefit_cost_unaware=False):
    
    alpha0 = params['alphas'][0]
    dict_trigger = dict.fromkeys(trigger_models.keys())
    dict_clf = dict.fromkeys(params['classifiers'])
    dict_data = dict.fromkeys(params['datasets'])
    metrics_alpha = {k1 : {k2: copy.deepcopy(dict_data) for k2, _ in copy.deepcopy(dict_clf).items()} 
                     for k1, _ in copy.deepcopy(dict_trigger).items()}

    os.chdir(os.path.expanduser('~'))
    pbar_data = params['datasets']
    for dataset in pbar_data:
        print(f"Processing {dataset} ...")
        os.chdir(params['DATAPATH'])
        data = load_dataset(dataset, params['split'], params['z_normalized'])
        n_classes = len(np.unique(data['y_train']))

        for idx, clf in enumerate(params['classifiers'].keys()):

            try:
                base_clf = eval(clf.split("_")[0])(**params['classifiers'][clf])
            except NameError:
                base_clf = clf

            try:
                features_extractor = params['extractors'][clf].copy()
                if isinstance(features_extractor['method'], list):
                    features_extractor['method'] = features_extractor['method'][0]
                    params['extractors'][clf]['method'].pop(0)
                print(f"Using {clf} with {features_extractor['method']} ...")
            except KeyError:
                features_extractor = None
                print(f"Using {clf} ...")
            
            chrono_clf, X_trigger, y_trigger = _fit_chrono_clf(data['X_train'], data['y_train'], dataset, 
                                                               base_clf, features_extractor, params, alpha)
            if features_extractor:
                try:
                    path = os.path.join(features_extractor['path'], dataset, 
                                        type(chrono_clf.base_classifier).__name__, features_extractor['method'])
                    if not prefit_cost_unaware:
                        extract_and_save_features(X_trigger, chrono_clf, path+'/train')
                        extract_and_save_features(data['X_test'], chrono_clf, path+'/test')
                    features_paths = {"train": path+'/train', "test": path+'/test'}
                except KeyError:
                    features_paths = None

            past_post = None
            for i, (trigger, t) in enumerate(trigger_models.items()):
                # if trigger model invariant to classifiers used
                # only train it once, i.e. the first one 
                if idx != 0 and trigger in ["edsc", "ects"]:
                    continue

                print(f"Testing {trigger} ....")
                    
                if prefit_cost_unaware and (trigger in ["ecdire", "edsc", "ects"]) and params['SAVEPATH_early_clf']:
                    if trigger == 'ecdire':
                        if features_extractor:
                            early_path = os.path.join(params['SAVEPATH_early_clf'], dataset, f"{type(base_clf).__name__}", 
                                                    features_extractor['method'], f"alpha_{alpha0}")
                        else:
                            early_path = os.path.join(params['SAVEPATH_early_clf'], dataset, 
                                                    f"{type(base_clf).__name__}", f"alpha_{alpha0}")
                    else:
                        early_path = os.path.join(params['SAVEPATH_early_clf'], dataset, "RidgeClassifierCV", 
                                                  "minirocket", f"alpha_{alpha0}")

                    with open(early_path + f"/early_classifier_{trigger}.pkl", "rb") as load_file:
                        early_clf = pkl.load(load_file)
                    
                    def delay_cost(t):
                        inflexion_point = 0
                        return np.exp(((t/data['X_train'].shape[1])-inflexion_point) * np.log(100))
                    
                    small_values = 1
                    misclf_cost = small_values - np.eye(n_classes) * small_values
                    classes, counts = np.unique(data['y_train'], return_counts=True)
                    idx_min_class = classes[counts.argmin()]
                    misclf_cost[:, idx_min_class] *= 100
                    
                    early_clf.cost_matrices = CostMatrices(chrono_clf.models_input_lengths, n_classes, 
                                                           alpha=alpha, delay_cost=delay_cost, missclf_cost=misclf_cost)
                else:
                    if features_extractor:
                        if features_paths:
                            chrono_clf.feature_extraction = features_paths['train']
                    early_clf = _fit_early_classifier(X_trigger, y_trigger, dataset, chrono_clf, 
                                                      trigger, t, alpha, n_classes, params, features_extractor)

                get_post = True if i == 0 else False
                if features_extractor:
                    if features_paths:
                        early_clf.chronological_classifiers.feature_extraction = features_paths['test']
                        if trigger == "ecdire":
                            early_clf.new_chronological_classifiers.feature_extraction = features_paths['test']

                metrics, post, add = get_output_metrics(early_clf, data['X_test'], data['y_test'], 
                                                        compute_post=get_post, dict_post_cache=past_post)
                past_post = post
                metrics_alpha[trigger][clf][dataset] = {**metrics, **post, **add}
    
    tmp_dir = os.path.join(params['RESULTSPATH'], f"alpha_{str(alpha)}")
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    with open(os.path.join(tmp_dir, f"res.json"), "w") as tmp_file:
        json.dump(metrics_alpha, tmp_file, cls=NpEncoder)

    return {alpha: metrics_alpha}

def compute_baselines(alphas, params, trigger_models):

    dict_trigger = dict.fromkeys(trigger_models.keys())
    dict_clf = dict.fromkeys(params['classifiers'])
    dict_data = dict.fromkeys(params['datasets'])
    metrics = {k1 : {k2: copy.deepcopy(dict_data) for k2, _ in copy.deepcopy(dict_clf).items()}
               for k1, _ in copy.deepcopy(dict_trigger).items()}
    metrics_alpha = {alpha : copy.deepcopy(metrics) for alpha in alphas}

    os.chdir(os.path.expanduser('~'))
    for dataset in params["datasets"]:
        print(f"Processing {dataset} ...")
        os.chdir(params['DATAPATH'])
        data = load_dataset(dataset, params['split'])
        n_classes = len(np.unique(data['y_train']))

        for clf in params['classifiers'].keys():
            try:
                base_clf = eval(clf.split("_")[0])(**params['classifiers'][clf])
            except NameError:
                base_clf = clf
            try:
                features_extractor = params['extractors'][clf].copy()
                if isinstance(features_extractor['method'], list):
                    features_extractor['method'] = features_extractor['method'][0]
                    params['extractors'][clf]['method'].pop(0)
                print(f"Using {clf} with {features_extractor['method']} ...")
            except KeyError:
                features_extractor = None
                print(f"Using {clf} ...")
            
            chrono_clf, X_trigger, y_trigger = _fit_chrono_clf(data['X_train'], data['y_train'], dataset, 
                                                               base_clf, features_extractor, params)
            
            for alpha in alphas:
                for trigger, t in trigger_models.items():
                    print(f"Testing {trigger} with alpha : {alpha} ....")
                    
                    early_clf = _fit_early_classifier(X_trigger, y_trigger, dataset, chrono_clf, 
                                            trigger, t, alpha, n_classes, params, features_extractor) 
                    m = early_clf.score(data['X_test'], data['y_test'], return_metrics=True)       
                    metrics_alpha[alpha][trigger][clf][dataset] = copy.deepcopy(m)
    
    for alpha in  alphas:
        path = os.path.join(params['RESULTSPATH'], f"alpha_{str(alpha)}")
        if not os.path.isdir(path):
            os.mkdir(path)
            
        with open(os.path.join(path, f"baselines.json"), "w") as tmp_file:
            json.dump(metrics_alpha[alpha], tmp_file, cls=NpEncoder)
    
    return metrics_alpha 

if __name__ == '__main__':

    with open(PARAMSPATH) as param_file:
        params = json.load(param_file)
    
    TRIGGERS = {
    "calimera": CALIMERA(np.arange(10)), 
    "ecdire": ECDIRE(ClassifiersCollection()),
    "ecec": ECEC(np.arange(10)),  
    "economy": EconomyGamma(np.arange(10), random_state=params['random_state']),
    "proba_threshold": ProbabilityThreshold(np.arange(10)),
    "stopping_rule": StoppingRule(np.arange(10)),
    "teaser_hm": TEASER(np.arange(10)), 
    "teaser_avg_cost": TEASER(np.arange(10), objective="avg_cost")
    }
    
    if args.preload_clf:
        params['LOADPATH'] = params['SAVEPATH_clf']
    else:
        params['LOADPATH'] = None 

    if not args.save:
        params['SAVEPATH_clf'] = None
        params['SAVEPATH_early_clf'] = None
    
    results = []
    # first run with parallelisation over trigger models 
    # to learn and save cost-unaware models 
    for name, p in TRIGGERS.items():
        p.set_params(n_jobs=params['n_jobs'])
    results.append(train_for_one_alpha(params['alphas'][0], params, TRIGGERS, False))

    if args.save:
        params['LOADPATH'] = params['SAVEPATH_clf']
    
    for name, p in TRIGGERS.items():
        p.set_params(n_jobs=params['n_jobs'])
    res = Parallel(n_jobs=params['n_jobs'], backend='multiprocessing') \
        (delayed(train_for_one_alpha)(a, params, TRIGGERS, True) for a in params['alphas'][1:])
    results.extend(res)
    
    if args.baseline:
        TRIGGERS = {"asap": ProbabilityThreshold(np.arange(10), manual_threshold=0),   
                    "alap": ProbabilityThreshold(np.arange(10), manual_threshold=1.1)}
        compute_baselines(params['alphas'], params, TRIGGERS)