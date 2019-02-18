from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pmlb import fetch_data

def label_binarize_fix(y, classes):
    y_binarized = label_binarize(y, classes)
    if len(classes) > 2:
        return y_binarized
    else:
        y_binarized_fix = np.zeros((len(y), 2)).astype(int)
        y_binarized_fix[np.arange(len(y)), y_binarized.ravel()] = 1
        return y_binarized_fix
    

class MajorClassClassifier(BaseEstimator):
    
    def __init__(self, binarized_labels = False):
        self.binarized_labels = binarized_labels
        pass
    
    def fit(self, X, y):
        if not self.binarized_labels:
            self.classes_ =  np.array(list(set(y)))
            self.major_class_ = Counter(y).most_common(1)[0][0]
        else:
            self.classes_ = np.arange(y.shape[1])
            self.major_class_ = np.argmax(np.sum(y, axis = 1))
        if len(self.classes_) < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
            " class: %r" % self.classes_[0])

    def predict_proba(self, X):
        result = np.zeros((len(X), len(self.classes_)))
        i = np.where(self.classes_ == self.major_class_)[0][0]
        result[:, i] = -0.001*np.random.rand(len(result)) + 1.0
        for k in range(result.shape[1]):
            if k != i:
                result[:, k] = (1 - result[:, i])/(result.shape[1] - 1)  
        return result

    def predict(self, X):
        preds = self.classes_[np.argmax(self.predict_proba(X), axis = 1)]
        if self.binarized_labels:
            preds = label_binarize_fix(preds, self.classes_)
        return preds

class SSMF(BaseEstimator):
    r"""Wraps input features with StandardScaler which is fit on the training sample.
      """
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
    
    def fit(self, X_l, y_l, X_h, y_h):
        self.scaler = StandardScaler()
        self.scaler.fit(np.vstack((X_l, X_h)))
        self.base_classifier.fit(self.scaler.transform(X_l), y_l, self.scaler.transform(X_h), y_h)
        
    def predict(self, X):
        return self.base_classifier.predict(self.scaler.transform(X))
    
    def predict_proba(self, X):
        return self.base_classifier.predict_proba(self.scaler.transform(X))

def safe_roc_auc_score(y_true, y_score, **kwargs):
    try:
        return roc_auc_score(y_true, y_score, **kwargs)
    except ValueError:
        return 0.5
        
def test_cv_benchmark(X, y, estimators, scoring, verbose=True, cv=10):
    scoring_keys = sorted(scoring.keys())
    scoring_keys
    df = pd.DataFrame()
    for clf in estimators:
        if verbose:
            print(clf.__class__.__name__)
        np.random.seed(0)
        gs_clf = GridSearchCV(clf, cv=cv, scoring = scoring, error_score=0.5, refit='ROCAUC', param_grid = {}, verbose=0)
        gs_clf.fit(X, y)
        scores = []
        for s in scoring_keys:
            scores.append(gs_clf.cv_results_['mean_test_' + s][0])
            #print('%.4f' % (gs_clf.cv_results_['mean_test_' + s][0]), s)
        df = df.append([[gs_clf.best_estimator_.__class__.__name__] + scores], ignore_index=True)
    df.columns = ['classifier'] + scoring_keys
    return df

def train_test_split_with_indices(X, y, test_size, random_state):
    X_with_index = np.hstack((X, np.arange(len(X)).reshape(-1, 1)))
    X_train, X_test, y_train, y_test = train_test_split(X_with_index, y, test_size=test_size, random_state=random_state)
    return X_train[:, :-1], X_test[:, :-1], y_train, y_test, X_train[:, -1].astype(int), X_test[:, -1].astype(int)


def test_clf_wrt_train_size_single_alpha(clf, scoring, X, y, y_corrupted = None, y_gold = None,
                                         runs = 50, alpha = 0.1, 
                                         low_fidelity_factor = 1.0, vanilla_mf = False, stacked_mf = False, 
                                         r_offset = [0], verbose = False):
    scoring_keys = sorted(scoring.keys())
    scoring_keys
    alpha_result = []
    num_classes = len(set(y))
    for r in tqdm(range(runs), disable = not verbose):
        # ensure that y_train has at least one representative from each class
        is_y_train_good = False
        while not is_y_train_good:
            X_train, X_test, y_train, y_test, i_train, i_test = train_test_split_with_indices(X, y, test_size=1 - alpha, random_state=r + r_offset[0])
            is_y_train_good = (len(set(y_train)) == num_classes)
            if y_gold is not None:
                y_test = y_gold[i_test]
            if y_corrupted is not None:
                X_train_lf, _, y_train_lf, _ = train_test_split(X, y_corrupted, test_size=1 - alpha*low_fidelity_factor, random_state=r + r_offset[0] + 1)
                is_y_train_good &= (len(set(y_train_lf)) == num_classes)
            r_offset[0] += 1
        try:
            if stacked_mf:
                clf.fit(X_train_lf, y_train_lf)
                X_train = np.hstack((X_train, clf.predict_proba(X_train)))
                X_test = np.hstack((X_test, clf.predict_proba(X_test)))
            if vanilla_mf:
                X_train = np.vstack((X_train, X_train_lf))
                y_train = np.hstack((y_train, y_train_lf))
            if y_corrupted is None or vanilla_mf or stacked_mf:
                clf.fit(X_train, y_train)
            else:
                clf.fit(X_train_lf, y_train_lf, X_train, y_train)
        except Exception:   
            print(clf.__class__.__name__, r, alpha, y_train)
            raise
        scores = []
        for s in scoring_keys:
            scores.append(scoring[s](clf, X_test, y_test))
        alpha_result.append(scores)
    return np.array(alpha_result), scoring_keys

def test_clf_wrt_train_size(clf, scoring, X, y, y_corrupted = None, 
                            points = 10, runs = 50, max_alpha = 1.0, 
                            low_fidelity_factor = 1.0, vanilla_mf = False):
    scoring_keys = sorted(scoring.keys())
    scoring_keys
    result = []
    r_offset = [0]
    for alpha in tqdm(np.linspace(0, max_alpha, points + 1)[1:-1]):
        alpha_result, _ = test_clf_wrt_train_size_single_alpha(clf, scoring, X, y, y_corrupted, 
                                                                runs=runs, alpha=alpha, low_fidelity_factor=low_fidelity_factor, 
                                                                vanilla_mf=vanilla_mf, r_offset=r_offset)
        result.append(alpha_result)
    return np.array(result), scoring_keys

def plot_single_estimator_scores(X, y, clf, scoring, runs = 50, max_alpha = 1.0):
    res, sk = test_clf_wrt_train_size(X, y, clf, scoring, runs = runs, max_alpha=max_alpha)
    xs = np.linspace(0, max_alpha, len(res) + 2)[1:-1]
    plt.figure(figsize = (15, 4))
    plt.suptitle(clf.__class__.__name__)
    for i in range(len(scoring)):
        plt.subplot(1, len(scoring), i + 1)
        plt.fill_between(xs, np.min(res[:, :, i], axis = 1), np.max(res[:, :, i], axis = 1), alpha=0.1, color = 'C0')
        plt.fill_between(xs, np.percentile(res[:, :, i], 5, axis = 1), np.percentile(res[:, :, i], 95, axis = 1), alpha=0.1, color = 'C0')
        plt.plot(xs, np.median(res[:, :, i], axis = 1), color = 'C0')
        plt.title(sk[i])
        plt.xlabel('train sample share')
        plt.ylabel('test performance')
        
def plot_scores_comparison(X, y, estimators, scoring, dataset_name = '', runs = 50, max_alpha = 1.0):
    plt.figure(figsize = (15, 4))
    for clf_i, clf in enumerate(estimators):
        res, sk = test_clf_wrt_train_size(X, y, clf, scoring, runs = runs, max_alpha=max_alpha)
        xs = np.linspace(0, max_alpha, len(res) + 2)[1:-1]
        plt.suptitle(dataset_name)
        
        for i in range(len(scoring)):
            plt.subplot(1, len(scoring), i + 1)
            plt.plot(xs, np.percentile(res[:, :, i], 5, axis=1), color='C' + str(clf_i), linestyle='--', lw=1)
            plt.plot(xs, np.percentile(res[:, :, i], 95, axis=1), color='C' + str(clf_i), linestyle='--', lw=1)
            plt.plot(xs, np.median(res[:, :, i], axis=1), color='C' + str(clf_i), label=clf.__class__.__name__)
            plt.title(sk[i])
            plt.xlabel('train sample share')
            plt.ylabel('test performance')
            if (i == len(scoring) - 1) and (clf_i == len(estimators) - 1):
                plt.legend(loc=2, bbox_to_anchor=(1, 1))

def run_tests(X, y, scoring, methods, y_corrupted = None, y_gold = None, p = 0.1, hf_points = 20, runs = 25, verbose = True, use_x5 = True):
    ma = hf_points/len(y)
    if y_corrupted is None:
        y_corrupted = (y + (np.random.rand(len(y)) < p).astype(int)) % 2
    
    res_major_vote, sk = test_clf_wrt_train_size_single_alpha(methods['major_vote'], scoring, X, y, y_gold=y_gold, runs = runs, alpha=ma, low_fidelity_factor=3.0, verbose = verbose)
    if 'ss_mf_gpc' in methods:
        res_x1, sk = test_clf_wrt_train_size_single_alpha(methods['ss_mf_gpc'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, low_fidelity_factor=1.0, verbose = verbose)
        res_x3, sk = test_clf_wrt_train_size_single_alpha(methods['ss_mf_gpc'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, low_fidelity_factor=3.0, verbose = verbose)
        if use_x5:
            res_x5, sk = test_clf_wrt_train_size_single_alpha(methods['ss_mf_gpc'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, low_fidelity_factor=5.0, verbose = verbose)
        else:
            res_x5 = None
    else:
        res_x1 = None
        res_x3 = None
        res_x5 = None
    if 'ss_gpc' in methods:
        res_gpc, sk = test_clf_wrt_train_size_single_alpha(methods['ss_gpc'], scoring, X, y, y_gold=y_gold, runs = runs, alpha=ma, verbose = verbose)
        res_vanilla_gpc, sk = test_clf_wrt_train_size_single_alpha(methods['ss_gpc'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, low_fidelity_factor=3.0, verbose = verbose, vanilla_mf=True)
        res_stacked_gpc, sk = test_clf_wrt_train_size_single_alpha(methods['ss_gpc'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, low_fidelity_factor=3.0, verbose = verbose, stacked_mf=True)    
    else:
        res_gpc = None
        res_vanilla_gpc = None
        res_stacked_gpc = None
    res_logit, sk = test_clf_wrt_train_size_single_alpha(methods['ss_logit'], scoring, X, y, y_gold=y_gold, runs = runs, alpha=ma, verbose = verbose)
    res_vanilla_logit, sk = test_clf_wrt_train_size_single_alpha(methods['ss_logit'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, low_fidelity_factor=3.0, verbose = verbose, vanilla_mf=True)
    res_xgb, sk = test_clf_wrt_train_size_single_alpha(methods['xgb'], scoring, X, y, y_gold=y_gold, runs = runs, alpha=ma, verbose = verbose)
    res_vanilla_xgb, sk = test_clf_wrt_train_size_single_alpha(methods['xgb'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, low_fidelity_factor=3.0, verbose = verbose, vanilla_mf=True)
    res_stacked_logit, sk = test_clf_wrt_train_size_single_alpha(methods['ss_logit'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, low_fidelity_factor=3.0, verbose = verbose, stacked_mf=True)
    res_stacked_xgb, sk = test_clf_wrt_train_size_single_alpha(methods['xgb'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, low_fidelity_factor=3.0, verbose = verbose, stacked_mf=True)    

    cv_df_logit = test_cv_benchmark(X, y, [methods['ss_logit']], scoring, verbose = verbose)
    cv_df_xgb = test_cv_benchmark(X, y, [methods['xgb']], scoring, verbose = verbose)

    method_names = ['major_vote', 'mfgpc x1', 'mfgpc x3', 'mfgpc x5', 'gpc', 'logit', 'xgb', 
                    'vanilla_mf gpc x3', 'vanilla_mf logit x3', 'vanilla_mf xgb x3',
                    'stacked_mf gpc x3', 'stacked_mf logit x3', 'stacked_mf xgb x3',]
    results = [res_major_vote, res_x1, res_x3, res_x5, res_gpc, res_logit, res_xgb, 
               res_vanilla_gpc, res_vanilla_logit, res_vanilla_xgb,
               res_stacked_gpc, res_stacked_logit, res_stacked_xgb]
    
    df = pd.DataFrame()
    for method_name, res in zip(method_names, results):
        if res is not None:
            for i in range(len(sk)):
                for j in range(len(res)):
                    df = df.append([[method_name, sk[i], res[j, i]]], ignore_index=True)
    df.columns = ['method', 'metric', 'value']
    for metric in sk:
        baseline_df = pd.DataFrame([['logit cv10', metric, cv_df_logit[metric].values[0]], ['xgb cv10', metric, cv_df_xgb[metric].values[0]]])
        baseline_df.columns = ['method', 'metric', 'value']
        df = df.append(baseline_df, ignore_index=True)
    return df

def run_tests_budget(X, y, scoring, methods, y_corrupted = None, y_gold = None, p = 0.1, total_points = 300, runs = 10, verbose = True):
    ma = total_points/len(y)
    if y_corrupted is None:
        y_corrupted = (y + (np.random.rand(len(y)) < p).astype(int)) % 2
    
    res_x = []
    low_fidelity_factors = [1, 3, 5, 9]
    for lff in low_fidelity_factors: 
        res_x_i, sk = test_clf_wrt_train_size_single_alpha(methods['ss_mf_gpc'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma/(1.+lff), 
                                                            low_fidelity_factor=lff, verbose = verbose)
        res_x.append(res_x_i)
    
    res_gpc_hf, sk = test_clf_wrt_train_size_single_alpha(methods['ss_gpc'], scoring, X, y, y_gold=y_gold, runs = runs, alpha=ma, verbose = verbose)
    res_gpc_lf, sk = test_clf_wrt_train_size_single_alpha(methods['ss_gpc'], scoring, X, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, verbose = verbose)

    method_names = ['gpc hf'] + ['mfgpc bx' + str(lff) for lff in low_fidelity_factors] + ['gpc lf']
    results = [res_gpc_hf] + res_x + [res_gpc_lf]
    
    df = pd.DataFrame()
    for method_name, res in zip(method_names, results):
        if res is not None:
            for i in range(len(sk)):
                for j in range(len(res)):
                    df = df.append([[method_name, sk[i], res[j, i]]], ignore_index=True)
    df.columns = ['method', 'metric', 'value']
    return df

def run_tests_budget_cost(X, y, scoring, methods, y_corrupted = None, y_gold = None, p = 0.1, total_budget = 300, runs = 10, verbose = True, hf_to_lf_cost_ratio=4.):
    if y_corrupted is None:
        y_corrupted = (y + (np.random.rand(len(y)) < p).astype(int)) % 2
    
    res_x = []
    hf_shares = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for hf_share in hf_shares: 
        q = hf_share/(1. - hf_share)
        lf_points_cnt = total_budget/(q*hf_to_lf_cost_ratio + 1.)
        hf_points_cnt = (total_budget - lf_points_cnt)/hf_to_lf_cost_ratio
        ma = hf_points_cnt/len(y)
        lff = lf_points_cnt/hf_points_cnt
        res_x_i, sk = test_clf_wrt_train_size_single_alpha(methods['ss_mf_gpc'], scoring, X, y, y_corrupted, y_gold=y_gold, runs = runs, alpha=ma, 
                                                            low_fidelity_factor=lff, verbose = verbose)
        res_x.append(res_x_i)
    hf_points_cnt = total_budget/hf_to_lf_cost_ratio
    lf_points_cnt = total_budget
    res_gpc_hf, sk = test_clf_wrt_train_size_single_alpha(methods['ss_gpc'], scoring, X, y, y_gold=y_gold, runs = runs, alpha=hf_points_cnt/len(y), verbose = verbose)
    res_gpc_lf, sk = test_clf_wrt_train_size_single_alpha(methods['ss_gpc'], scoring, X, y_corrupted, y_gold=y_gold, runs = runs, alpha=lf_points_cnt/len(y), verbose = verbose)

    method_names = ['gpc hf'] + ['mfgpc hx' + str(hf_share) for hf_share in hf_shares] + ['gpc lf']
    results = [res_gpc_hf] + res_x + [res_gpc_lf]
    
    df = pd.DataFrame()
    for method_name, res in zip(method_names, results):
        if res is not None:
            for i in range(len(sk)):
                for j in range(len(res)):
                    df = df.append([[method_name, sk[i], res[j, i]]], ignore_index=True)
    df.columns = ['method', 'metric', 'value']
    return df

def run_tests_budget_gpc_only(X, y, scoring, methods, y_gold = None, total_points = 300, runs = 10, verbose = True):
    ma = total_points/len(y)
    res_gpc_hf, sk = test_clf_wrt_train_size_single_alpha(methods['ss_gpc'], scoring, X, y, y_gold=y_gold, runs = runs, alpha=ma, verbose = verbose)
    
    method_names = ['gpc hf'] 
    results = [res_gpc_hf] 
    
    df = pd.DataFrame()
    for method_name, res in zip(method_names, results):
        if res is not None:
            for i in range(len(sk)):
                for j in range(len(res)):
                    df = df.append([[method_name, sk[i], res[j, i]]], ignore_index=True)
    df.columns = ['method', 'metric', 'value']
    return df

def plot_df_resutls(df, corrupt_probability, hf_points, dataset_name):
    sk = sorted(df['metric'].unique())
    plt.figure()
    nonboxmethods = ['major_vote', 'logit cv10', 'xgb cv10']
    sns.boxplot(data = df[df['method'].apply(lambda m: m not in nonboxmethods)], x = 'metric', y = 'value', hue = 'method', palette='Set3',
            linewidth = 0.5, flierprops={'marker':'.', 'markersize':5}, order = sk)
    for method, c in zip(nonboxmethods,
                         ['black', 'blue', 'green']):
        mv_df = df[df['method'] == method]
        for i in range(len(sk)):
            plt.hlines(mv_df[mv_df['metric'] == sk[i]].median(), i - 0.5, i + 0.5, linestyle = '--', linewidth = 1., label = method if i == 0 else None,
                color = c)
    plt.legend(bbox_to_anchor=(1., 1.), loc=2)
    plt.title(dataset_name + '\n' + 'p = ' + str(corrupt_probability) + '; ' + 'hf = ' + str(hf_points))

def bin_search_decreasing_function(interval, fun, target, eps = 1e-3):
    L, R = interval
    L_val = fun(L)
    R_val = fun(R)
    while R - L > eps:
        M = (L + R)/2
        M_val = fun(M)
        if M_val > target:
            L, L_val = M, M_val
        else:
            R, R_val = M, M_val
    return M

def noise_level(rho, f_high, delta, grid_points):
    cur_f = lambda x:f_high(x) * rho + delta(x)
    grid = cur_f(grid_points)
    grid_high = f_high(grid_points)
    return 1 - np.mean((grid > 0) == (grid_high > 0))
    
def fit_rho_to_noise_level(f_high, delta, target_noise, grid_points):
    fun = lambda rho: noise_level(rho, f_high, delta, grid_points)
    return bin_search_decreasing_function((0.001, 100.), fun, target_noise)


def get_binary_dataset(dataset_alias):
    enc = OneHotEncoder()
    if dataset_alias == 'diabetes':
        data = fetch_data('diabetes')
        X, y = data[data.columns[:-1]].values, (data['target'].values == 2).astype(int)
    if dataset_alias == 'german':
        data = fetch_data('german')
        X, y = data[data.columns[:-1]].values.astype(float), data['target'].values
    if dataset_alias == 'waveform-40':
        data = fetch_data('waveform-40')
        X, y = data[data.columns[:-1]].values.astype(float), (data['target'].values == 0).astype(int)
    if dataset_alias == 'satimage-1':
        data = fetch_data('satimage')
        X, y = data[data.columns[:-1]].values.astype(float), (data['target'].values == 1).astype(int)
    if dataset_alias == 'splice':
        data = fetch_data('splice')
        X, y = data[data.columns[:-1]].values.astype(float), (data['target'].values == 0).astype(int)
        X = enc.fit_transform(X).toarray()
    if dataset_alias == 'spambase':
        data = fetch_data('spambase')
        X, y = data[data.columns[:-1]].values.astype(float), data['target'].values
    if dataset_alias == 'hypothyroid':
        data = fetch_data('hypothyroid')
        X, y = data[data.columns[:-1]].values.astype(float), data['target'].values
    if dataset_alias == 'mushroom':
        data = fetch_data('mushroom')
        X, y = data[data.columns[:-1]].values.astype(float), data['target'].values
        X = enc.fit_transform(X).toarray()
    return X, y