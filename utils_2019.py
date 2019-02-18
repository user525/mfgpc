from tqdm import tqdm
import numpy as np
from copy import deepcopy
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def test_clf(clf, X_train_lf, y_train_lf, X_train_hf, y_train_hf, X_test, y_test, scoring, mode=''):
    '''
    Fits classifier on train sample and scores it on test sample.

    mode: 
        'multi-fidelity' - calls .fit method from 4 args: X_train_lf, y_train_lf, X_train_hf, y_train_hf
        'high-fidelity' - calls .fit method from 2 args (in standard sklearn notation): X_train_hf, y_train_hf
        'concatenation' - calls .fit method from 2 args (in standard sklearn notation) of concatenated Xs and ys: (X_train_lf, X_train_hf), (y_train_lf, y_train_hf)
        'stacking' - first trains clf on X_train_lf, y_train_lf, then adds its predictions for X_train_hf and X_test and fits method for them as in high-fidelity mode
    '''
    
    if mode == 'multi-fidelity':
        clf.fit(X_train_lf, y_train_lf, X_train_hf, y_train_hf)
    elif mode == 'high-fidelity':
        # print('X_train_hf.shape:', X_train_hf.shape)
        # print('y_train_hf:', y_train_hf)
        clf.fit(X_train_hf, y_train_hf)
    elif mode == 'concatenation':
        X_train_hf = np.vstack((X_train_lf, X_train_hf))
        y_train_hf = np.hstack((y_train_lf, y_train_hf))
        clf.fit(X_train_hf, y_train_hf)
    elif mode == 'stacking':
        clf.fit(X_train_lf, y_train_lf)
        X_train_hf = np.hstack((X_train_hf, clf.predict_proba(X_train_hf)))
        X_test = np.hstack((X_test, clf.predict_proba(X_test)))
        clf.fit(X_train_hf, y_train_hf)
    else:
        raise Exception('unknown mode')
        
    scoring_keys = sorted(scoring.keys())
    scores = []
    for s in scoring_keys:
        # print('X_test.shape:', X_test.shape)
        # print(clf.predict_proba(X_test))
        # print(clf.predict(X_test).shape)
        scores.append(scoring[s](clf, X_test, y_test))
    
    return scores


def run_tests_clf(clf, X, y_lf, y_hf, y_groundtruth, scoring, test_size, train_lf_size, train_hf_size, runs=10, mode='', verbose=False):
    '''
    Tests classifier in multiple randomized runs w.r.t. specified test paramters.
    '''
    num_classes = len(set(y_hf)|set(y_lf))
    all_scores = []
    seed_offset = 0
    for seed in tqdm(range(runs), disable=not verbose):
        cur_seed = seed
        
        is_y_train_good = False
        while not is_y_train_good:
            cur_seed = seed + seed_offset
            np.random.seed(cur_seed)
            test_inds = np.random.choice(len(X), test_size, replace=False)
            not_test_inds = np.setdiff1d(np.arange(len(X)), test_inds)
            train_lf_inds = np.random.choice(not_test_inds, train_lf_size, replace=False)
            train_hf_inds = np.random.choice(not_test_inds, train_hf_size, replace=False)
            is_y_train_good = ((len(set(y_lf[train_lf_inds])) == num_classes)|(mode == 'high-fidelity')) & (len(set(y_hf[train_hf_inds])) == num_classes)
            seed_offset += 1
            if seed_offset > 1000:
                print('WARNING! stacked in loop "while not is_y_train_good", breaknig')
                print(mode, train_lf_size, train_hf_size)
                break
            
        scores = test_clf(clf, 
                          X[train_lf_inds] if mode != 'high-fidelity' else None, y_lf[train_lf_inds] if mode != 'high-fidelity' else None, 
                          X[train_hf_inds], y_hf[train_hf_inds], 
                          X[test_inds], y_groundtruth[test_inds],
                          scoring, mode=mode
                         )
        
        all_scores.append([cur_seed] + scores)
    return np.array(all_scores)


def make_test_results_df(method, m_name, mode, kwargs):
    cur_kwargs = deepcopy(kwargs)
    cur_kwargs['clf'] = method
    cur_kwargs['mode'] = mode

    test_results = run_tests_clf(**cur_kwargs)
    
    col_names = ['seed'] + sorted(cur_kwargs['scoring'].keys())
    df = pd.DataFrame(test_results, columns=col_names)
    df['method'] = m_name
    df['mode'] = mode
    col_names = ['method', 'mode'] + col_names
    
    return df[col_names]

def run_tests_all_clfs(methods, X, y_lf, y_hf, y_groundtruth, scoring, test_size, train_lf_size, train_hf_size, runs=10, modes=['high-fidelity', 'concatenation', 'stacking'], verbose=False):
    '''
    Tests methods with run_tests_clf function and forms table of results
    '''

    df_all = pd.DataFrame()

    kwargs = {
      'X':X, 
      'y_lf':y_lf, 
      'y_hf':y_hf, 
      'y_groundtruth':y_groundtruth, 
      'scoring':scoring,
      'test_size':test_size, 
      'train_lf_size':train_lf_size, 
      'train_hf_size':train_hf_size, 
      'runs':runs, 
      'verbose':verbose
    }

    for m_name in ['ss_gpc', 'ss_logit', 'xgb']:
        if m_name in methods:
            for mode in modes:
                if verbose:
                    print(m_name, mode)
                tmp = make_test_results_df(methods[m_name], m_name, mode, kwargs)
                # print(tmp.shape)
                df_all = df_all.append(tmp, ignore_index=False)

    if 'major_vote' in methods:
        if verbose:
            print('major_vote', 'high-fidelity')
        df_all = df_all.append(make_test_results_df(methods['major_vote'], 'major_vote', 'high-fidelity', kwargs), ignore_index=False)

    if 'ss_mf_gpc' in methods:
        if verbose:
            print('ss_mf_gpc', 'multi-fidelity')
        df_all = df_all.append(make_test_results_df(methods['ss_mf_gpc'], 'ss_mf_gpc', 'multi-fidelity', kwargs), ignore_index=False)

    if 'hetmogp' in methods:
        if verbose:
            print('hetmogp', 'multi-fidelity')
        df_all = df_all.append(make_test_results_df(methods['hetmogp'], 'hetmogp', 'multi-fidelity', kwargs), ignore_index=False)

    return df_all


def make_roc_auc_profile(dm_df, acc_ticks=100):
    num_solvers = dm_df.shape[1]
    dmz = np.zeros((num_solvers, acc_ticks), float)
    accs = np.linspace(0, 1, acc_ticks)
    for acc_i, acc_threshold in enumerate(accs):
        for solver_i in range(num_solvers):
            dmz[solver_i, acc_i] = (1 - dm_df[dm_df.columns[solver_i]] > acc_threshold).mean()
    return dmz, accs, num_solvers


def plot_roc_auc_profile(dm_df, dmz, accs, num_solvers, title=None, plot_legend=False):
    colors = [cm.Set1(0) if x.find('gpc') != -1 else (cm.Set1(1) if x.find('xgb') != -1 else (cm.Set1(2) if x.find('logit') != -1 else cm.Greys(0.5))) for x in dm_df.columns]
    linestyles = ['--' if x.find('stacking') != -1 else ('-.' if x.find('oncatenation') != -1 else ('-' if x.find('major_vote') != -1 else ':')) for x in dm_df.columns]

    plt.figure(figsize = (5, 5))
    for cnt_solver in range(num_solvers):
        if dm_df.columns[cnt_solver].find('ss_mf_gpc') == -1 and dm_df.columns[cnt_solver].find('hetmogp') == -1:
            plt.plot(accs, dmz[cnt_solver], label = dm_df.columns[cnt_solver], color = colors[cnt_solver], linestyle=linestyles[cnt_solver])
    if 'ss_mf_gpc:multi-fidelity' in dm_df.columns:
        cnt_solver = dm_df.columns.tolist().index('ss_mf_gpc:multi-fidelity')
        plt.plot(accs, dmz[cnt_solver], label = dm_df.columns[cnt_solver], color = 'k')
    if 'hetmogp:multi-fidelity' in dm_df.columns:
        cnt_solver = dm_df.columns.tolist().index('hetmogp:multi-fidelity')
        plt.plot(accs, dmz[cnt_solver], label = dm_df.columns[cnt_solver], color = 'orange')
    if 'GPMA' in dm_df.columns:
        cnt_solver = dm_df.columns.tolist().index('GPMA')
        plt.plot(accs, dmz[cnt_solver], label = dm_df.columns[cnt_solver], color = 'magenta')
    plt.xlim([0.4, 1])
    if plot_legend:
        plt.legend(loc = 2, bbox_to_anchor = (1, 1))
    if title is not None:
        plt.title(title);


def get_sub_dm_df(file_path, conditions={}, method_modes_to_plot=[]):
    full_dfs = pd.read_csv(file_path)
    sub_df = full_dfs
    for c in conditions:
        sub_df = sub_df[sub_df[c] == conditions[c]]
    sub_dm_df = pd.DataFrame()
    for n, g in sub_df.groupby(['method', 'mode']):
        if n in method_modes_to_plot:
            sub_dm_df[n] = 1 - g['ROCAUC'].values
    return sub_dm_df


