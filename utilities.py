import theano.tensor as tt
import pymc3 as pm
import numpy as np
import theano
from scipy.special import expit as invlogit, logit
import matplotlib.pyplot as plt


def generate_dataset(n_l = 30, n_h = 10, m = 100, noise = 0.05, rho = 0.5,
                  lengthscale_l = 0.1, f_scale_l = 1, lengthscale_d = 0.3, f_scale_d = 0.8, 
                  points_seed = 0, gp_seed = 0):
    np.random.seed(points_seed)
    X0_h = np.sort(np.random.rand(n_h))[:, None]
    X0_l = np.sort(np.random.rand(n_l))[:, None]
    X = np.linspace(0, 1, m)[:, None]
    cov_l = f_scale_l * pm.gp.cov.ExpQuad(1, lengthscale_l)
    cov_d = f_scale_d * pm.gp.cov.ExpQuad(1, lengthscale_d)

    K = tt.alloc(0.0, n_l + n_h, n_l + n_h)
    K = tt.set_subtensor(K[:n_l, :n_l], cov_l(X0_l))
    K = tt.set_subtensor(K[n_l:n_l + n_h, n_l:n_l + n_h], rho**2 * cov_l(X0_h) + cov_d(X0_h))
    K = tt.set_subtensor(K[:n_l, n_l:n_l + n_h], rho * cov_l(X0_l, X0_h))
    K = tt.set_subtensor(K[n_l:n_l + n_h, :n_l], rho * cov_l(X0_h, X0_l))
    K_noiseless = K.copy() + 1e-6 * tt.eye(n_l + n_h)
    K = tt.inc_subtensor(K[:n_l, :n_l], noise * tt.eye(n_l))
    K = tt.inc_subtensor(K[:, :], 1e-6 * tt.eye(n_l + n_h))
    
    K_s = tt.alloc(0.0, n_l + n_h, len(X))
    K_s = tt.set_subtensor(K_s[:n_l, :], rho * cov_l(X0_l, X))
    K_s = tt.set_subtensor(K_s[n_l:n_l + n_h, :], rho**2 * cov_l(X0_h, X) + cov_d(X0_h, X))
    
    np.random.seed(gp_seed)
    f = np.random.multivariate_normal(mean=np.zeros(n_l + n_h), cov=K.eval())
    f_latent = np.array(f)
    f[f > 0] = 1
    f[f <= 0] = 0
    
    
    K_cross = tt.alloc(0.0, n_l + n_h, n_l)
    K_cross = tt.set_subtensor(K_cross[:n_l, :], rho * cov_l(X0_l, X0_l))
    K_cross = tt.set_subtensor(K_cross[n_l:n_l + n_h, :], rho**2 * cov_l(X0_h, X0_l) + cov_d(X0_h, X0_l))
    L = np.linalg.cholesky(K.eval())
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_latent))
    post_mean = np.dot(K_cross.T.eval(), alpha)

    low_fidelity_error_inds = np.where(np.sign(post_mean) != np.sign(f_latent[:n_l]))[0]
    
    return X0_l, X0_h, X, cov_l, cov_d, K, K_noiseless, K_s, f, f_latent, low_fidelity_error_inds

def mcmc_mf_clf(n_l, n_h, K, K_noiseless, K_s, f, f_latent, trials=1000):
    with pm.Model(theano_config={"compute_test_value": "ignore"}) as model:
        f_sample = pm.Flat('f_sample', shape=n_l + n_h)
        f_transform = pm.invlogit(f_sample)
        y = pm.Binomial('y', observed=f, n=np.ones(n_l + n_h), p=f_transform, shape=n_l + n_h)
        L_h = tt.slinalg.cholesky(K)
        f_pred = pm.Deterministic('f_pred', pm.invlogit(tt.dot(K_s.T, tt.slinalg.solve(L_h.T, tt.slinalg.solve(L_h, f_sample)))))
        ess_step = pm.EllipticalSlice(vars=[f_sample], prior_cov=K)
        trace = pm.sample(trials, tune = trials, start=model.test_point, step=[ess_step], progressbar=True, njobs = 1)
    return model, trace

def mcmc_sf_clf(n, K, K_noiseless, K_s, f, f_latent):
    with pm.Model(theano_config={"compute_test_value": "ignore"}) as model:
        f_sample = pm.Flat('f_sample', shape=n)
        f_transform = pm.invlogit(f_sample)
        y = pm.Binomial('y', observed=f, n=np.ones(n), p=f_transform, shape=n)
        L_h = tt.slinalg.cholesky(K)
        f_pred = pm.Deterministic('f_pred', pm.invlogit(tt.dot(K_s.T, tt.slinalg.solve(L_h.T, tt.slinalg.solve(L_h, f_sample)))))
        ess_step = pm.EllipticalSlice(vars=[f_sample], prior_cov=K)
        trace = pm.sample(1000, tune = 1000, start=model.test_point, step=[ess_step], progressbar=True, njobs = 1)
    return model, trace



def plot_dataset(X0_l, X0_h, X, cov_l, cov_d, K, K_noiseless, K_s, f, f_latent, low_fidelity_error_inds = None,
                 trace_vals = None, trace_high_only_vals = None, post_mean_hf_label_regr = None, is_legend_on = False):
    n_l = len(X0_l)
    n_h = len(X0_h)
    
    fig, ax = plt.subplots(figsize=(14, 4));
    
    ax.scatter(X0_h, f[n_l:n_l + n_h], s=40, color='black', label='$D_H$', zorder = 2)
    
    if low_fidelity_error_inds is not None:
        low_fidelity_correct_inds = np.setdiff1d(np.arange(n_l), low_fidelity_error_inds)
        ax.scatter(X0_l[low_fidelity_correct_inds], f[low_fidelity_correct_inds], s=20, color='grey', 
                   label='$D_L$', marker = 'o', facecolor = 'none')
        ax.scatter(X0_l[low_fidelity_error_inds], f[low_fidelity_error_inds], color = 'coral', marker = 'x',
                  label = 'Ошибки в $D_L$')
    else:
        ax.scatter(X0_l, f[:n_l], s=20, color='grey', label='$D_L$', marker = 'o', facecolor = 'none')
    

    plt.hlines(0.5, 0, 3, linestyle=':', linewidth = 1, color = 'grey', label = 'Граница классов')

    L_hf = np.linalg.cholesky(K.eval())
    alpha_hf = np.linalg.solve(L_hf.T, np.linalg.solve(L_hf, f_latent))
    post_mean_hf = invlogit(np.dot(K_s.T.eval(), alpha_hf))
    ax.plot(X, post_mean_hf, color='g', alpha=0.8, label='Истинное $\sigma(f_H)$');

   

    if post_mean_hf_label_regr is not None:
        ax.plot(X, post_mean_hf_label_regr, color = 'gray', label= 'Регрессия', linestyle = '--')
    
    if trace_high_only_vals is not None:
        ax.plot(X, invlogit(np.mean(logit(trace_high_only_vals), axis = 0)), color = 'blue', label= '$p(c(x_*)=1|D_H, x_*)$',
               linestyle = '--')
    
    if trace_vals is not None:
        ax.plot(X, invlogit(np.mean(logit(trace_vals), axis = 0)), color = 'darkblue', label= '$p(c(x_*)=1|D_L, D_H, x_*)$')
        
    
    plt.xlabel('$\Omega$', fontsize = 16)
    plt.ylabel('Значения классов и прогнозов', fontsize = 14)
    ax.set_xlim(0, 1);
    ax.set_ylim(-0.1, 1.1);
    if is_legend_on:
        ax.legend(bbox_to_anchor = (1, 1), loc = 2, fontsize = 14);
    

