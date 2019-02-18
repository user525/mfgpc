"""Gaussian processes classification."""

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# Generalization : fmsnew
#
# License: BSD 3 clause

import warnings
from operator import itemgetter

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve, sqrtm, solve_triangular
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import erf, expit

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.gaussian_process.kernels \
    import RBF, CompoundKernel, ConstantKernel as C
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import warnings

class GLOBALLOGGER:
    class __GLOBALLOGGER:
        def __init__(self):
            self.logged_vars = {}
           
    instance = None
    
    def __init__(self):
        if not GLOBALLOGGER.instance:
            GLOBALLOGGER.instance = GLOBALLOGGER.__GLOBALLOGGER()
    
    def log_variable(self, var, var_name):
        self.instance.logged_vars[var_name] = var

#global_logger = GLOBALLOGGER()

        
# Values required for approximating the logistic sigmoid by
# error functions. coefs are obtained via:
# x = np.array([0, 0.6, 2, 3.5, 4.5, np.inf])
# b = logistic(x)
# A = (erf(np.dot(x, self.lambdas)) + 1) / 2
# coefs = lstsq(A, b)[0]
LAMBDAS = np.array([0.41, 0.4, 0.37, 0.44, 0.39])[:, np.newaxis]
COEFS = np.array([-1854.8214151, 3516.89893646, 221.29346712,
                  128.12323805, -2010.49422654])[:, np.newaxis]


class _BinaryMultiFidelityGaussianProcessClassifierLaplace(BaseEstimator):
    """Binary Gaussian process classification based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2, and 5.1 of
    ``Gaussian Processes for Machine Learning'' (GPML) by Rasmussen and
    Williams.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the  signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer: int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer=0 implies that one
        run is performed.

    max_iter_predict: int, optional (default: 100)
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.

    warm_start : bool, optional (default: False)
        If warm-starts are enabled, the solution of the last Newton iteration
        on the Laplace approximation of the posterior mode is used as
        initialization for the next call of _posterior_mode(). This can speed
        up convergence when _posterior_mode is called several times on similar
        problems as in hyperparameter optimization.

    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (also required for prediction)

    y_train_ : array-like, shape = (n_samples,)
        Target values in training data (also required for prediction)

    classes_ : array-like, shape = (n_classes,)
        Unique class labels.

    kernel_ : kernel object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters

    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in X_train_

    pi_ : array-like, shape = (n_samples,)
        The probabilities of the positive class for the training points
        X_train_

    W_sr_ : array-like, shape = (n_samples,)
        Square root of W, the Hessian of log-likelihood of the latent function
        values for the observed labels. Since W is diagonal, only the diagonal
        of sqrt(W) is stored.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``

    """
    def __init__(self, kernel=None, optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0, max_iter_predict=100,
                 warm_start=False, copy_X_train=True, random_state=None, rho=0, rho_bounds=(-1, 1),
                 eval_gradient=False):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.rho = rho
        self.rho_bounds = rho_bounds
        self.eval_gradient = eval_gradient

    def fit(self, X, y):
        """Fit Gaussian process classification model

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples,)
            Target values, must be binary

        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_l_ = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")
        else:
            self.kernel_l_ = clone(self.kernel)
        self.kernel_d_ = clone(self.kernel_l_)

        self.rng = check_random_state(self.random_state)

        self.X_train_ = np.copy(X[:, :-1]) if self.copy_X_train else X[:, :-1]
        self.n_l_ = np.min(np.where(X[:, -1] == 1)[0])
        self.n_ = len(X)
        self.n_h_ = len(X) - self.n_l_

        # Encode class labels and check that it is a binary classification
        # problem
        label_encoder = LabelEncoder()
        self.y_train_ = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        if self.classes_.size > 2:
            raise ValueError("%s supports only binary classification. "
                             "y contains classes %s"
                             % (self.__class__.__name__, self.classes_))
        elif self.classes_.size == 1:
            raise ValueError("{0:s} requires 2 classes.".format(
                self.__class__.__name__))
            
        theta_initial = np.hstack((np.array([self.rho]), self.kernel_l_.theta, self.kernel_d_.theta))
        if self.optimizer is not None:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=self.eval_gradient): # todo: switch to true to speed up optimization
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            
            #print(np.array(self.rho_bounds)[np.newaxis])
            #print(self.kernel_l_.bounds, self.kernel_d_.bounds)
            theta_bounds = np.r_[np.array(self.rho_bounds)[np.newaxis],
                            self.kernel_l_.bounds,
                            self.kernel_d_.bounds]
            #print(theta_initial)
            #print(theta_bounds)
            
            
            optima = [self._constrained_optimization(
                obj_func, theta_initial, theta_bounds, self.eval_gradient
            )]
            #print(theta_initial)

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not (np.isfinite(self.kernel_l_.bounds).all() and np.isfinite(self.kernel_d_.bounds).all() and np.isfinite(self.rho_bounds).all()):
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                #print(np.array(self.rho_bounds).reshape(1, -1).shape)
                
                bounds = np.vstack((np.array(self.rho_bounds).reshape(1, -1), self.kernel_l_.bounds, self.kernel_d_.bounds))
                for iteration in range(self.n_restarts_optimizer):
                    #print(theta_initial)
                    theta_initial = np.hstack((
                                        self.rng.uniform(bounds[0, 0], bounds[0, 1]),
                                        np.exp(self.rng.uniform(bounds[1:, 0], bounds[1:, 1]))
                                        
                    ))
                    
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error')
                        try:
                            
                            optima.append(
                                self._constrained_optimization(obj_func, theta_initial,
                                                               bounds, self.eval_gradient))
                        except Warning as w:
                            print('(warning in constrained hyperparameters optimization)', w, theta_initial)
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            
            lml_values = list(map(itemgetter(1), optima))
            #print(eval_gradient)
            #print('lml',lml_values)
            best_hyperparams = optima[np.argmin(lml_values)][0]
            #print(np.argmin(lml_values), best_hyperparams)
            self.rho = best_hyperparams[0]
            self.kernel_l_.theta = best_hyperparams[1:1 + len(self.kernel_l_.theta)]
            self.kernel_d_.theta = best_hyperparams[1 + len(self.kernel_l_.theta):]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(theta_initial)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = np.vstack((
                np.hstack(( self.kernel_l_(self.X_train_), np.zeros((self.n_, self.n_h_)) )),
                np.hstack(( np.zeros((self.n_h_, self.n_)), self.kernel_d_(self.X_train_[self.n_l_:]),  ))
                      ))

        #
        _, (self.pi_, self.W_sr_, self.L_, _, _, _) = \
            self._posterior_mode(K, self.rho, return_temporaries=True)

        return self

    # TODO: check that in the curresnt setup the predictions preserve their structure as in 3.21?
    # currenlty implement is as exact mfgp inference
    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array, shape = (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """
        check_is_fitted(self, ["X_train_", "y_train_", "pi_", "W_sr_", "L_"])

        # As discussed on Section 3.4.2 of GPML, for making hard binary
        # decisions, it is enough to compute the MAP of the posterior and
        # pass it through the link function
        #K_star = self.kernel_(self.X_train_, X)  # K_star =k(x_star)
        
        K_star = np.vstack((self.rho * self.kernel_l_(self.X_train_[:self.n_l_], X), 
                            self.rho**2 * self.kernel_l_(self.X_train_[self.n_l_:], X) + self.kernel_d_(self.X_train_[self.n_l_:], X)))
        K = np.vstack((
                    np.hstack(( self.kernel_l_(self.X_train_[:self.n_l_]), 
                                self.rho * self.kernel_l_(self.X_train_[:self.n_l_], self.X_train_[self.n_l_:]) )),
                    np.hstack(( self.rho * self.kernel_l_(self.X_train_[self.n_l_:], self.X_train_[:self.n_l_]), 
                                self.rho**2 *self.kernel_l_(self.X_train_[self.n_l_:]) + self.kernel_d_(self.X_train_[self.n_l_:]) ))
                     ))
        K += np.eye(K.shape[0])*1e-6 # for numerical stability
        L = cholesky(K, lower=True)
        f = np.hstack((self.f_cached[:self.n_l_],
                       self.rho * self.f_cached[self.n_l_:self.n_l_ + self.n_h_] + self.f_cached[self.n_l_ + self.n_h_:]))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, f))
        
        #-----------original--------------
        #f_star = K_star.T.dot(self.y_train_ - self.pi_)  # Algorithm 3.2,Line 4
        #-----------general---------------
        #f_star = K_star.T.dot(self._d_log_likelihood(self.f_cached))  # Algorithm 3.2,Line 4
        #f_star = K_star.T.dot(np.linalg.inv(K).dot(f))# exact mfgp inference, slow and weak numerical stability (sinse cholesky is not used)
        f_star = K_star.T.dot(alpha)
        #---------------------------------
        return np.where(f_star > 0, self.classes_[1], self.classes_[0])

    # TODO: check that in the curresnt setup the predictions preserve their structure as in 3.21?
    # currenlty implement is as exact mfgp inference
    def predict_proba(self, X):
        """Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array-like, shape = (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute ``classes_``.
        """
        check_is_fitted(self, ["X_train_", "y_train_", "pi_", "W_sr_", "L_"])

        # Based on Algorithm 3.2 of GPML
        #K_star = self.kernel_(self.X_train_, X)  # K_star =k(x_star)
        #print(X.shape)
        #print(self.X_train_[:self.n_l_].shape)
        #print(self.kernel_l_(self.X_train_[:self.n_l_], X).shape)
        #print(self.kernel_l_(self.X_train_[self.n_l_:], X).shape)
        K_star = np.vstack((self.rho * self.kernel_l_(self.X_train_[:self.n_l_], X), 
                            self.rho**2 * self.kernel_l_(self.X_train_[self.n_l_:], X) + self.kernel_d_(self.X_train_[self.n_l_:], X)))
        K = np.vstack((
                    np.hstack(( self.kernel_l_(self.X_train_[:self.n_l_]), 
                                self.rho * self.kernel_l_(self.X_train_[:self.n_l_], self.X_train_[self.n_l_:]) )),
                    np.hstack(( self.rho * self.kernel_l_(self.X_train_[self.n_l_:], self.X_train_[:self.n_l_]), 
                                self.rho**2 *self.kernel_l_(self.X_train_[self.n_l_:]) + self.kernel_d_(self.X_train_[self.n_l_:]) ))
                     ))
        K += np.eye(K.shape[0])*1e-6 # for numerical stability
        L = cholesky(K, lower=True)
        f = np.hstack((self.f_cached[:self.n_l_],
                       self.rho * self.f_cached[self.n_l_:self.n_l_ + self.n_h_] + self.f_cached[self.n_l_ + self.n_h_:]))
        alpha = solve_triangular(L.T, solve_triangular(L, f, lower=True), lower=False)
        v = solve_triangular(L, K_star, lower=True)
        #-----------original--------------
        #f_star = K_star.T.dot(self.y_train_ - self.pi_)  # Line 4
        #-----------general---------------
        #f_star = K_star.T.dot(self._d_log_likelihood(self.f_cached))  # Line 4
        #f_star = K_star.T.dot(np.linalg.inv(K).dot(f)) # exact mfgp inference, slow and weak numerical stability (sinse cholesky is not used)
        f_star = K_star.T.dot(alpha)
        #---------------------------------
        #v = solve(self.L_, self.W_sr_[:, np.newaxis] * K_star)  # Line 5
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        #var_f_star = self.kernel_.diag(X) - np.einsum("ij,ij->j", v, v)
        
        #var_f_star = np.diag(self.rho**2 *self.kernel_l_(X) + self.kernel_d_(X) - K_star.T.dot(np.linalg.inv(K).dot(K_star))) # exact mfgp inference, slow and weak numerical stability (sinse cholesky is not used)
        #var_f_star = np.diag(self.rho**2 *self.kernel_l_(X) + self.kernel_d_(X) - v.T.dot(v))
        var_f_star = self.rho**2 *self.kernel_l_.diag(X) + self.kernel_d_.diag(X) - np.einsum("ij,ij->j", v, v)
        # Line 7:
        # Approximate \int log(z) * N(z | f_star, var_f_star)
        # Approximation is due to Williams & Barber, "Bayesian Classification
        # with Gaussian Processes", Appendix A: Approximate the logistic
        # sigmoid by a linear combination of 5 error functions.
        # For information on how this integral can be computed see
        # blitiri.blogspot.de/2012/11/gaussian-integral-of-error-function.html
        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = np.sqrt(np.pi / alpha) \
            * erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS**2))) \
            / (2 * np.sqrt(var_f_star * 2 * np.pi))
        pi_star = (COEFS * integrals).sum(axis=0) + .5 * COEFS.sum()

        return np.vstack((1 - pi_star, pi_star)).T

    
    def debug_components_rho_derivative(self, theta=None):
        kernel_l = self.kernel_l_.clone_with_theta(theta[1: 1 + len(self.kernel_l_.theta)])
        kernel_d = self.kernel_d_.clone_with_theta(theta[-len(self.kernel_d_.theta):])
        K = np.vstack(( # todo: use np.block
                        np.hstack(( kernel_l(self.X_train_), np.zeros((self.n_, self.n_h_)) )),
                        np.hstack(( np.zeros((self.n_h_, self.n_)), kernel_d(self.X_train_[self.n_l_:]) ))
                  ))
        rho = theta[0]
        Z, (pi, W_sr, L, b, a, nabla_ll_f) = \
            self._posterior_mode(K, rho, return_temporaries=True)
        f_combined = np.hstack((self.f_cached[:self.n_l_], rho * self.f_cached[self.n_l_:self.n_l_+self.n_h_] + self.f_cached[self.n_l_+self.n_h_:]))
        component_1 = self.f_cached.T.dot(np.linalg.inv(K + np.eye(K.shape[0])*1e-6)).dot(self.f_cached)
        component_2 = np.sum(self._log_likelihood(f_combined))
        B = np.eye(W_sr.shape[0]) + W_sr.dot(K).dot(W_sr)
        component_3 = np.log(np.linalg.det(B))
        
        W_sr_K = W_sr.dot(K)
        W = W_sr.dot(W_sr)
        assert np.linalg.norm(np.eye(W.shape[0]) + W_sr_K.dot(W_sr) - B) < 1e-6
        L = cholesky(B, lower=True)
        b = W.dot(self.f_cached) + nabla_ll_f
        a = b - W_sr.dot(cho_solve((L, True), W_sr_K.dot(b)))
        f = K.dot(a)
        assert np.linalg.norm(f - self.f_cached) < 1e-6
        assert np.abs(a.T.dot(f) - component_1) < 1e-3
        
        assert np.abs(self._log_likelihood(f_combined).sum() - component_2) < 1e-3
        #print(2*np.log(np.diag(L)).sum(), component_3)
        assert np.abs(2*np.log(np.diag(L)).sum() - component_3) < 1e-3
        
        lml = -0.5 * a.T.dot(f) \
            + self._log_likelihood(f_combined).sum() \
            - np.log(np.diag(L)).sum()
        
        return component_1, component_2, component_3, -0.5*component_1 + component_2 - 0.5 * component_3, lml
    
    def log_marginal_likelihood(self, theta=None, eval_gradient=False, debug_verbose = False):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        
        kernel_l = self.kernel_l_.clone_with_theta(theta[1: 1 + len(self.kernel_l_.theta)])
        kernel_d = self.kernel_d_.clone_with_theta(theta[-len(self.kernel_d_.theta):])

        if eval_gradient:
            K_l, K_l_gradient = kernel_l(self.X_train_, eval_gradient=True)
            K_d, K_d_gradient = kernel_d(self.X_train_[self.n_l_:], eval_gradient=True)
            
            #print(K_l.shape, K_d.shape)
            
            tmp_r1 = np.hstack(( K_l, np.zeros((self.n_, self.n_h_)) ))
            tmp_r2 = np.hstack(( np.zeros((self.n_h_, self.n_)), K_d ))
            
            #print(tmp_r1.shape, tmp_r2.shape)
            K = np.vstack((
                            tmp_r1,
                            tmp_r2
                         ))
            K_gradient = np.zeros((self.n_l_ + 2*self.n_h_, self.n_l_ + 2*self.n_h_, K_l_gradient.shape[2] + K_d_gradient.shape[2]))
            K_gradient[:self.n_, :self.n_, :K_l_gradient.shape[2]] = K_l_gradient
            K_gradient[self.n_:, self.n_:, K_l_gradient.shape[2]:] = K_d_gradient
        else:
            K = np.vstack(( # todo: use np.block
                            np.hstack(( kernel_l(self.X_train_), np.zeros((self.n_, self.n_h_)) )),
                            np.hstack(( np.zeros((self.n_h_, self.n_)), kernel_d(self.X_train_[self.n_l_:]) ))
                      ))

        rho = theta[0]
        # Compute log-marginal-likelihood Z and also store some temporaries
        # which can be reused for computing Z's gradient
        Z, (pi, W_sr, L, b, a, nabla_ll_f) = \
            self._posterior_mode(K, rho, return_temporaries=True)

        #print(Z, theta)
        if not eval_gradient:
            return Z
        ########################################################################################
        ########################################################################################
        ############    TODO: CHECK THE PART BELOW & ADD GRADIENT W.R.T. RHO   #################
        ########################################################################################
        ########################################################################################
        f_combined = np.hstack((self.f_cached[:self.n_l_], rho * self.f_cached[self.n_l_:self.n_l_+self.n_h_] + self.f_cached[self.n_l_+self.n_h_:]))
        
        # Compute gradient based on Algorithm 5.1 of GPML
        d_Z = np.zeros(theta.shape[0])
        # XXX: Get rid of the np.diag() in the next line
        R = W_sr.dot(cho_solve((L, True), W_sr))  # Line 7
        #assert np.linalg.norm(R - W_sr.dot(np.linalg.inv(np.eye(K.shape[0])*(1 + 1e-6) + W_sr.dot(K).dot(W_sr))).dot(W_sr)) < 1e-3
        C = solve_triangular(L, W_sr.dot(K), lower=True)  # Line 8
        # Line 9: (use einsum to compute np.diag(C.T.dot(C))))
        #-------------original---------------
        #s_2 = -0.5 * (np.diag(K) - np.einsum('ij, ij -> j', C, C)) \
        #    * (pi * (1 - pi) * (1 - 2 * pi))  # third derivative
        #-------------general---------------
        #s_2 = -0.5 * (np.diag(K) - np.einsum('ij, ij -> j', C, C)) \
        #    * self._ddd_log_likelihood(self.f_cached)  # third derivative
        zeta = -self._ddd_log_likelihood(f_combined)
        M = K - C.T.dot(C)
#         global_logger.log_variable(M, 'M')
#         global_logger.log_variable(K, 'K')
#         global_logger.log_variable(K_gradient, 'K_gradient')
#         global_logger.log_variable(W_sr, 'W_sr')
#         global_logger.log_variable(nabla_ll_f, 'nabla_ll_f')
#         global_logger.log_variable(zeta, 'zeta')
        s_2 = np.zeros(self.n_l_ + 2*self.n_h_)
        s_2[:self.n_l_] = np.diag(M[:self.n_l_, :self.n_l_]) * zeta[:self.n_l_]
   
        slice_flh = np.s_[self.n_l_:self.n_l_ + self.n_h_]
        slice_d = np.s_[self.n_l_ + self.n_h_:]
        #slice_flh = np.s_[self.n_l_:-self.n_h_]
        #slice_d = np.s_[-self.n_h_:]      
        
        M_flh = np.diag(M[slice_flh, slice_flh])
        M_flh_d = np.diag(M[slice_flh, slice_d])
        M_d = np.diag(M[slice_d, slice_d])

        s_2[slice_flh] = (rho**3 * M_flh + 2 * rho**2 * M_flh_d + rho * M_d) * zeta[self.n_l_:]
        s_2[slice_d] = (rho**2 * M_flh + 2 * rho * M_flh_d + M_d) * zeta[self.n_l_:]
        s_2 *= -0.5
        #------------------------------------
        # calculating d_Z[0] - derivatives w.r.t rho hyperparameter
        tmp_t = (2*self.y_train_[self.n_l_:] - 1)
        f_L_X_H = self.f_cached[self.n_l_:self.n_l_+self.n_h_]
        dd_ll_dd_f_H =  self._dd_log_likelihood(f_combined)[self.n_l_:]
        
        ###################################################################################
        # todo: replace with exact formula like the one incorrect below
        f_omega = f_L_X_H * dd_ll_dd_f_H # = - f^L(X_H) * \omega(f_i)f
        #d_f_d_rho_explicit = np.hstack((np.zeros(self.n_l_),
        #                               dd_ll_dd_f_H + rho * f_omega, # this in not correct
        #                               f_omega)) 
        d_f_d_rho_explicit = np.hstack((np.zeros(self.n_l_),
                               nabla_ll_f[self.n_l_ + self.n_h_:] + rho * f_omega,
                               f_omega))
        #T = cholesky(np.eye(self.n_l_ + 2*self.n_h_) + W_sr.dot(W_sr).dot(K), lower=True) # this should be stable since we add eye, upd: no
        
        d_f_d_rho = M.dot(d_f_d_rho_explicit)
        tmp_H = np.diag(-dd_ll_dd_f_H)
        tmp_Z = f_L_X_H * np.diag(zeta[self.n_l_:])
        d_W_d_rho_explicit = np.zeros(W_sr.shape)
        d_W_d_rho_explicit[slice_flh, slice_flh] = rho**2*tmp_Z + 2*rho*tmp_H
        d_W_d_rho_explicit[slice_flh, slice_d] = rho*tmp_Z + tmp_H
        d_W_d_rho_explicit[slice_d, slice_flh] = rho*tmp_Z + tmp_H
        d_W_d_rho_explicit[slice_d, slice_d] = tmp_Z

        d_W_d_rho_implicit = np.zeros(W_sr.shape)
        # todo: get rid of these matrices with exact formulas or sparse matrices or rewrite without cycle
#         for i in range(len(self.f_cached)):
#             d_W_d_f_i = np.zeros((len(self.f_cached), len(self.f_cached)))
#             if i < self.n_l_:
#                 d_W_d_f_i[i, i] = zeta[i]
#             elif (i >= self.n_l_) and (i < self.n_l_ + self.n_h_):
#                 d_W_d_f_i[i, i] = rho**3 * zeta[i]
#                 d_W_d_f_i[i + self.n_h_, i] = rho**2 * zeta[i]
#                 d_W_d_f_i[i, i + self.n_h_] = rho**2 * zeta[i]
#                 d_W_d_f_i[i + self.n_h_, i + self.n_h_] = rho * zeta[i]
#             elif (i >= self.n_l_ + self.n_h_):
#                 d_W_d_f_i[i, i] = zeta[i - self.n_h_]
#                 d_W_d_f_i[i - self.n_h_, i] = rho * zeta[i - self.n_h_]
#                 d_W_d_f_i[i, i - self.n_h_] = rho * zeta[i - self.n_h_]
#                 d_W_d_f_i[i - self.n_h_, i - self.n_h_] = rho**2 * zeta[i - self.n_h_]
#             d_W_d_rho_implicit += d_W_d_f_i * d_f_d_rho[i]
        d_W_d_rho_implicit[:self.n_l_, :self.n_l_] = np.diag(zeta[:self.n_l_] * d_f_d_rho[:self.n_l_])
        d_W_d_rho_implicit[slice_flh, slice_flh] = np.diag(zeta[slice_flh]) * (rho**3 * d_f_d_rho[slice_flh] + rho**2 * d_f_d_rho[slice_d])
        d_W_d_rho_implicit[slice_d, slice_d] = np.diag(zeta[slice_flh]) * (rho * d_f_d_rho[slice_flh] + d_f_d_rho[slice_d])
        d_W_d_rho_implicit[slice_flh, slice_d] = np.diag(zeta[slice_flh]) * (rho**2 * d_f_d_rho[slice_flh] + rho * d_f_d_rho[slice_d])
        d_W_d_rho_implicit[slice_d, slice_flh] = d_W_d_rho_implicit[slice_flh, slice_d]
        
        d_W_d_rho = d_W_d_rho_explicit + d_W_d_rho_implicit
        
        d_log_det_B_d_rho = np.sum(M * d_W_d_rho) # = np.trace(M.dot(d_W_d_rho))
        #d_log_det_B_d_rho = np.sum(2 * (rho * M_flh + M_flh_d) * zeta[self.n_l_:]) # this is not correct wrong
        ###################################################################################
        #d_ll_d_f = self._d_log_likelihood(f_combined)
        

        chol_K = cholesky(K + np.eye(K.shape[0])*1e-6, lower=True)
        xi_left = solve_triangular(chol_K, self.f_cached, lower=True)
        xi_right = solve_triangular(chol_K, d_f_d_rho, lower=True)
        first_term = xi_left.T.dot(xi_right)
        
        
        d_ll_d_rho_explicit = np.sum(tmp_t * f_L_X_H * (1 - expit(tmp_t * f_combined[self.n_l_:])))
        d_ll_d_rho_implicit = nabla_ll_f.dot(d_f_d_rho)
        d_ll_d_rho = d_ll_d_rho_explicit + d_ll_d_rho_implicit
        
        #first_term = self.f_cached.dot(cho_solve((T, True), d_f_d_rho_explicit))
        
        if debug_verbose:
            print(first_term, d_ll_d_rho, d_log_det_B_d_rho)
        
        d_Z[0] = -first_term + d_ll_d_rho - 0.5 * d_log_det_B_d_rho
        #print(d_Z[0]) 
        # calculating derivatives w.r.t. kernel parameters
        for j in range(1, d_Z.shape[0]): # skip 1st since it is related to rho optimization
            
            C = K_gradient[:, :, j - 1]   # Line 11
            # Line 12: 
            s_1 = 0.5 * a.T.dot(C).dot(a) - 0.5 * np.trace(R.dot(C)) # TODO: calculate trace faster (only diagonal elements) see: R.T.ravel().dot(C.ravel())
            #-------------original---------------   
            #b = C.dot(self.y_train_ - pi)  # Line 13
            #-------------general----------------
            b = C.dot(nabla_ll_f)  # Line 13
            #------------------------------------
            s_3 = b - K.dot(R.dot(b))  # Line 14

            d_Z[j] = s_1 + s_2.T.dot(s_3)  # Line 15

            
        return Z, d_Z
    
    def _likelihood(self, f):
        return expit((2*self.y_train_ - 1)*f) # note: a different scale for y is used here: y_i \in {0, 1}, but the book uses y_i \in {-1, 1}
    
    def _log_likelihood(self, f):
        return -np.log(1 + np.exp(-(2*self.y_train_ - 1)*f)) # is it more numerically stable than np.log(_likelihood(y, f))? note: a different scale for y is used here: y_i \in {0, 1}, but the book uses y_i \in {-1, 1}
    
    def _d_log_likelihood(self, f):
        return self.y_train_ - expit(f) # see formula 3.15; note: a different scale for y is used here: y_i \in {0, 1}, but the book uses y_i \in {-1, 1}
    
    def _dd_log_likelihood(self, f):
        pi = expit(f)
        return -pi*(1 - pi)# see formula 3.15
    
    def _ddd_log_likelihood(self, f):
        pi = expit(f)
        return -pi*(1 - pi)*(1 - 2*pi)

    def _posterior_mode(self, K, rho, return_temporaries=False):
        """Mode-finding for binary Laplace GPC and fixed kernel.

        This approximates the posterior of the latent function values for given
        inputs and target observations with a Gaussian approximation and uses
        Newton's iteration to find the mode of this approximation.
        """
        # Based on Algorithm 3.1 of GPML

        # If warm_start are enabled, we reuse the last solution for the
        # posterior mode as initialization; otherwise, we initialize with 0
        if self.warm_start and hasattr(self, "f_cached") \
           and len(self.f_cached) == self.n_l_ + 2*self.n_h_:
            f = self.f_cached
        else:
            f = np.zeros(self.n_l_ + 2*self.n_h_, dtype=np.float64)

        # Use Newton's iteration method to find mode of Laplace approximation
        log_marginal_likelihood = -np.inf
        for _ in range(self.max_iter_predict):
            # Line 4
            
            f_combined = np.hstack((f[:self.n_l_], rho * f[self.n_l_:self.n_l_+self.n_h_] + f[self.n_l_+self.n_h_:]))
            #print(f)
            #print(f_combined)
            #print(rho)
            pi = expit(f_combined)
            #-----------original--------------
            #W = pi * (1 - pi)
            #-----------general------------
            W = np.zeros((self.n_l_ + 2*self.n_h_, self.n_l_ + 2*self.n_h_))
            
            
            slice_flh = np.s_[self.n_l_:-self.n_h_]
            slice_d = np.s_[-self.n_h_:]
            W[:self.n_l_, :self.n_l_] = np.diag(-self._dd_log_likelihood(f[:self.n_l_]))  # note that this is correct since dd doesn't depend on y
            tmp_H = np.diag(-self._dd_log_likelihood(rho * f[slice_flh] + f[slice_d])) # note that this is correct since dd doesn't depend on y
            W[slice_flh, slice_flh] = rho**2 * tmp_H
            W[slice_flh, slice_d] = rho * tmp_H
            W[slice_d, slice_flh] = rho * tmp_H
            W[slice_d, slice_d] = tmp_H
            
            #--------------------------
            
            
            # Line 5
            #W_sr = sqrtm(W + np.eye(W.shape[0])*1e-5) # todo: replace with exact formula
            W_sr = np.zeros((self.n_l_ + 2*self.n_h_, self.n_l_ + 2*self.n_h_))
            koef = 1/(1 + rho**2)**(0.5)
            W_sr[:self.n_l_, :self.n_l_] = np.diag(np.sqrt(np.diag(W[:self.n_l_, :self.n_l_])))
            tmp_H = np.diag(np.sqrt(np.diag(W[slice_d, slice_d])))
            W_sr[slice_flh, slice_flh] = koef * rho**2 * tmp_H
            W_sr[slice_flh, slice_d] = koef * rho * tmp_H
            W_sr[slice_d, slice_flh] = koef * rho * tmp_H
            W_sr[slice_d, slice_d] = koef * tmp_H
            
#             global_logger.log_variable(W, 'W')
            #print(np.isnan(W).any())
            #print(rho)
            #print(np.isnan(W_sr).any())
            W_sr_K = W_sr.dot(K)
            B = np.eye(W.shape[0]) + W_sr_K.dot(W_sr) # todo: replace wih sparse matrices? to speed up multiplication
            L = cholesky(B, lower=True)
            # Line 6
            #-----------original--------------
            #b = W * f + (self.y_train_ - pi)
            #-----------general------------
            tmp_f = self._d_log_likelihood(f_combined)
            nabla_ll_f = np.zeros(self.n_l_ + 2*self.n_h_)
            nabla_ll_f[:self.n_l_] = tmp_f[:self.n_l_]
            nabla_ll_f[self.n_l_:self.n_l_ + self.n_h_] = rho * tmp_f[self.n_l_:]
            nabla_ll_f[self.n_l_ + self.n_h_:] = tmp_f[self.n_l_:]
            b = W.dot(f) + nabla_ll_f
            #--------------------------
            # Line 7
            a = b - W_sr.dot(cho_solve((L, True), W_sr_K.dot(b)))
            # Line 8
            f = K.dot(a)

            # Line 10: Compute log marginal likelihood in loop and use as
            #          convergence criterion
            #-----------original--------------
            #lml = -0.5 * a.T.dot(f) \
            #    - np.log(1 + np.exp(-(self.y_train_ * 2 - 1) * f)).sum() \
            #    - np.log(np.diag(L)).sum()
            #-----------general------------
            lml = -0.5 * a.T.dot(f) \
                + self._log_likelihood(f_combined).sum() \
                - np.log(np.diag(L)).sum()
            #--------------------------
            # Check if we have converged (log marginal likelihood does
            # not decrease)
            # XXX: more complex convergence criterion
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml

        self.f_cached = f  # Remember solution for later warm-starts
        if return_temporaries:
            return log_marginal_likelihood, (pi, W_sr, L, b, a, nabla_ll_f)
        else:
            return log_marginal_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds, eval_grad = False):
        if self.optimizer == "fmin_l_bfgs_b":
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, approx_grad = 1 - int(eval_grad))
            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict)
        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min


class MultiFidelityGaussianProcessClassifier(BaseEstimator, ClassifierMixin):
    """Gaussian process classification (GPC) based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2, and 5.1 of
    Gaussian Processes for Machine Learning (GPML) by Rasmussen and
    Williams.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function. For multi-class classification, several binary one-versus rest
    classifiers are fitted. Note that this class thus does not implement
    a true multi-class Laplace approximation.

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the  signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer=0 implies that one
        run is performed.

    max_iter_predict : int, optional (default: 100)
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.

    warm_start : bool, optional (default: False)
        If warm-starts are enabled, the solution of the last Newton iteration
        on the Laplace approximation of the posterior mode is used as
        initialization for the next call of _posterior_mode(). This can speed
        up convergence when _posterior_mode is called several times on similar
        problems as in hyperparameter optimization.

    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    multi_class : string, default : "one_vs_rest"
        Specifies how multi-class classification problems are handled.
        Supported are "one_vs_rest" and "one_vs_one". In "one_vs_rest",
        one binary Gaussian process classifier is fitted for each class, which
        is trained to separate this class from the rest. In "one_vs_one", one
        binary Gaussian process classifier is fitted for each pair of classes,
        which is trained to separate these two classes. The predictions of
        these binary predictors are combined into multi-class predictions.
        Note that "one_vs_one" does not support predicting probability
        estimates.

    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    Attributes
    ----------
    kernel_ : kernel object
        The kernel used for prediction. In case of binary classification,
        the structure of the kernel is the same as the one passed as parameter
        but with optimized hyperparameters. In case of multi-class
        classification, a CompoundKernel is returned which consists of the
        different kernels used in the one-versus-rest classifiers.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``

    classes_ : array-like, shape = (n_classes,)
        Unique class labels.

    n_classes_ : int
        The number of classes in the training data

    .. versionadded:: 0.18
    """
    def __init__(self, kernel=None, optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0, max_iter_predict=100,
                 warm_start=False, copy_X_train=True, random_state=None,
                 multi_class="one_vs_rest", n_jobs=1, rho = 0, rho_bounds = (-1, 1), eval_gradient = False):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.rho = rho
        self.rho_bounds = rho_bounds
        self.eval_gradient = eval_gradient

    def fit(self, X_l, y_l, X_h, y_h):
        """Fit Gaussian process classification model

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples,)
            Target values, must be binary

        Returns
        -------
        self : returns an instance of self.
        """
        X_l, y_l = check_X_y(X_l, y_l, multi_output=False)
        X_h, y_h = check_X_y(X_h, y_h, multi_output=False)

        self.base_estimator_ = _BinaryMultiFidelityGaussianProcessClassifierLaplace(
            self.kernel, self.optimizer, self.n_restarts_optimizer,
            self.max_iter_predict, self.warm_start, self.copy_X_train,
            self.random_state, self.rho, self.rho_bounds, self.eval_gradient)
        
        X_with_fidelity = np.hstack((np.vstack((X_l, X_h)), np.hstack((np.zeros(len(X_l)), np.ones(len(X_h)))).reshape(-1, 1)))
        y = np.hstack((y_l, y_h))

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        if self.n_classes_ == 1:
            raise ValueError("GaussianProcessClassifier requires 2 or more "
                             "distinct classes. Only class %s present."
                             % self.classes_[0])
        if self.n_classes_ > 2:
            if self.multi_class == "one_vs_rest":
                self.base_estimator_ = \
                    OneVsRestClassifier(self.base_estimator_,
                                        n_jobs=self.n_jobs)
            elif self.multi_class == "one_vs_one":
                self.base_estimator_ = \
                    OneVsOneClassifier(self.base_estimator_,
                                       n_jobs=self.n_jobs)
            else:
                raise ValueError("Unknown multi-class mode %s"
                                 % self.multi_class)

        self.base_estimator_.fit(X_with_fidelity, y)

        if self.n_classes_ > 2:
            self.log_marginal_likelihood_value_ = np.mean(
                [estimator.log_marginal_likelihood()
                 for estimator in self.base_estimator_.estimators_])
        else:
            self.log_marginal_likelihood_value_ = \
                self.base_estimator_.log_marginal_likelihood()

        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array, shape = (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """
        check_is_fitted(self, ["classes_", "n_classes_"])
        X = check_array(X)
        return self.base_estimator_.predict(X)

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array-like, shape = (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """
        check_is_fitted(self, ["classes_", "n_classes_"])
        if self.n_classes_ > 2 and self.multi_class == "one_vs_one":
            raise ValueError("one_vs_one multi-class mode does not support "
                             "predicting probability estimates. Use "
                             "one_vs_rest mode instead.")
        X = check_array(X)
        return self.base_estimator_.predict_proba(X)

    @property
    def kernel_(self):
        if self.n_classes_ == 2:
            return self.base_estimator_.kernel_l_, self.base_estimator_.kernel_d_
        else:
            return CompoundKernel(
                [estimator.kernel_l_
                 for estimator in self.base_estimator_.estimators_]), \
                   CompoundKernel(
                [estimator.kernel_d_
                 for estimator in self.base_estimator_.estimators_])

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.

        In the case of multi-class classification, the mean log-marginal
        likelihood of the one-versus-rest classifiers are returned.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or none
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. In the case of multi-class classification, theta may
            be the  hyperparameters of the compound kernel or of an individual
            kernel. In the latter case, all individual kernel get assigned the
            same theta values. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. Note that gradient computation is not supported
            for non-binary classification. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        check_is_fitted(self, ["classes_", "n_classes_"])

        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        theta = np.asarray(theta)
        if self.n_classes_ == 2:
            return self.base_estimator_.log_marginal_likelihood(
                theta, eval_gradient)
        else:
            if eval_gradient:
                raise NotImplementedError(
                    "Gradient of log-marginal-likelihood not implemented for "
                    "multi-class GPC.")
            estimators = self.base_estimator_.estimators_
            n_dims = estimators[0].kernel_.n_dims * 2 # fmsnew: multiplied by 2 since two kernels are estimated: k_l and k_d
            if theta.shape[0] == n_dims:  # use same theta for all sub-kernels
                return np.mean(
                    [estimator.log_marginal_likelihood(theta)
                     for i, estimator in enumerate(estimators)])
            elif theta.shape[0] == n_dims * self.classes_.shape[0]:
                # theta for compound kernel
                return np.mean(
                    [estimator.log_marginal_likelihood(
                        theta[n_dims * i:n_dims * (i + 1)])
                     for i, estimator in enumerate(estimators)])
            else:
                raise ValueError("Shape of theta must be either %d or %d. "
                                 "Obtained theta with shape %d."
                                 % (n_dims, n_dims * self.classes_.shape[0],
                                    theta.shape[0]))