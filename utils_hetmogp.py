import sys
import climin
from functools import partial
import warnings
import os
sys.path.append('/home/fmsnew/MFGPC/HetMOGP')

import numpy as np
from scipy.stats import multinomial
from scipy.linalg.blas import dtrmm

import GPy
from GPy.util import choleskies
from GPy.core.parameterization.param import Param
from GPy.kern import Coregionalize
from GPy.likelihoods import Likelihood
from GPy.util import linalg

from likelihoods.bernoulli import Bernoulli
from likelihoods.gaussian import Gaussian
from likelihoods.categorical import Categorical
from likelihoods.hetgaussian import HetGaussian
from likelihoods.beta import Beta
from likelihoods.gamma import Gamma
from likelihoods.exponential import Exponential

from hetmogp.util import draw_mini_slices
from hetmogp.het_likelihood import HetLikelihood
from hetmogp.svmogp import SVMOGP
from hetmogp import util
from hetmogp.util import vem_algorithm as VEM

import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from matplotlib import rc, font_manager
from matplotlib import rcParams

from scipy.special import expit
from sklearn.base import BaseEstimator


class HetmogpWrapeper(BaseEstimator):

	def __init__(self, M=20, vem_iters=3):
		self.likelihoods_list = [Bernoulli(), Bernoulli()] # Real + Binary
		self.likelihood = HetLikelihood(self.likelihoods_list)
		self.Y_metadata = self.likelihood.generate_metadata()
		self.D = self.likelihood.num_output_functions(self.Y_metadata)
		self.M = M
		self.vem_iters = vem_iters
		pass

	def fit(self, X_l, y_l, X_h, y_h):
		input_dim = X_h.shape[1]
		X = [X_l, X_h]
		Y = [y_l.reshape(-1, 1), y_h.reshape(-1, 1)]
		Q = 2 # number of latent functions
		ls_q = np.array(([.05]*Q))
		var_q = np.array(([.5]*Q))
		kern_list = util.latent_functions_prior(Q, lenghtscale=ls_q, variance=var_q, input_dim=input_dim)
		#Z = np.linspace(0, 1, self.M)
		#Z = Z[:, np.newaxis]
		Z = np.random.randn(self.M, input_dim)
		self.model = SVMOGP(X=X, Y=Y, Z=Z, kern_list=kern_list, likelihood=self.likelihood, Y_metadata=self.Y_metadata)
		self.model = VEM(self.model, stochastic=False, vem_iters=self.vem_iters, optZ=True, verbose=False, verbose_plot=False, non_chained=False)

	def predict_proba(self, X):
		htmogp_1_m, htmogp_1_v = self.model._raw_predict_f(X, output_function_ind=1)
		pos_preds = expit(htmogp_1_m)
		return np.hstack((1 - pos_preds, pos_preds))

	def predict(self, X):
		proba = self.predict_proba(X)[:, 1]
		return (proba > 0.5).astype(int)