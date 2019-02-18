import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pyDOE
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.gaussian_process import kernels
from scipy.interpolate import griddata
from sklearn.metrics.pairwise import pairwise_distances
from scipy.interpolate import Rbf
from scipy.stats import expon
from scipy.stats import uniform
from scipy.stats import norm
import scipy_benchmarks

from sklearn.gaussian_process import GaussianProcessRegressor

def softmax(v):
    exp_v = np.exp(v)
    return exp_v/np.sum(exp_v)

def random_int_with_probability(probabilities):
    choice = 0
    distribution = np.cumsum(probabilities)
    threshold = np.random.rand()
    for i in range(len(distribution)):
        if distribution[i] < threshold:
            choice += 1
    return choice

class ModelPoolHandler:
    def __init__(self, eng):
        self.eng = eng
        self.eng.gpInitModelPool(nargout = 0)
        self.models_count = 0
        pass
    
    def trainGPModel(self, points, values, model_id = None):
        if model_id is None:
            model_id = self.models_count + 1
            self.models_count += 1
        self.eng.gpTrainWrapper(matlab.double(points.tolist()), matlab.double(values.T.tolist()))
        self.eng.gpPoolSetModel(model_id)
        return model_id
    
    def incrementalTrainGPModel(self, points, values, model_id_old, model_id = None):
        if model_id is None:
            model_id = self.models_count + 1
            self.models_count += 1
        self.eng.gpIncrementalTrainWrapper(model_id_old, matlab.double(points.tolist()), matlab.double(values.T.tolist()))
        self.eng.gpPoolSetModel(model_id)
        return model_id
    
    def simGPModel(self, points, model_id):
        points = [p.tolist() for p in points]
        estimation = np.array(self.eng.gpSimWrapper(model_id, matlab.double(points)))
        return estimation[:, 0], estimation[:, 1]
    
    def getModelParameters(self, model_id):
        return eng.gpModelParams(model_id)
    
class RandomFunctionGenerator:
    
    def __init__(self, 
                 domain_dimension = 1, 
                 function_type = 'simple',
                 **kwargs):
        self.domain_dimension = domain_dimension
        self.function_type = function_type
        self.params = kwargs
        if 'sampling_points_count' not in self.params:
            self.sampling_points_count = 50
        else:
            self.sampling_points_count = self.params['sampling_points_count']
        if 'multiply_sampling_points_count_by_dimension' in self.params:
            if self.params['multiply_sampling_points_count_by_dimension']:
                self.sampling_points_count *= self.domain_dimension
        

    
    def __str__(self):
        return 'RandomFunctionGenerator(domain_dimension=' + str(self.domain_dimension) + \
               ', function_type=' + repr(self.function_type) + \
               ', **' + str(self.params) + ')'
    
    def __repr__(self):    
        return self.__str__()
    
    def _generate_simple_sample(self):
        
        # L - Lipschitz constant
        if 'L' not in self.params:
            L = 3
        else:
            L = self.params['L']
        
        # sample the first point & value
        points = np.random.rand(1, self.domain_dimension)
        values = uniform.rvs(size = 1)
        # iteratively add new points & values if they satisfy Lipschitz continuity 
        for i in range(1, self.sampling_points_count):
            added = False
            fail_counts = 0
            while not added:
                new_point = np.random.rand(self.domain_dimension)
                new_value = uniform.rvs(size = 1)
                dF = pairwise_distances(np.hstack((values, new_value)).reshape(-1, 1))
                dX = pairwise_distances(np.vstack((points, new_point)))
                if np.all(L * dX - dF >= 0):
                    values = np.hstack((values, new_value))
                    points = np.vstack((points, new_point))
                    added = True
                else:
                    fail_counts += 1
                    if fail_counts > 1000:
                        # if we have been unlucky to add a new random point without violiting Lipschitz continuity for a lot of attempts
                        break    
        return points, values
        
        
    def _generate_gaussian_process_sample(self):
        if 'length_scale' in self.params:
            if isinstance(self.params['length_scale'], tuple):
                min_ls = self.params['length_scale'][0]
                max_ls = self.params['length_scale'][1]
                ls = min_ls + (max_ls - min_ls)*np.random.rand()
            else:
                ls = self.params['length_scale']
        else:
            ls = 1.0
        if 'kernel' in self.params:
            if self.params['kernel'] == 'RBF':
                kernel = kernels.RBF(length_scale = ls)
            elif self.params['kernel'] == 'Matern':
                kernel = kernels.Matern(length_scale = ls)
            else:
                raise Exception('unknown kernel')
        else:
            kernel = kernels.RBF(length_scale = ls)
        gpr = GaussianProcessRegressor(kernel=kernel)
        X = np.zeros((1, self.domain_dimension))
        y = np.zeros(1)
        gpr.fit(X, y)
        points = np.random.rand(self.sampling_points_count, self.domain_dimension)
        values = gpr.sample_y(points, random_state = np.random.randint(100000))
        return points, values
    
    def _generate_custom_sample(self):
        if 'alpha' in self.params:
            if isinstance(self.params['alpha'], tuple):
                min_alpha = self.params['alpha'][0]
                max_alpha = self.params['alpha'][1]
                alpha = min_alpha + (max_alpha - min_alpha)*np.random.rand()
            else:
                alpha = self.params['alpha']
        else:
            alpha = 0.5
        rfg_base = RandomFunctionGenerator(self.domain_dimension, 'gaussian_process',  
                      **{'length_scale':0.1, 'kernel':'Matern', 'smooth':0.5,
                      'sampling_points_count':50})
        rfg_base_function = rfg_base.generate_function()
        rfg_extra = RandomFunctionGenerator(self.domain_dimension, 'gaussian_process', 
                      **{'length_scale':(0.005, 0.05), 'kernel':'Matern', 'smooth':(0.01, 0.05),
                      'sampling_points_count':50, 
                       'multiply_sampling_points_count_by_dimension':True})
        rfg_extra_function = rfg_extra.generate_function()
        points = np.random.rand(self.sampling_points_count, self.domain_dimension)
        values = (1 - alpha) * rfg_base_function(points) + alpha * rfg_extra_function(points)
        return points, values
    
    def generate_function(self, seed = None):
        if seed is not None:
            np.random.seed(seed)
        if 'min_value_upper_bound' not in self.params:
            self.min_value_upper_bound = 0.0
        else:
            self.min_value_upper_bound = np.random.rand()*self.params['min_value_upper_bound']
        if 'max_value_lower_bound' not in self.params:
            self.max_value_lower_bound = 1.0
        else:
            self.max_value_lower_bound = 1 - np.random.rand()*(1 - self.params['max_value_lower_bound'])
        
        if self.function_type == 'simple':
            points, values = self._generate_simple_sample()
        if self.function_type == 'gaussian_process':
            points, values = self._generate_gaussian_process_sample()
        if self.function_type == 'custom':
            points, values = self._generate_custom_sample()
        values = (values - np.min(values))/(np.max(values) - np.min(values))
        values = values*(self.max_value_lower_bound - self.min_value_upper_bound) + self.min_value_upper_bound
        
        if 'smooth' not in self.params:
            sm = 0.5
        else:
            if isinstance(self.params['smooth'], tuple):
                min_sm = self.params['smooth'][0]
                max_sm = self.params['smooth'][1]
                sm = min_sm + (max_sm - min_sm)*np.random.rand()
            else:
                sm = self.params['smooth']
        
        if self.domain_dimension == 1:
            rbf = Rbf(points.ravel(), values, function = 'multiquadric', smooth = sm)
            return lambda x: rbf(x.ravel())
        else:
            targs= tuple(points.T) + (values,)
            rbf = Rbf(*targs, function = 'multiquadric', smooth = sm)
            return lambda x: rbf(*tuple(x.T))
        
def plot_random_functions(params = {}, n = 5):
   
    # plot 1d samples
    params['domain_dimension'] = 1
    rfg = RandomFunctionGenerator(**params)
    plt.figure(figsize = (10, 5))
    plt.suptitle('1D random functions', fontsize = 14)
    gs = gridspec.GridSpec(n, 2*n)
    for it in range(n*n):
        ax = plt.subplot(gs[it/n, it % n])
        ax.set_ylim([0, 1])
        f = rfg.generate_function()
        grid = f(np.linspace(0, 1, 300))
        if it % n == n-1:
            ax.plot(np.linspace(0, 1, 300), grid, color = cm.Set1(4/9.))
            ax.set_xticks([])
            ax.set_yticks([])
            ax = plt.subplot(gs[it/n, n:])
            ax.plot(np.linspace(0, 1, 300), grid, color = cm.Set1(4/9.))
            ax.set_ylim([0, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            if it/n == 0:
                ax.set_title('stretched version of the previous column')
        else:
            ax.plot(np.linspace(0, 1, 300), grid)
            ax.set_ylim([0, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            
    # plot 2d samples
    z = (np.linspace(0, 1, 150), )
    meshgrid = np.meshgrid(*z)
    meshgrid[0] = meshgrid[0].ravel()
    grid_points_slice = np.vstack(tuple(meshgrid)).T
    z += (np.linspace(0, 1, 150), )
    meshgrid = np.meshgrid(*z)
    for d in range(2):
        meshgrid[d] = meshgrid[d].ravel()
    grid_points = np.vstack(tuple(meshgrid)).T
    plt.figure(figsize = (10, 5))
    plt.suptitle('2D random functions', fontsize = 14)
    params['domain_dimension'] = 2
    rfg = RandomFunctionGenerator(**params)
    for it in range(n*n):
        ax = plt.subplot(gs[it/n,it%n])
        f = rfg.generate_function()
        grid = f(grid_points)
        ax.imshow(grid.reshape(150, 150), cmap=plt.cm.gray, interpolation='none', extent=[0,1,0,1], origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        if it % n == n-1:
            for t in range(n):
                ax.hlines((t+0.5)/(n), 0, 1, color = plt.cm.jet((t+0.5)/n), linewidth = 1.0, alpha = 0.3)    
            for t in range(n):
                ax = plt.subplot(gs[it/n, 2*n - t - 1])
                if it/n == 0 and t == n/2:
                    ax.set_title('1D slices of the last 2D column')
                grid = f(np.hstack((grid_points_slice, (t+0.5)/(n) + np.zeros(len(grid_points_slice)).reshape(-1, 1))))
                ax.plot(grid.ravel(), color = plt.cm.jet((t+0.5)/(n)))
                ax.set_ylim([0, 1])
                ax.set_xticks([])
                ax.set_yticks([])     
                
    # plot 2d slices of 3d samples
    gs = gridspec.GridSpec(n, 2*n)
    plt.figure(figsize = (10, 5))
    plt.suptitle('2D slices of 3D random functions', fontsize = 14)
    params['domain_dimension'] = 3
    rfg = RandomFunctionGenerator(**params)
    z_colormaps = []
    cmap_name = 'z_colormap'
    for t in range(n):
        z_colors = [(1, 1, 1), plt.cm.jet((t+0.5)/(n))[:3], (0, 0, 0)]
        z_colormap = LinearSegmentedColormap.from_list(cmap_name, z_colors)
        z_colormaps.append(z_colormap)
    for it in range(2*n):
        for t in range(n):
            if it < n:
                ax = plt.subplot(gs[it,t])
            else:
                ax = plt.subplot(gs[it - n,n + t])
            f = rfg.generate_function()
            grid = f(np.hstack((grid_points, t/n + np.zeros(len(grid_points)).reshape(-1, 1))))
            ax.imshow(grid.reshape(150, 150), cmap=z_colormaps[t], interpolation='none', extent=[0,1,0,1], origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])    
            
            
class BenachmarkFunctionCollection:
    
    def __init__(self):
        self.benchmarks_names = filter(lambda x: x[0].isupper() and x != 'Benchmark' and \
                                       not np.isnan(getattr(scipy_benchmarks, x)().fglob), scipy_benchmarks.__all__)
        
        
            
    def __find_max(self, f):
        sample_points = pyDOE.lhs(f.N, samples=100000)
        return np.max(map(f.fun, self.__stretch(sample_points, f.bounds)))

    def __str__(self):
        return 'BenachmarkFunctionGenerator'
    
    def __repr__(self):    
        return self.__str__()
    
    def __stretch(self, x, bounds):
        sx = np.array(x)
        for d in range(len(bounds)):
            sx[:, d] = x[:, d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
        return sx

    def __allign(self, y, original_bounds, alligned_bounds):
        return ((y - original_bounds[0]) / (original_bounds[1] - original_bounds[0])) * \
                (alligned_bounds[1] - alligned_bounds[0]) + alligned_bounds[0]
    
    def generate_function(self, ind):
        name = self.benchmarks_names[ind]
        cur_f = getattr(scipy_benchmarks, name)()
        self.DIMENSION = cur_f.N
        a_min = np.random.rand() * 0.3
        a_max = 1 - np.random.rand() * 0.3
        self.MAX = a_max
        fglob_min = -self.__find_max(cur_f)
        fglob_max = -cur_f.fglob
        return \
                lambda x: self.__allign(-np.array(map(cur_f.fun, 
                                                      self.__stretch(x, cur_f.bounds))),                                              (fglob_min, fglob_max),
                                         (a_min, a_max))
                
def _find_max(f):
    sample_points = pyDOE.lhs(f.N, samples=100000)
    return np.max(map(f.fun, _stretch(sample_points, f.bounds)))


def _stretch(x, bounds):
    sx = np.array(x)
    for d in range(len(bounds)):
        sx[:, d] = x[:, d] * (bounds[d][1] - bounds[d][0]) + bounds[d][0]
    return sx

def _allign(y, original_bounds, alligned_bounds):
    return ((y - original_bounds[0]) / (original_bounds[1] - original_bounds[0])) * \
            (alligned_bounds[1] - alligned_bounds[0]) + alligned_bounds[0]
    
    
def ProbabilityOfImprovementPrecomputed(estimated_values, estimated_errors, model_params, eps = None):
    estimated_values = estimated_values.ravel()
    estimated_errors = estimated_errors.ravel()
    eps = (1.0 / model_params['trainSize'])**2
    cur_f_max = np.max(np.array(model_params['restoredTrainValues']))
    delta = estimated_values - cur_f_max - eps
    non_zero_error_inds = np.where(estimated_errors > 1e-6)[0]
    zero_error_inds = np.where(estimated_errors <= 1e-6)[0]
    Z = np.zeros(len(delta))
    ranked_cdf = np.zeros(len(delta))
    if len(non_zero_error_inds) > 0:
        Z[non_zero_error_inds] = delta[non_zero_error_inds]/estimated_errors[non_zero_error_inds]
        Zb = np.zeros(len(delta))
        b = np.min(Z[non_zero_error_inds])
        Zb[non_zero_error_inds] = Z[non_zero_error_inds] - b
        sorted_Zb_inds = np.argsort(Zb)
        ranked_cdf[sorted_Zb_inds] = np.arange(len(delta))/float(len(delta))
    ranked_cdf[zero_error_inds] = 0
    return ranked_cdf


def ExpectedImprovementPrecomputed(estimated_values, estimated_errors, model_params, eps = None):
    estimated_values = estimated_values.ravel()
    estimated_errors = estimated_errors.ravel()
    eps = 0.05/model_params['trainSize']
    restored_train_values = np.array(model_params['restoredTrainValues'])
    cur_f_max = np.max(restored_train_values)
    delta = estimated_values - cur_f_max - eps
    non_zero_error_inds = np.where(estimated_errors > 1e-6)[0]
    Z = np.zeros(len(delta))
    Z[non_zero_error_inds] = delta[non_zero_error_inds]/estimated_errors[non_zero_error_inds]
    

    # in log-scale
    log_EI = np.log(estimated_errors) + norm.logpdf(Z) + np.log(1 + Z * np.exp(norm.logcdf(Z) - norm.logpdf(Z)))
    ranked_EI = np.zeros(len(delta))
    ranked_EI[np.argsort(log_EI)] = np.arange(len(delta))/float(len(delta))
    return ranked_EI


def UpperConfidenceBoundPrecomputed(estimated_values, estimated_errors, model_params, beta = None):
    estimated_values = estimated_values.ravel()
    estimated_errors = estimated_errors.ravel()
    delta = 0.1
    
    # The next formula was taken from 'A Tutorial on Bayesian Optimization of Expensive Cost Functions', Brochu et al., 2010
    # caution: it is probably not precise or even wrong from the theoretical point of view:
    # see 'Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design', Srinivas et al., 2010
    d = model_params['dimension']
    t = model_params['trainSize']
    tau = 2 * np.log(t ** (d * 0.5 + 2) * (np.pi ** 2) / (3 * delta))
    if beta is None:
        beta = np.sqrt(tau)
    UCB = estimated_values + beta * estimated_errors
    ranked_UCB = np.zeros(len(UCB))
    ranked_UCB[np.argsort(UCB)] = np.arange(len(UCB))/float(len(UCB))
    return ranked_UCB


def MaximumUncertainty(estimated_values, estimated_errors, model_params):
    estimated_values = estimated_values.ravel()
    estimated_errors = estimated_errors.ravel()
    MU = estimated_errors
    ranked_MU = np.zeros(len(MU))
    ranked_MU[np.argsort(MU)] = np.arange(len(MU))/float(len(MU))
    return ranked_MU


def PureRandom(estimated_values, estimated_errors, model_params):
    estimated_values = estimated_values.ravel()
    estimated_errors = estimated_errors.ravel()
    PR = np.random.rand(len(estimated_errors))
    ranked_PR = np.zeros(len(PR))
    ranked_PR[np.argsort(PR)] = np.arange(len(PR))/float(len(PR))
    return ranked_PR


