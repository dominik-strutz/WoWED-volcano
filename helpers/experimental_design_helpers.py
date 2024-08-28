import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import math
from sklearn.utils.extmath import fast_logdet
from scipy.special import logsumexp

def add_gradient(surface_data):
    '''
    Helper function that adds gradient data to the topography data. Necessary to correctly array orientation.
    
    Parameters
    ----------
    surface_data : xr.Dataset
        The topography data.
        
    Returns
    -------
    xr.Dataset
        The topography data with gradient data.
    '''
    
    dE = ((surface_data['E'].max() - surface_data['E'].min()) / (surface_data['E'].size - 1)).values
    dN = ((surface_data['N'].max() - surface_data['N'].min()) / (surface_data['N'].size - 1)).values
    
    grad_E, grad_N = np.gradient(surface_data['topography'], dE, dN)
    surface_data['grad_E'] = (('E', 'N'), grad_E)
    surface_data['grad_N'] = (('E', 'N'), grad_N)
    
    return surface_data

def get_prior_samples(prior_data, n_samples):
    
    dE = prior_data['E'].values[1] - prior_data['E'].values[0]
    dN = prior_data['N'].values[1] - prior_data['N'].values[0]
    dZ = prior_data['Z'].values[1] - prior_data['Z'].values[0]
    
    source_location_mask = prior_data.values > 0.0
    source_locations = np.meshgrid(
        prior_data['E'].values,
        prior_data['N'].values,
        prior_data['Z'].values,
        indexing='ij'
    )
    source_locations = np.stack(source_locations, axis=-1)[source_location_mask]
        
    np.random.seed(0)
    model_samples = source_locations[
        np.random.choice(
            len(source_locations),
            n_samples,
            p=prior_data.values[source_location_mask].flatten()
            )]

    # add uniform noise to vary source locations within their grid cell
    model_samples += np.random.uniform(
        -0.5, 0.5, model_samples.shape) * np.array(
            [dE, dN, dZ])
        
    return model_samples

class NMC_method:
    def __init__(self, forward_function, prior_data, n_model_samples=1000):
    
        self.forward_function = forward_function
        
        self.model_samples    = self._construct_model_samples(
            prior_data, n_model_samples
        )

    def __call__(self, design):

        np.random.seed(0)
        tt, cov = self.forward_function(design, self.model_samples)

        N, M = tt.shape[0], tt.shape[0]
                
        tt_noise = tt + np.random.normal(0, 1, tt.shape) * np.sqrt(cov)
        likelihood_term = -0.5 * np.sum(np.log(2 * np.pi * cov) + (tt_noise - tt)**2 / cov, axis=-1)
        
        # evidence_term = -0.5 * np.sum(np.log(2 * np.pi * cov) + (
        #     tt_noise.reshape((N, 1, -1)) - tt.reshape(1, N, -1))**2 / cov, axis=-1)
        
        # evidence_term = logsumexp(evidence_term, axis=1) - np.log(M)
        log_scale = np.log(np.sqrt(cov.reshape(N, 1, -1)))   
        evidence_term = logsumexp(np.sum(
                -((tt_noise.reshape(N, 1, -1) - tt.reshape(1, N, -1)) ** 2) / (2 * cov.reshape(N, 1, -1))
                - log_scale
                - np.log(np.sqrt(2 * np.pi)), axis=-1),
                axis=-1,
            ) - np.log(M)
            
        print(likelihood_term.sum(0) / N)
        print(evidence_term.sum(0) / N)

        eig = (likelihood_term - evidence_term).sum(0) / N

        return eig.item()
    
    @staticmethod
    def _construct_model_samples(prior_data, n_model_samples):
        
        return get_prior_samples(prior_data, n_model_samples)



class DN_method:
    def __init__(self, forward_function, prior_data, n_model_samples=1000):
    
        self.forward_function = forward_function
        
        self.model_samples    = self._construct_model_samples(
            prior_data, n_model_samples
        )

    def __call__(self, design):

        np.random.seed(0)
        tt, cov = self.forward_function(design, self.model_samples)
        
        tt_noise = tt + np.random.normal(0, 1, tt.shape) * np.sqrt(cov)
        likelihood_term = -0.5 * np.sum(np.log(2 * np.pi * cov) + (tt_noise - tt)**2 / cov, axis=-1)
        likelihood_term = likelihood_term.sum(0) / tt.shape[0]
        
        if tt_noise.shape[-2] >= 2:
            evidence_covariance = np.cov(tt_noise.T)
        else:
            evidence_covariance = np.cov(tt_noise.T)[None, None]
                    
        evidence_cov = fast_logdet(evidence_covariance)
        
        k = evidence_covariance.shape[-1]
        evidence_term = -(k/2 + k/2 * np.log(2 * np.pi) + 0.5 * evidence_cov)
                                
        eig = likelihood_term - evidence_term

        return eig.item()
    
    @staticmethod
    def _construct_model_samples(prior_data, n_model_samples):
        
        return get_prior_samples(prior_data, n_model_samples)
    

class Design_Optimisation:
    def __init__(
        self,
        design_criterion,
        surface_data,
        prior_data,
        design_space_dict,
        preexisting_design=None):
        
        self.design_criterion = design_criterion
        self.surface_data = surface_data
        self.prior_data = prior_data
        
        self.surface_data = add_gradient(self.surface_data)

        # convert all tuples to lists
        if preexisting_design is not None:
            self.preexisting_design = [[d[0], np.array(d[1])] for d in preexisting_design]
                        
            # if 'array' in d[0] add normal vector
            for i, d in enumerate(self.preexisting_design):
                if 'array' in d[0]:
                    st_coords = d[1]
                                        
                    gradient = (surface_data['grad_E'].interp(E=st_coords[0], N=st_coords[1]).values.item(),
                                    surface_data['grad_N'].interp(E=st_coords[0], N=st_coords[1]).values.item())        
                    normal_vector = np.array([-gradient[0], -gradient[1], 1])
                    normal_vector /= np.linalg.norm(normal_vector)
                    st_coords = np.hstack((st_coords, normal_vector))
        
                    self.preexisting_design[i][1] = st_coords
        else:
            self.preexisting_design = None
    
        design_points_dict = {}
        design_points_mask_dict = {}
        
        for receiver_type in ['node', 'array']:
            design_points, design_points_mask = self._construct_design_points(
                design_space_dict[receiver_type], self.surface_data, receiver_type)
            design_points_dict[receiver_type] = design_points
            design_points_mask_dict[receiver_type] = design_points_mask
        
        self.design_points_dict = design_points_dict
        self.design_points_mask_dict = design_points_mask_dict
    
    def _construct_design_points(
        self, design_space, surface_data, receiver_type='node'):
        
        design_points_hor = np.stack(
        np.meshgrid(design_space.E,
                    design_space.N,
                    indexing='ij'),
            axis=-1
        ).reshape(-1, 2)

        design_points_vert = surface_data['topography'].values.flatten()[..., None]
        design_points = np.hstack(
            [design_points_hor, design_points_vert],
        )
        
        design_points_mask = design_space.values.flatten()
        design_points      = design_points[design_points_mask]
        
        if receiver_type == 'array':
            
            dE = surface_data.E.values[1] - surface_data.E.values[0]
            dN = surface_data.N.values[1] - surface_data.N.values[0]
            
            ds_array_grad = np.gradient(
                surface_data['topography'], dE, dN)
            ds_array_normal = np.stack(
                [-ds_array_grad[0][design_points_mask.reshape(*surface_data['topography'].shape)],
                -ds_array_grad[1][design_points_mask.reshape(*surface_data['topography'].shape)],
                np.ones_like(ds_array_grad[0][design_points_mask.reshape(*surface_data['topography'].shape)])],
                axis=-1
            )

            ds_array_normal /= np.linalg.norm(ds_array_normal, axis=-1, keepdims=True)

            # add normal vector of the surface to the design space
            design_points = np.hstack(
                [design_points, ds_array_normal])
        
        return design_points, design_points_mask
    
    def get_optimal_design(
        self,
        available_stations,
        optimisation_algorithm='genetic',
        optimisation_kwargs={},
    ):

        optimisation_kwargs = dict(optimisation_kwargs)   

        if isinstance(available_stations, dict):
            N_node = available_stations.get('n_node', 0)
            N_array = available_stations.get('n_array', 0)
            available_stations = [('tt', 'asl'),] * N_node
            available_stations += [('tt', 'asl', 'array'),] *N_array            
        elif isinstance(available_stations, tuple):
            for station in available_stations:
                assert isinstance(station, tuple)
                # assert all(st in ['tt', 'asl', 'array'] for st in station)
                #TODO: add check for data type
        else:
            raise ValueError("available_stations must be either a tuple or a dict")

        if optimisation_algorithm == 'differential_evolution':
            best_design, info = self._differential_evolution_optimisation(
                available_stations, optimisation_kwargs)
        elif optimisation_algorithm == 'sequential':
            raise NotImplementedError
        elif optimisation_algorithm == 'random':
            raise NotImplementedError
        else:
            raise ValueError("optimisation_algorithm must be one of ['differential_evolution',]")
        
        return best_design, info
            
    def _differential_evolution_optimisation(
        self, available_stations, optimisation_kwargs):
        
        optimisation_kwargs.setdefault('maxiter', 100)
        optimisation_kwargs.setdefault('popsize', 15)
        optimisation_kwargs.setdefault('tol', 1e-3)
        optimisation_kwargs.setdefault('seed', 0)
        
        plot_fitness = optimisation_kwargs.pop('plot_fitness', False)
        progress_bar = optimisation_kwargs.pop('progress_bar', True)
    
        def fitness_function(design, *args):            
            design_with_type = []
            for i, st in enumerate(available_stations):
                if 'array' in st:                   
                    st_coords = self.design_points_dict['array'][int(design[i])]
                    design_with_type += [(st, st_coords)]
                else:
                    st_coords = self.design_points_dict['node'][int(design[i])]
                    design_with_type += [(st, st_coords)]

            if self.preexisting_design is not None:
                design_with_type = self.preexisting_design + design_with_type

            eig = self.design_criterion(design_with_type)
            
            return -eig
        
        from scipy.optimize import differential_evolution
                
        bounds = []
        for st in available_stations:
            if 'array' in st:
                bounds.append((0, len(self.design_points_dict['array'])-1))
            else:
                bounds.append((0, len(self.design_points_dict['node'])-1))
        
        with tqdm(
            total=optimisation_kwargs['maxiter'],
            desc='GA progress',
            postfix={'EIG: ': 0.0},
            disable=not progress_bar,
            ) as pbar:

            EIG_history = []
        
            def callback(xk, convergence):
                pbar.set_postfix({'EIG: ': -fitness_function(xk)})
                pbar.update(1)
                EIG_history.append(-fitness_function(xk))
        
            result = differential_evolution(
                fitness_function,
                bounds=bounds,   
                integrality = [True]*len(available_stations),
                callback=callback,
                **optimisation_kwargs
            )
        
        best_design = []
        for i, st in enumerate(available_stations):
            if 'array' in st:
                st_coords = self.design_points_dict['array'][int(result.x[i])]
                best_design.append((st, st_coords))
            else:
                st_coords = self.design_points_dict['node'][int(result.x[i])]
                best_design.append((st, st_coords))
                
        if plot_fitness:
            fig, ax = plt.subplots()
            ax.plot(EIG_history)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Fitness')
            ax.set_title('Fitness over iterations')
            plt.show()
            
        result.EIG_history = EIG_history
            
        return best_design, result