import numpy as np
from tqdm.auto import tqdm
import pygad
from matplotlib import pyplot as plt

import math
import torch
from torch.distributions import Normal, Independent
from sklearn.utils.extmath import fast_logdet

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

        torch.manual_seed(0)
        tt, cov = self.forward_function(design, self.model_samples)

        tt = torch.tensor(tt, dtype=torch.float32)
        cov = torch.tensor(cov, dtype=torch.float32)

        N, M = tt.shape[0], tt.shape[0]
        # M = int(math.sqrt(N))
                
        N_data_likelihoods = Independent(Normal(tt, cov ** (1/2)), 1)
        M_data_likelihoods = N_data_likelihoods.expand((M, N))
        
        N_data_samples = N_data_likelihoods.sample()
        M_data_samples = N_data_samples

        evidence_term = M_data_likelihoods.log_prob(
                        M_data_samples.unsqueeze(0).swapaxes(0, 1)
                        ).swapaxes(0, 1).logsumexp(0) - math.log(M)
        
        likelihood_term = N_data_likelihoods.log_prob(N_data_samples)

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

        torch.manual_seed(0)
        tt, cov = self.forward_function(design, self.model_samples)
        
        tt = torch.tensor(tt, dtype=torch.float32)
        cov = torch.tensor(cov, dtype=torch.float32)
                
        N_data_likelihoods = Independent(Normal(tt, cov ** (1/2)), 1)
        tt_noise = N_data_likelihoods.sample()
        likelihood_term = N_data_likelihoods.log_prob(tt_noise)
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
                assert all(st in ['tt', 'asl', 'array'] for st in station)
        else:
            raise ValueError("available_stations must be either a tuple or a dict")

        if optimisation_algorithm == 'genetic':
            best_design, info = self._genetic_optimisation(
                available_stations,
                optimisation_kwargs,
            )
        elif optimisation_algorithm == 'sequential':
            raise NotImplementedError
        elif optimisation_algorithm == 'random':
            raise NotImplementedError
        else:
            raise ValueError("optimisation_algorithm must be one of ['genetic', 'sequential', 'random']")
        
        return best_design, info
            
    def _genetic_optimisation(
        self,
        available_stations,
        optimisation_kwargs,
    ):
        
        optimisation_kwargs.setdefault('num_generations', 100)
        optimisation_kwargs.setdefault('num_parents_mating', 4)
        optimisation_kwargs.setdefault('mutation_percent_genes', 'default')
        optimisation_kwargs.setdefault('sol_per_pop', 64)
        optimisation_kwargs.setdefault('gene_type', int)
        optimisation_kwargs.setdefault('allow_duplicate_genes', False)
        optimisation_kwargs.setdefault('mutation_type', 'random')
        optimisation_kwargs.setdefault('suppress_warnings', True)
        optimisation_kwargs.setdefault('random_seed', 1)
        
        optimisation_kwargs['num_genes'] = len(available_stations)
        
        plot_fitness = optimisation_kwargs.pop('plot_fitness', False)
        progress_bar = optimisation_kwargs.pop('progress_bar', True)
                        
        gene_space = []
        for station in available_stations:
            if 'array' in station:
                gene_space.append(np.arange(len(self.design_points_dict['array']),  dtype=int).tolist())
            else:
                gene_space.append(np.arange(len(self.design_points_dict['node']),  dtype=int).tolist())

        def fitness_function(ga_instance, solution, solution_idx):
            
            design_with_type = []
            for i, st in enumerate(available_stations):
                if 'array' in st:
                    st_coords = self.design_points_dict['array'][solution[i]]
                    design_with_type += [(st, st_coords)]
                else:
                    st_coords = self.design_points_dict['node'][solution[i]]
                    design_with_type += [(st, st_coords)]

            if self.preexisting_design is not None:
                design_with_type = self.preexisting_design + design_with_type

            eig = self.design_criterion(design_with_type)
            
            return eig            
            
        with tqdm(
            total=optimisation_kwargs['num_generations'],
            desc='GA progress',
            postfix={'DN criterion': 0.0},
            disable=not progress_bar,
            ) as pbar:
            
            def on_generation(ga_instance):
                pbar.update(1)
                pbar.set_postfix(
                    {'DN criterion': ga_instance.last_generation_fitness.max(),})
                
            ga_instance = pygad.GA(
                fitness_func=fitness_function,
                gene_space=gene_space,
                on_generation=on_generation,
                **optimisation_kwargs,
            )
            
            ga_instance.run()
            
        
        best_design_idx = ga_instance.best_solution()[0]
        best_design_EIG = ga_instance.last_generation_fitness.max()

        best_design = []
        for i, st in enumerate(available_stations):
            if 'array' in st:
                best_design.append(
                    (st, self.design_points_dict['array'][best_design_idx[i]]))
            else:
                best_design.append(
                    (st, self.design_points_dict['node'][best_design_idx[i]]))

        if self.preexisting_design is not None:
            best_design = self.preexisting_design + best_design

        if plot_fitness:
            fitness = np.array(ga_instance.best_solutions_fitness)
            
            fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
            ax.plot(fitness, color='black', linewidth=2)
            ax.set_xlabel('Generation')
            ax.set_ylabel('DN criterion')
            
            ax.set_title('Fitness over generations')
            
            plt.show()
            
        out_info = dict(
            best_design_EIG=best_design_EIG,
            ga_instance=ga_instance,
        )
                
        return best_design, out_info