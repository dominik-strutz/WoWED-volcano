import numpy as np
import warnings

def add_gradient(surface_data):
    '''
    Helper function that adds gradient data to the topography data. Necessary to correct array orientation.
    
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

class Forward_Function:
    def __init__(self, data_dict, surface_data=None, data_lookup=None, prior_samples=None):
        '''
        Forward function class that can be called to generate synthetic data and variances for a given model and design.
        
        Parameters
        ----------
        data_dict : dict
            Dictionary with the data types as keys and the information needed to generate the data as values. Currently implemented are 'arrival_*', 'amplitude_*', and 'array', where * can be substituted with the name of the seismic phase.
            For 'arrival_*' the following keys are required:
                - 'v_hom' : float : homogeneous velocity if no data_lookup is provided
                - 'std_vel' : float : standard deviation in the travel time pick due to uncertainty in the velocity model
                - 'std_obs' : float : standard deviation in the travel time pick due to observational uncertainty
            For 'amplitude_*' the following keys are required:
                - 'v_hom' : float : homogeneous velocity if no data_lookup is provided
                - 'std_vel' : float : standard deviation in the travel time pick due to uncertainty in the velocity model
                - 'std_obs' : float : standard deviation in the travel time pick due to observational uncertainty
                - 'f' : float : frequency of interest in Hz
                - 'Q' : float : quality factor of the medium
                - 'std_Q' : float : standard deviation in the quality factor
            For 'array' the following keys are required:
                - 'v_hom' : float : homogeneous velocity if no data_lookup is provided
                - 'std_baz' : float : standard deviation in the backazimuth
                - 'std_inc' : float : standard deviation in the incidence angle
                The standard deviations can be defined in for baz and inc in degrees or in the x and y components of the slowness vector.
                - 'p_x_std' : float : standard deviation in the x-component of the slowness vector
                - 'p_y_std' : float : standard deviation in the y-component of the slowness vector
                - 'correct_for_topography' : bool : whether to correct for topography
                - 'deg_std' : bool : whether the standard deviations are in degrees
                - 'baz_only' : bool : whether to only use the backazimuth
        surface_data : xarray.Dataset, optional
            Dataset with the surface data. Must contain the coordinates 'E' and 'N' for the easting and northing of the surface data. Used to correct array orientation for topography.
        data_lookup : xarray.Dataset
            Dataset with the data lookup table. Must contain the coordinates 'E', 'N', and 'm_sample' for the easting, northing, and model sample indices.
        prior_samples : np.ndarray
            Array with the prior samples. If not provided, the first N_samples in the lookup table are used. Will only be used if data_lookup is provided, to check if the model samples are in the lookup table.
        '''
        
        self.data_dict = data_dict
        self.data_types = list(data_dict.keys())
        
        if surface_data is not None:
            self.surface_data = surface_data
            self.surface_data = add_gradient(surface_data)
        
        self.data_lookup = data_lookup
        self.prior_samples = prior_samples
        
        if data_lookup is not None:
            self.design_E = self.data_lookup['E'].values
            self.design_N = self.data_lookup['N'].values


    def __call__(self, design, model_samples):
        
        if self.data_lookup is not None:
            if self.prior_samples is None:
                warnings.warn('No data_lookup or prior_samples provided. Using the first N_samples in the lookup table. Make sure that those were generated from the right prior distribution')
            
                if self.data_lookup.coords['m_sample'].shape[0] < model_samples.shape[0]:
                    raise ValueError('Not enough samples in the lookup table')
                model_samples = np.arange(model_samples.shape[0])
            else:
                # find the indices of the model samples in the prior samples
                model_samples = np.argwhere(self.prior_samples == model_samples[:, None])[::3, 1]
                
        # print(model_samples)

        # print(model_samples)
              
        mean_list = []
        cov_list  = []
        
        for data_types, location in design:
            
            for d_type in data_types:

                # if lookup table is provided, model samples are indices
                if d_type[:7] == 'arrival':
                    data, cov = self._arrival(d_type, location, model_samples, self.data_dict[d_type])
                elif d_type[:9] == 'amplitude':
                    data, cov = self._amplitude(d_type, location, model_samples, self.data_dict[d_type])
                elif d_type[:5] == 'array':
                    data, cov = self._array(location, model_samples, self.data_dict[d_type])
                else:
                    try:
                        data, cov = self.data_dict[
                            d_type]['fwd_func'](
                                location, model_samples, self.data_dict[d_type])
                    except KeyError:
                        raise ValueError('Data type must be arrival, amplitude, or array. If not, a forward function must be provided in the data dictionary as a callable at key "fwd_func"')
                          
                mean_list.append(data)
                cov_list.append(cov) 
   
        # print([m.shape for m in mean_list])
        # print([c.shape for c in cov_list])
   
        return np.vstack(mean_list).T, np.vstack(cov_list).T
        
    def _raydist(self, design, model_samples):

        model_samples = model_samples[..., None, :, :]
        design        = design[..., None, :3]
                
        return np.linalg.norm(
            (model_samples - design), 2, axis=-1)
        
    def _arrival(self, d_type, design, model_samples, data_dict):
        
        if self.data_lookup is not None and d_type in self.data_lookup.keys():
            
            # mean = self.data_lookup[d_type][model_samples].interp(
            #     E=design[..., 0],
            #     N=design[..., 1],
            #     method='nearest').values
            
            indices_E = np.argmin(np.abs(self.design_E - design[..., 0]), axis=-1)
            indices_N = np.argmin(np.abs(self.design_N - design[..., 1]), axis=-1)
            mean = self.data_lookup[d_type][model_samples].values[:, indices_E, indices_N]
            
            mean = mean[None, ...]
                        
        else:
            ray_dist = self._raydist(design, model_samples)
            mean = ray_dist / data_dict['v_hom']
        
                
        cov = mean * data_dict['std_vel']**2
        if 'std_obs' in data_dict:
            cov += data_dict['std_obs']**2

        return mean,  cov

    def _amplitude(self, d_type, design, model_samples, data_dict):
        ray_dist = self._raydist(design, model_samples)
        s_tt = ray_dist / data_dict['v_hom']
        
        s_tt_cov     = s_tt * data_dict['std_vel']**2
        ray_dist_cov = data_dict['v_hom']**2 * (s_tt * data_dict['std_vel']**2)

        C = np.pi * data_dict['f'] / data_dict['Q']
        C_cov = C**2 * ( data_dict['std_Q'] / data_dict['Q'])**2

        # from uncertainties import unumpy, covariance_matrix

        # un_s_tt = unumpy.uarray(s_tt, np.sqrt(s_tt_cov))
        # un_ray_dist = unumpy.uarray(ray_dist, np.sqrt(ray_dist_cov))
        # un_C = unumpy.uarray(self.C, np.sqrt(C_cov))        
        
        # full_term = unumpy.exp(-un_C * un_s_tt) / un_ray_dist                    
        # asl_data = unumpy.log(unumpy.exp(-un_C * un_s_tt) / un_ray_dist)
                
        ray_dist_term = (1/ray_dist)
        ray_dist_term_cov = (1/ray_dist)**2 * ((np.sqrt(ray_dist_cov)/ray_dist)**2)
                
        exponent_term = -C * s_tt
        exponent_term_cov = exponent_term**2 * ((np.sqrt(C_cov)/C)**2 + (np.sqrt(s_tt_cov)/s_tt)**2)
                
        exp_term = np.exp(exponent_term)
        exp_term_cov = exp_term**2 * exponent_term_cov
                
        asl_data = ray_dist_term * exp_term
        asl_data_cov = asl_data**2 * ((np.sqrt(ray_dist_term_cov)/ray_dist_term)**2 + (np.sqrt(exp_term_cov)/exp_term)**2)

        if 'std_obs' in data_dict:
            asl_data_cov += data_dict['std_obs']**2
                
        asl_data_log = np.log(asl_data)
        asl_data_log_cov = (np.sqrt(asl_data_cov)/asl_data)**2

        return asl_data_log, asl_data_log_cov
    
    def _array(self, design, model_samples, data_dict):
            
        if self.data_lookup is not None and \
            'grad_x' in self.data_lookup.keys() and \
                'grad_y' in self.data_lookup.keys():
    
            indices_E = np.argmin(np.abs(self.design_E - design[..., 0]), axis=-1)
            indices_N = np.argmin(np.abs(self.design_N - design[..., 1]), axis=-1)

            tt_dx = self.data_lookup.grad_x[model_samples].values[:, indices_E, indices_N][..., None]
            tt_dy = self.data_lookup.grad_y[model_samples].values[:, indices_E, indices_N][..., None]
            tt_dz = self.data_lookup.grad_z[model_samples].values[:, indices_E, indices_N][..., None]        
    
                
            connection = np.hstack([tt_dx, tt_dy, tt_dz])
        
        else:
            model_samples = model_samples[..., None, :, :]
            coords        = design[..., None, :3]
            
            connection = coords - model_samples

        if (data_dict['correct_for_topography']):
            if (np.any(np.isnan(design[..., 3:]))):
                if self.surface_data is None:
                    raise ValueError('Topography data must be provided to correct for topography')
                
                gradient = (self.surface_data['grad_E'].interp(E=design[0], N=design[1]).values.item(),
                                self.surface_data['grad_N'].interp(E=design[0], N=design[1]).values.item())        
                normal_vector = np.array([-gradient[0], -gradient[1], 1])
                normal_vector /= np.linalg.norm(normal_vector)
                design = np.hstack((design, normal_vector))

            _r = _rotation(
                np.array([0, 0, 1]), design)
            # both r and connection are normalized
            for i in range(len(connection)):
                connection[i] = np.dot(_r, connection[i].T).T
    
        baz = np.rad2deg(np.arctan2(
            connection[..., 1],
            connection[..., 0]))
        
                
        if data_dict['baz_only']:
            slowness = np.zeros((2, *baz.shape))
            slowness[0] = np.cos(np.radians(baz))
            slowness[1] = np.sin(np.radians(baz))
            
            if data_dict['deg_std']:
                cov = np.ones((slowness.shape))            
                cov *= (data_dict['std_baz']/180 * np.pi)**2
            else:
                raise ValueError('Only degrees are supported when using array_baz_only')
                
        else:
            v_hom = data_dict['v_hom']
            std_baz = data_dict['std_baz']
            std_inc = data_dict['std_inc']
            
            incidence = 90 - np.rad2deg(np.arcsin(
                connection[..., 2] / 
                np.linalg.norm(connection, 2, axis=-1)))
            
            slowness = np.zeros((2, *baz.shape))
            slowness[0] = np.cos(np.radians(baz))
            slowness[1] = np.sin(np.radians(baz))
            slowness *= np.abs(np.sin(np.radians(incidence)[None])/v_hom)
            
            cov = np.ones((2, *baz.shape))
            
            if data_dict['deg_std']:

                cov[0] = (np.sin(np.radians(baz)) * std_baz/v_hom/180 * np.pi)**2 + \
                    (np.cos(np.radians(baz)) *      std_inc/v_hom/180 * np.pi)**2
                cov[1] = (np.cos(np.radians(baz)) * std_baz/v_hom/180 * np.pi)**2 + \
                    (np.sin(np.radians(baz)) *      std_inc/v_hom/180 * np.pi)**2
                                
            else:
                p_x_std = data_dict['p_x_std']
                p_y_std = data_dict['p_y_std']
                
                cov[0] *= p_x_std**2
                cov[1] *= p_y_std**2

        slowness = slowness.reshape(2, -1)
        cov = cov.reshape(2, -1)

        return slowness, cov
    

def _rotation(v1, v2):
    """
    Compute a matrix R that rotates v1 to align with v2.
    v1 and v2 must be length-3 1d numpy arrays.
    """
    
    if len(v2) != 6:
        return np.identity(3)
    v2 = v2[3:]
    
    # unit vectors
    u = v1 / np.linalg.norm(v1)
    Ru = v2 / np.linalg.norm(v2)
    
    # dimension of the space and identity
    dim = u.size
    # the cos angle between the vectors
    c = np.dot(u, Ru)
    # a small number
    eps = 1.0e-10
    if np.abs(c - 1.0) < eps:
        # same direction
        return np.identity(dim)
    elif np.abs(c + 1.0) < eps:
        # opposite direction
        return -np.identity(dim)
    else:
        # the cross product matrix of a vector to rotate around
        K = np.outer(Ru, u) - np.outer(u, Ru)
        # Rodrigues' formula
        return np.identity(dim) + K + (K @ K) / (1 + c)