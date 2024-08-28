import numpy as np


class Forward_Function:
    def __init__(self, data_dict, surface_data, prior_data):
        
        self.data_dict = data_dict
        self.data_types = list(data_dict.keys())
        
        self.surface_data = surface_data
        self.prior_data = prior_data
        
        print('Data types:', self.data_types)
        
    def __call__(self, design, model_samples):
        
        
        mean_list = []
        cov_list  = []
        
        for data_types, location in design:
            
            for d_type in data_types:

                if d_type[:7] == 'arrival':
                    data, cov = self._arrival(location, model_samples, self.data_dict[d_type])
                elif d_type[:9] == 'amplitude':
                    data, cov = self._amplitude(location, model_samples, self.data_dict[d_type])
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
   
        return np.vstack(mean_list).T, np.vstack(cov_list).T
        
    def _raydist(self, design, model_samples):

        model_samples = model_samples[..., None, :, :]
        design        = design[..., None, :3]
                
        return np.linalg.norm(
            (model_samples - design), 2, axis=-1)
        
    def _arrival(self, design, model_samples, data_dict):
        ray_dist = self._raydist(design, model_samples)
        mean = ray_dist / data_dict['v_hom']
        
        cov = mean * data_dict['std_vel']**2
        if 'std_obs' in data_dict:
            cov += data_dict['std_obs']**2

        return mean,  cov

    def _amplitude(self, design, model_samples, data_dict):
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
    
        model_samples = model_samples[..., None, :, :]
        coords        = design[..., None, :3]
        
        connection = coords - model_samples

        if (data_dict['correct_for_topography']):
            if (np.any(np.isnan(design[..., 3:]))):
                raise ValueError('Normal vector is missing')
            else:
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

        slowness = slowness.reshape(-1, model_samples.shape[-2])
        cov = cov.reshape(-1, model_samples.shape[-2])

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