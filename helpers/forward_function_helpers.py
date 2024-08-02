import numpy as np


class Forward_Function:
    def __init__(
        self, vp=4000, ps_ratio=1/np.sqrt(3), Q=50, f=5.0,
        tt_std_obs=0.01, 
        tt_std_vel=0.1,
        asl_std_Q = 10,
        baz_std=6,
        inc_std=20,
        p_x_std=None,
        p_y_std=None,
        correct_array_orientation=True,
        array_baz_only=True
        ):
        '''
        Forward function for an homogenous subsurface model defined by the parameters vp, ps_ratio, and Q.
        
        Parameters
        ----------
        vp : float, optional
            The velocity of the P-waves in the subsurface. Units are m/s. The default is 4000.
        ps_ratio : float, optional
            The ratio between the P-wave and S-wave velocities. The default is 1/sqrt(3).
        Q : float, optional
            The quality factor of the subsurface. The default is 50.
        f : float, optional
            The frequency used for amplitude source location. The default is 5.0.
        tt_std_obs : float, optional
            The standard deviation of the observed travel times due to measurement errors. The default is 0.01.
        tt_std_vel : float, optional
            The standard deviation of the observed travel times due to uncertainties in the velocity model. Approximates the uncertainty accumulated over the ray path as a radnom walk, resulting in the following standard deviation contribution: sqrt(tt * tt_std_vel**2). The default is 0.1.
        asl_std_Q : float, optional
            The standard deviation of the uncertainties in the quality factor. Is propagated through the amplitude source location calculation. The default is 10.
        baz_std : float, optional
            The standard deviation of the uncertainties in the backazimuth. The default is 6 degrees.
        inc_std : float, optional
            The standard deviation of the uncertainties in the incidence angle. The default is 20 degrees.
        p_x_std : float, optional
            The standard deviation of the uncertainties in the x-component of the slowness vector. Can be used as an alternative to baz_std and inc_std. The default is None.
        p_y_std : float, optional
            The standard deviation of the uncertainties in the y-component of the slowness vector. Can be used as an alternative to baz_std and inc_std. The default is None.
        correct_array_orientation : bool, optional
            Correct the orientation of the array based on the local gradient of the surface. The default is True.
        array_baz_only : bool, optional
            Use only the backazimuth for the array design. Incidence angles are often hard to estimate. The default is True.        
        '''
        
        self.vp = vp
        
        self.Q = Q
        self.f = f
        self.C = np.pi * f / Q
        
        self.ps_ratio = ps_ratio

        self.tt_std_obs = tt_std_obs        
        self.tt_std_vel = tt_std_vel
        
        self.asl_std_Q   = asl_std_Q
        
        deg_std = baz_std is not None
        self.deg_std = deg_std
        
        if deg_std:
            self.baz_std = baz_std
            self.inc_std = inc_std
        else:
            self.p_x_std = p_x_std
            self.p_y_std = p_y_std
        
        self.array_correct_orientation = correct_array_orientation
        self.array_baz_only = array_baz_only

    def __call__(self, design, model_samples):
        
        design_coord_dict = {}
        for desc, loc in design:
            for d_i in desc:
                if d_i not in design_coord_dict:
                    design_coord_dict[d_i] = []
                    
                if loc.shape[-1] == 3:
                    # add dummy normal vector
                    loc = np.concatenate([loc, np.array([np.nan, np.nan, np.nan])])
                
                design_coord_dict[d_i].append(loc)
                
        design_data = []
        for d_i in design_coord_dict:
            design_coord_dict[d_i] = np.vstack(design_coord_dict[d_i])
            
            design_data.append({
                    # 'p': self._p_tt,
                    # 's': self._s_tt,
                    # 'ps_diff': self._ps_diff,
                    'tt': self._p_tt,
                    'asl': self._asl,
                    'array': self._array,
                }[d_i](design_coord_dict[d_i], model_samples))
            
        data_array = [d[0].T for d in design_data]
        cov_array = [d[1].T for d in design_data]

        return np.hstack(data_array), np.hstack(cov_array)
    
    def _raydist(self, design, model_samples):

        model_samples = model_samples[..., None, :, :]
        design        = design[..., None, :3]
                
        return np.linalg.norm(
            (model_samples - design), 2, axis=-1)
        
    def _p_tt(self, design, model_samples):
        ray_dist = self._raydist(design, model_samples)
        p_tt = ray_dist / self.vp
                
        return p_tt, self.tt_std_obs**2 + p_tt * self.tt_std_vel**2

    def _asl(self, design, model_samples):        
        ray_dist = self._raydist(design, model_samples)
        s_tt = ray_dist / (self.vp * self.ps_ratio)
        
        
        s_tt_cov     = self.tt_std_obs**2 + s_tt * self.tt_std_vel**2
        ray_dist_cov =  (self.vp * self.ps_ratio)**2 * (s_tt * self.tt_std_vel**2)

        C = self.C
        C_cov = C**2 * (self.asl_std_Q / self.Q)**2

        # from uncertainties import unumpy, covariance_matrix

        # un_s_tt = unumpy.uarray(s_tt, np.sqrt(s_tt_cov))
        # un_ray_dist = unumpy.uarray(ray_dist, np.sqrt(ray_dist_cov))
        # un_C = unumpy.uarray(self.C, np.sqrt(C_cov))        
        
        # full_term = unumpy.exp(-un_C * un_s_tt) / un_ray_dist                    
        # asl_data = unumpy.log(unumpy.exp(-un_C * un_s_tt) / un_ray_dist)
                
        ray_dist_term = (1/ray_dist)
        ray_dist_term_cov = (1/ray_dist)**2 * ((np.sqrt(ray_dist_cov)/ray_dist)**2)
                
        exponent_term = -self.C * s_tt
        exponent_term_cov = exponent_term**2 * ((np.sqrt(C_cov)/C)**2 + (np.sqrt(s_tt_cov)/s_tt)**2)
                
        exp_term = np.exp(exponent_term)
        exp_term_cov = exp_term**2 * exponent_term_cov
                
        asl_data = ray_dist_term * exp_term
        asl_data_cov = asl_data**2 * ((np.sqrt(ray_dist_term_cov)/ray_dist_term)**2 + (np.sqrt(exp_term_cov)/exp_term)**2)
                
        asl_data_log = np.log(asl_data)
        asl_data_log_cov = (np.sqrt(asl_data_cov)/asl_data)**2
                
        return asl_data_log, asl_data_log_cov
    
    def _array(self, design, model_samples):
    
        model_samples = model_samples[..., None, :, :]
        coords        = design[..., None, :3]
        
        connection = coords - model_samples

        if (self.array_correct_orientation):
            if (np.any(np.isnan(design[..., 3:]))):
                raise ValueError('Normal vector is missing')
            else:
                _rotations = [_rotation(
                    np.array([0, 0, 1]), d[3:]) for d in design]
                # both r and connection are normalized
                for i, r in enumerate(_rotations):
                    connection[i] = np.dot(r, connection[i].T).T
    
        baz = np.rad2deg(np.arctan2(
            connection[..., 1],
            connection[..., 0]))
        
        if self.array_baz_only:
            slowness = np.zeros((2, *baz.shape))
            slowness[0] = np.cos(np.radians(baz))
            slowness[1] = np.sin(np.radians(baz))
            
            if self.deg_std:
                cov = np.ones((slowness.shape))            
                cov *= (self.baz_std/180 * np.pi)**2
            else:
                raise ValueError('Only degrees are supported when using array_baz_only')
                
        else:
            incidence = 90 - np.rad2deg(np.arcsin(
                connection[..., 2] / 
                np.linalg.norm(connection, 2, axis=-1)))
            
            slowness = np.zeros((2, *baz.shape))
            slowness[0] = np.cos(np.radians(baz))
            slowness[1] = np.sin(np.radians(baz))
            slowness *= np.abs(np.sin(np.radians(incidence)[None])/self.vp)
            
            cov = np.ones((2, *baz.shape))
            
            if self.deg_std:
                cov[0] = (np.sin(np.radians(baz)) * self.baz_std/self.vp/180 * np.pi)**2 + \
                    (np.cos(np.radians(baz)) * self.inc_std/self.vp/180 * np.pi)**2
                cov[1] = (np.cos(np.radians(baz)) * self.baz_std/self.vp/180 * np.pi)**2 + \
                    (np.sin(np.radians(baz)) * self.inc_std/self.vp/180 * np.pi)**2
                                
            else:
                cov[0] *= self.p_x_std**2
                cov[1] *= self.p_y_std**2

        slowness = slowness.reshape(-1, model_samples.shape[-2])
        cov = cov.reshape(-1, model_samples.shape[-2])

        return slowness, cov

def _rotation(v1, v2):
    """
    Compute a matrix R that rotates v1 to align with v2.
    v1 and v2 must be length-3 1d numpy arrays.
    """
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