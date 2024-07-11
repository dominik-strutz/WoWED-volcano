import numpy as np
from scipy.ndimage import label


def construct_design_space(
    surface_data,
    max_incline=None,
    exclude_below_sea_level=True,
    safety_margin=0.0,
    min_area=0.0,
    ):
    
    design_space = surface_data['topography'].copy()
    
    # one/true means that the point is in the design space
    design_space.values = np.ones(surface_data['topography'].shape).astype(bool)
    
    if max_incline is not None:
        dE = surface_data.E.values[1] - surface_data.E.values[0]
        dN = surface_data.N.values[1] - surface_data.N.values[0] 
    
        steepness = np.gradient(surface_data['topography'].values)
        steepness = np.sqrt((steepness[0]/dE)**2 + (steepness[1]/dN)**2)    
        steepness = np.rad2deg(np.arcsin(steepness))
    
        design_space.values = np.where(
            steepness >= max_incline, False, design_space)
    if exclude_below_sea_level:
        design_space.values = np.where(
            surface_data['topography'].values < 0.0, False, design_space)
    
    if safety_margin > 0.0:
        E_grid, N_grid = np.meshgrid(
            surface_data.E.values, surface_data.N.values, indexing='ij')
        
        design_space.values = np.where(
            np.sqrt(
                (E_grid - surface_data.attrs['local_E'])**2 +
                    (N_grid - surface_data.attrs['local_E'])**2) \
                        < safety_margin*1e3,
            False, design_space.values)

    if min_area > 0.0:
        
        # only allow areas where connected true design space is larger than min_area

        # convert min_area to number of grid cells
        min_area = min_area*1e6 / dE / dN

        # find connected areas
        # structure = np.ones((3, 3), dtype=int)
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        
        labeled_array, num_features = label(design_space.values, structure)
        
        for i in range(1, num_features+1):
            if np.sum(labeled_array == i) < min_area:
                design_space.values = np.where(
                    labeled_array == i, False, design_space.values)
                
                

    return design_space