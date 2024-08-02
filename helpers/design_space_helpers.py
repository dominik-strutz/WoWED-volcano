import numpy as np
from scipy.ndimage import label


def construct_design_space(
    surface_data,
    max_incline=None,
    exclude_below_sea_level=True,
    safety_margin=0.0,
    safety_margin_center=(0.0, 0.0),
    min_area=0.0,
    ):
    '''
    Construct a design space based on a number of criteria. 
    
    Parameters
    ----------
    surface_data : xarray.DataArray
        The surface data. Should contain the 'topography' variable with the topography data, and the 'E' and 'N' coordinates with the UTM easting and northing values.
    max_incline : float, optional
        The maximum incline in degrees. The default is None.
    exclude_below_sea_level : bool, optional
        Exclude points below sea level. The default is True.
    safety_margin : float, optional
        The safety margin in km around the safety_margin_center. The default is 0.0.
    safety_margin_center : iterable, optional
        The center of the safety margin. The default is (0.0, 0.0).
    min_area : float, optional
        The minimum area in km^2 of connected design space. The default is 0.0. Relvant for when a larger array is planned.
    
    Returns
    -------
    design_space : xarray.DataArray
        The design space with the same coordinates as the surface_data. The values are True for points in the design space and False for points outside the design space.
    '''
    
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
                (E_grid - safety_margin_center[0])**2 +
                    (N_grid - safety_margin_center[1])**2) \
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