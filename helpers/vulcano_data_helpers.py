import os
import re
import requests
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from PIL import Image
from io import BytesIO

def get_vulcanoe_data(vulcanoe_name):
    '''
    Get vulcanoe data from the Global Volcanism Program database.
    
    Parameters
    ----------
    vulcanoe_name : str
        The name of the vulcanoe.
        
    Returns
    -------
    dict
        A dictionary containing the vulcanoe data.
        
    Example
    -------
    >>> get_vulcanoe_data('Kilauea')
    {'Volcano Number': 332010,
     'Volcano Name': 'Kilauea',
     'Primary Volcano Type': 'Shield',
     'Country': 'United States',
     ...
     'lat': 19.421,
     'lon': -155.287}    
    '''
        
    volcanoe_data = pd.read_csv(
        'data/GVP_Volcano_List.csv', header=1)
    # remove everything between parentheses including parentheses
    volcanoe_data['Primary Volcano Type'] = [
        re.sub(r'\([^)]*\)', '', t) for t in volcanoe_data['Primary Volcano Type'].values]

    try:
        volcanoe_data = volcanoe_data[volcanoe_data['Volcano Name'] == vulcanoe_name].to_dict('records')[0]
    except IndexError:
        raise ValueError(f'Vulcanoe {vulcanoe_name} not found. Check the https://volcano.si.edu/ database for the correct name.')
    
    volcanoe_data['location'] = (volcanoe_data['Latitude'], volcanoe_data['Longitude'])
    volcanoe_data['lat'], volcanoe_data['lon'] = volcanoe_data['location']
        
    volcanoe_data['E'], volcanoe_data['N'] = utm.from_latlon(volcanoe_data['lat'], volcanoe_data['lon'])[:2]
        
    del volcanoe_data['Latitude']
    del volcanoe_data['Longitude']

    return volcanoe_data


def grid_latlon2utm(lat, lon):
    x_coor = np.zeros((lon.size, 2))
    x_coor[:, 0] = lon
    x_coor[:, 1] = lat[0]

    y_coor = np.zeros((lat.size, 2))
    y_coor[:, 0] = lon[0]
    y_coor[:, 1] = lat

    E, _, _, _ = utm.from_latlon(x_coor[:, 1], x_coor[:, 0])
    _, N, _, _ = utm.from_latlon(y_coor[:, 1], y_coor[:, 0])

    del x_coor, y_coor
    if N[0] < N[-1]:
        return E, N[::-1]
    else:
        return E, N
    
def fetch_topography_data(
    vulcano_data, bounding_box,
    api_key='demoapikeyot2022', demtype='SRTM15Plus',
    cells_per_dimension=128):

    if set(['min_lat', 'max_lat', 'min_lon', 'max_lon']) == set(bounding_box.keys()):
        bounding_box_type = 'latlon'
    elif set(['extent_south', 'extent_north', 'extent_west', 'extent_east']) == set(bounding_box.keys()):
        bounding_box_type = 'km'
    else:
        raise ValueError('Invalid bounding box. Use "min_lat", "max_lat", "min_lon", "max_lon" or "extent_south", "extent_noth", "extent_west", "extent_east"')

    center = vulcano_data['location']
    if bounding_box_type == 'latlon':
        south = bounding_box['min_lat']
        north = bounding_box['max_lat']
        west = bounding_box['min_lon']
        east = bounding_box['max_lon']
    elif bounding_box_type == 'km':
        south = center[0] - bounding_box['extent_south'] / 111.32
        north = center[0] + bounding_box['extent_north'] / 111.32
        
        west = center[1] - bounding_box['extent_west'] / np.cos(np.radians(center[0])) / 111.32
        east = center[1] + bounding_box['extent_east'] / np.cos(np.radians(center[0])) / 111.32        

    # cache_filename = f'data/topo_cache/{demtype}_{south}_{north}_{west:}_{east}.tif'
    cache_filename = f'data/topo_cache/{demtype}_{south:.5f}_{north:.5f}_{west:.5f}_{east:.5f}.tif'
        
    if os.path.exists(cache_filename):
        im = Image.open(cache_filename)
        print('Data loaded from cache')
    else:
        if api_key == 'demoapikeyot2022':
            warnings.warn('WARNING: You are using a demo key which should work for a limited number of requests. If you want to use the API more frequently, or if the key stops working, you can get your own key (see https://opentopography.org/blog/introducing-api-keys-access-opentopography-global-datasets for more information).')


        # api_request = https://portal.opentopography.org/API/globaldem?demtype=SRTM15Plus&south=50&north=50.1&west=14.35&east=14.6&outputFormat=GTiff&API_Key=demoapikeyot2022
        api_request = f"https://portal.opentopography.org/API/globaldem?demtype={demtype}&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff&API_Key={api_key}"

        response = requests.get(api_request)

        if response.status_code == 200:
            print('Data downloaded successfully')
        else:
            raise ValueError(f'Error fetching data. Status code: {response.status_code}, {response.text}')
            
        im = Image.open(BytesIO(response.content))
        
        im.save(cache_filename)
        
    lat, lon = np.linspace(south, north, im.size[1]), np.linspace(west, east, im.size[0])

    # reomve band coordinates
    E, N = grid_latlon2utm(lat, lon)

    mean_E = E.mean()
    mean_N = N.mean()

    # working with local coordinates
    E = E - mean_E
    N = N - mean_N

    topo_data = xr.Dataset(
        data_vars={
            'topography': (('E', 'N'), np.array(im).T)
        },
        coords={
            'N': N,
            'E': E
        }
    )
    topo_data = topo_data.assign_coords(dict(
        lat=('N', lat),
        lon=('E', lon)))
    
    topo_data = topo_data.interp(
        E=np.linspace(
            topo_data.E.min(), topo_data.E.max(), cells_per_dimension, dtype=np.float32),
        N=np.linspace(
            topo_data.N.min(), topo_data.N.max(), cells_per_dimension, dtype=np.float32),
        method='linear'
    )
    
    topo_data.attrs['mean_E'] = mean_E
    topo_data.attrs['mean_N'] = mean_N
    
    if bounding_box_type == 'latlon':
        topo_data.attrs['local_E'] = utm.from_latlon(center[0], center[1])[0] - mean_E
        topo_data.attrs['local_N'] = utm.from_latlon(center[0], center[1])[1] - mean_N
    else:
        topo_data.attrs['local_E'] = bounding_box['extent_west']*1e3 + E.min()
        topo_data.attrs['local_N'] = bounding_box['extent_south']*1e3 + N.min()

    topo_data.attrs['dE'] = (E.max() - E.min()) / (E.size - 1)
    topo_data.attrs['dN'] = (N.max() - N.min()) / (N.size - 1)

    # add another data var with local gradient
    grad_E, grad_N = np.gradient(topo_data['topography'], topo_data.attrs['dE'], topo_data.attrs['dN'])
    topo_data['grad_E'] = (('E', 'N'), grad_E)
    topo_data['grad_N'] = (('E', 'N'), grad_N)

    return topo_data


def construct_highly_opinionated_prior(
    vulcano_data, surface_data, depth_min,
    center_location,
    max_distance_hor=None,
    max_distance_vert=None,
    prop2elevation=True,
    hor_drop_power=1/2,
    vert_drop_power=1/2
    ):
    
    z_max = surface_data['topography'].max()
    z_min = depth_min*1e3 # add some extra space below the surface
    Nz = surface_data['topography'].shape[0]
    z = np.linspace(z_min, z_max, Nz, dtype=np.float32)
    dz = z[1] - z[0]
    
    prior_data = xr.DataArray(
        z[None, None, :]
        .repeat(surface_data["E"].size, axis=0)
        .repeat(surface_data["N"].size, axis=1),
        dims=["E", "N", "Z"],
        coords={"E": surface_data["E"], "N": surface_data["N"], "Z": z},
    )

    print(f"Size of the volume data: {prior_data.nbytes / 1e6:.2f} MB")

    distance_hor = np.sqrt(
        ((surface_data["E"] - center_location[0]*1e3) ** 2 + \
            (surface_data["N"] - center_location[1]*1e3) ** 2)
    )
    distance_vert = np.sqrt(
        (z - center_location[2]*1e3) ** 2
    )
    for i, hor_slice in enumerate(prior_data.values.swapaxes(0, 2).swapaxes(1, 2)):
        # set likelihood to 0 for all points above the surface
        prior_data.values[:, :, i] = np.where(
            hor_slice < surface_data["topography"].values, 1, np.nan
        )

        # set likeliehood from 1 to zero in zylinder with radius max_distance_hor
        if max_distance_hor is not None:
            prior_data.values[:, :, i] = np.where(
                distance_hor < max_distance_hor*1e3,
                prior_data.values[:, :, i] * \
                    (1 - distance_hor / max_distance_hor * 1e-3)**(hor_drop_power),
                np.nan)
        # set likeliehood from 1 to zero in zylinder with radius max_distance_vert
        if max_distance_vert is not None:
            prior_data.values[:, :, i] = np.where(
                distance_vert[i] < max_distance_vert*1e3,
                prior_data.values[:, :, i] * \
                    (1 - distance_vert[i] / max_distance_vert * 1e-3)**(vert_drop_power),
                np.nan)
             

        # assume that higher elevations means higher likelihood
        if prop2elevation:
            prior_data.values[:, :, i] = (
                prior_data.values[:, :, i]
                * surface_data["topography"].values
                / (surface_data["topography"].values.max() - surface_data["topography"].values.min())
            )

    prior_data.values = prior_data.where(prior_data.values >= 0, 0)
    prior_data.values = np.nan_to_num(prior_data.values, nan=0)
    prior_data.values = prior_data.values / np.sum(prior_data.values)
    
    prior_data.attrs['dE'] = surface_data['E'].values[1] - surface_data['E'].values[0]
    prior_data.attrs['dN'] = surface_data['N'].values[1] - surface_data['N'].values[0]
    prior_data.attrs['dZ'] = dz
    
    cell_volume = prior_data.attrs['dE'] * prior_data.attrs['dN'] * dz
    prior_data.attrs['cell_volume'] = cell_volume
        
    return prior_data

        
def get_elevation(location, surface_data):
    E = location[..., 0]*1e3
    N = location[..., 1]*1e3
    
    E_idx = np.argmin(np.abs(surface_data['E'].values - E))
    N_idx = np.argmin(np.abs(surface_data['N'].values - N))
    
    return surface_data['topography'].values[E_idx, N_idx]