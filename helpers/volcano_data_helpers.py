import os
import re
import requests
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import ipywidgets as widgets
from PIL import Image
from io import BytesIO

import utm

def choose_volcano(default_volcano='Etna'):
    
    volcano_data = pd.read_csv("data/GVP_Volcano_List_Holocene.csv", header=1, engine='python', encoding='latin-1')
    select_volcano = widgets.Combobox(
            value=default_volcano,
            placeholder='Choose Someone',
            options=volcano_data['Volcano Name'].values.tolist(),
            description='Choose volcano:',
            ensure_option=True,
            disabled=False
        )
    
    return select_volcano

def get_volcano_data(volcano_name):
    """
    Get volcano data from the Global Volcanism Program database.

    Parameters
    ----------
    volcano_name : str
        The name of the volcano.

    Returns
    -------
    dict
        A dictionary containing the volcano data.

    Example
    -------
    >>> get_volcano_data('Kilauea')
    {'Volcano Number': 332010,
     'Volcano Name': 'Kilauea',
     'Primary Volcano Type': 'Shield',
     'Country': 'United States',
     ...
     'lat': 19.421,
     'lon': -155.287}
    """

    volcanoe_data = pd.read_csv("data/GVP_Volcano_List_Holocene.csv", header=1, engine='python', encoding='latin-1')
    
    # Last Known Eruption from julian (ad, bc, bce) to datetime
    last_known_erruption = []
    for date in volcanoe_data["Last Known Eruption"]:
        if isinstance(date, str):            
            if date == "Unknown":
                year = np.nan
            elif 'BC' in date or 'BCE' in date:
                year = int(date.split()[0])
                year = -year
                
            elif 'AD' in date or 'CE' in date:
                year = int(date.split()[0])
            else:
                raise ValueError(f"Unknown date format: {date}")

            last_known_erruption.append(year)
        
    volcanoe_data["Last Known Eruption"] = last_known_erruption
    
    # remove everything between parentheses including parentheses
    volcanoe_data["Primary Volcano Type"] = [
        re.sub(r"\([^)]*\)", "", t)
        for t in volcanoe_data["Primary Volcano Type"].values
    ]

    try:
        volcanoe_data = volcanoe_data[
            volcanoe_data["Volcano Name"] == volcano_name
        ].to_dict("records")[0]
    except IndexError:
        raise ValueError(
            f"Volcanoe {volcano_name} not found. Check the https://volcano.si.edu/volcanolist_holocene.cfm database for the correct name."
        )

    volcanoe_data["lat"], volcanoe_data["lon"] = (
        volcanoe_data["Latitude"],
        volcanoe_data["Longitude"],
    )

    volcanoe_data["E"], volcanoe_data["N"] = utm.from_latlon(
        volcanoe_data["lat"], volcanoe_data["lon"]
    )[:2]

    del volcanoe_data["Latitude"]
    del volcanoe_data["Longitude"]

    return volcanoe_data


def grid_latlon2utm(lat, lon):
    """
    Helper function that converts latitude and longitude axes to UTM coordinate axes.

    Parameters
    ----------
    lat : np.array
        Array of latitude values.
    lon : np.array

    Returns
    -------
    np.array
        Array of UTM easting values.
    np.array
        Array of UTM northing values.

    """

    x_coor = np.zeros((lon.size, 2))
    x_coor[:, 0] = lon
    x_coor[:, 1] = lat[0]

    y_coor = np.zeros((lat.size, 2))
    y_coor[:, 0] = lon[0]
    y_coor[:, 1] = lat

    try:
        E, _, _, _ = utm.from_latlon(x_coor[:, 1], x_coor[:, 0])
        _, N, _, _ = utm.from_latlon(y_coor[:, 1], y_coor[:, 0])
    except ValueError:
        raise ValueError(
            "Latitude and longitude conversion does not allow for latitudes to cross the equator. Feel free to submit a pull request to fix this issue."
        )
    del x_coor, y_coor
    if N[0] < N[-1]:
        return E, N[::-1]
    else:
        return E, N


def fetch_topography_data(
    volcano_data,
    bounding_box,
    api_key="demoapikeyot2022",
    demtype="SRTM15Plus",
    local_coordinates=True,
    cells_E=256,
    cells_N=256,
):
    """
    Fetch topography data from the OpenTopography API and convert it to a xarray dataset with UTM coordinates and added gradient data.

    Parameters
    ----------
    volcano_data : dict
        The volcano data. Should contain the keys 'lat', 'lon', 'E', 'N' which are the latitude, longitude, easting and northing of the center of the volcano.
    bounding_box : dict
        The bounding box of the area to fetch. Should contain the keys 'min_lat', 'max_lat', 'min_lon', 'max_lon' or 'extent_south', 'extent_north', 'extent_west', 'extent_east'.
    api_key : str
        The OpenTopography API key. Default is 'demoapikeyot2022'. You can get your own key (see https://opentopography.org/blog/introducing-api-keys-access-opentopography-global-datasets for more information). The demo key should work for a limited number of requests.
    demtype : str
        The type of the digital elevation model. Default is 'SRTM15Plus'.
    local_coordinates : bool
        If True, the UTM coordinates are relative to the center of the volcano. Default is True.
    cells_E : int
        The number of cells in the easting direction. Default is 256.
    cells_N : int
        The number of cells in the northing direction. Default is 256.

    Returns
    -------
    xr.Dataset
        The topography data. The dataset contains the 'topography' variable with the topography data, and the 'E' and 'N' coordinates with the UTM easting and northing values. Latitude and longitude coordinates are added as secondary coordinates. The dataset is interpolated to the number of cells specified.
    """

    if set(["min_lat", "max_lat", "min_lon", "max_lon"]) == set(bounding_box.keys()):
        bounding_box_type = "latlon"
    elif set(["extent_south", "extent_north", "extent_west", "extent_east"]) == set(
        bounding_box.keys()
    ):
        bounding_box_type = "km"
    else:
        raise ValueError(
            'Invalid bounding box. Use "min_lat", "max_lat", "min_lon", "max_lon" or "extent_south", "extent_noth", "extent_west", "extent_east"'
        )

    center_latlon = volcano_data["lat"], volcano_data["lon"]
    center_utm = volcano_data["E"], volcano_data["N"]

    if bounding_box_type == "latlon":
        south = bounding_box["min_lat"]
        north = bounding_box["max_lat"]
        west = bounding_box["min_lon"]
        east = bounding_box["max_lon"]
    elif bounding_box_type == "km":
        south = center_latlon[0] - bounding_box["extent_south"] / 111.32
        north = center_latlon[0] + bounding_box["extent_north"] / 111.32

        west = (
            center_latlon[1]
            - bounding_box["extent_west"]
            / np.cos(np.radians(center_latlon[0]))
            / 111.32
        )
        east = (
            center_latlon[1]
            + bounding_box["extent_east"]
            / np.cos(np.radians(center_latlon[0]))
            / 111.32
        )

    print(
        f"Fetching topography data for {demtype} between {south:.5f} and {north:.5f} latitude and {west:.5f} and {east:.5f} longitude"
    )

    cache_filename = (
        f"data/topo_cache/{demtype}_{south:.5f}_{north:.5f}_{west:.5f}_{east:.5f}.tif"
    )

    if os.path.exists(cache_filename):
        im = Image.open(cache_filename)
        print("Data loaded from cache")
    else:
        if api_key == "demoapikeyot2022":
            warnings.warn(
                "WARNING: You are using a demo key which should work for a limited number of requests. If you want to use the API more frequently, or if the key stops working, you can get your own key (see https://opentopography.org/blog/introducing-api-keys-access-opentopography-global-datasets for more information)."
            )

        # api_request = https://portal.opentopography.org/API/globaldem?demtype=SRTM15Plus&south=50&north=50.1&west=14.35&east=14.6&outputFormat=GTiff&API_Key=demoapikeyot2022
        api_request = f"https://portal.opentopography.org/API/globaldem?demtype={demtype}&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff&API_Key={api_key}"

        response = requests.get(api_request)

        if response.status_code == 200:
            print("Data downloaded successfully")
        else:
            raise ValueError(
                f"Error fetching data. Status code: {response.status_code}, {response.text}"
            )

        im = Image.open(BytesIO(response.content))

        im.save(cache_filename)

    lat, lon = (
        np.linspace(south, north, im.size[1]),
        np.linspace(west, east, im.size[0]),
    )
    E, N = grid_latlon2utm(lat, lon)

    if local_coordinates:
        E = E - center_utm[0]
        N = N - center_utm[1]

    surface_data = xr.Dataset(
        data_vars={"topography": (("E", "N"), np.array(im).T)}, coords={"N": N, "E": E}
    )
    surface_data = surface_data.assign_coords(dict(lat=("N", lat), lon=("E", lon)))

    surface_data = surface_data.interp(
        E=np.linspace(
            surface_data.E.values.min(), surface_data.E.values.max(), cells_E, dtype=np.float32
        ),
        N=np.linspace(
            surface_data.N.values.min(), surface_data.N.values.max(), cells_N, dtype=np.float32
        ),
        method="linear", kwargs={"fill_value": 'extrapolate', "bounds_error": False}
    )
    
    return surface_data


def construct_highly_opinionated_prior(
    surface_data,
    depth_max,
    center_location=None,
    standard_deviation=None,
    prop2elevation=True,
    cells_Z=256,
):
    """
    Construct a discretised prior distribution with some simple opinionated assumptions.

    Parameters
    ----------
    surface_data : xr.Dataset
        The surface data. Should contain the 'topography' variable with the topography data, and the 'E' and 'N' coordinates with the UTM easting and northing values.
    depth_max : float
        The maximum depth of the prior distribution in km.
    center_location : tuple
        The center of the prior distribution in km. Should be a tuple with the E, N, Z coordinates.
    standard_deviation : float or iterable
        The standard deviation of the Gaussian prior distribution centered around the center_location. If a float, the same standard deviation is used for all dimensions. If an iterable, the standard deviation is used for each dimension.
    prop2elevation : bool
        If True, the prior likelihood is proportional to the elevation. Default is True.
    cells_Z : int
        The number of cells in the vertical direction. Default is 256.

    Returns
    -------
    xr.DataArray
        The discretised prior distribution. The data is normalized to sum to 1.
    """

    z_max = surface_data["topography"].values.max()
    z_min = -depth_max * 1e3  # add some extra space below the surface
    z = np.linspace(z_min, z_max, cells_Z, dtype=np.float32)

    prior_data = xr.DataArray(
        np.ones((surface_data["E"].size, surface_data["N"].size, z.size)),
        dims=["E", "N", "Z"],
        coords={"E": surface_data["E"], "N": surface_data["N"], "Z": z},
    )

    if (center_location is not None) and (standard_deviation is not None):
        E, N, Z = np.meshgrid(
            surface_data["E"].values, surface_data["N"].values, z, indexing="ij"
        )
        E = E - center_location[0] * 1e3
        N = N - center_location[1] * 1e3
        Z = Z - center_location[2] * 1e3

        if np.isscalar(standard_deviation):
            standard_deviation = np.array(
                [standard_deviation, standard_deviation, standard_deviation]
            )
        standard_deviation = np.array(standard_deviation) * 1e3

        prior_data.values = np.exp(
            -0.5
            * (
                (E / standard_deviation[0]) ** 2
                + (N / standard_deviation[1]) ** 2
                + (Z / standard_deviation[2]) ** 2
            )
        )

    # assume that higher elevations means higher likelihood
    if prop2elevation:
        prior_data.values[:, :, :] = (
            prior_data.values[:, :, :]
            * surface_data["topography"].values.reshape(
                prior_data["E"].size, prior_data["N"].size, 1
            )
            / (
                surface_data["topography"].values.max()
                - surface_data["topography"].values.min()
            )
        )

    for i, z in enumerate(prior_data["Z"].values):
        # # set likelihood to 0 for all points above the surface
        prior_data.values[:, :, i] = np.where(
            z <= surface_data["topography"].values, prior_data.values[:, :, i], 0.0
        )

    prior_data.values = prior_data.where(prior_data.values >= 0, 0)
    prior_data.values = np.nan_to_num(prior_data.values, nan=0)

    prior_data.values = prior_data.values / np.sum(prior_data.values)

    return prior_data


def calculate_prior_information(prior_data):
    """
    Calculate the prior information from the prior data given as a xarray DataArray.

    Parameters
    ----------
    prior_data : xr.DataArray
        The prior data. The data should be normalized to sum to 1.

    Returns
    -------
    float
        The prior information. Units are nats.
    """

    dE = prior_data["E"].values[1] - prior_data["E"].values[0]
    dN = prior_data["N"].values[1] - prior_data["N"].values[0]
    dZ = prior_data["Z"].values[1] - prior_data["Z"].values[0]

    cell_volume = dE * dN * dZ

    prior_mask = prior_data.values > 0.0
    prior_information = (
        np.log(prior_data.values[prior_mask] / cell_volume)
        * prior_data.values[prior_mask]
    ).sum()

    return prior_information


def get_elevation(location, surface_data):
    E = location[..., 0] * 1e3
    N = location[..., 1] * 1e3

    E_idx = np.argmin(np.abs(surface_data["E"].values - E))
    N_idx = np.argmin(np.abs(surface_data["N"].values - N))

    return surface_data["topography"].values[E_idx, N_idx]
