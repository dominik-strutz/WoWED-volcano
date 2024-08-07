import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from skimage import measure
from scipy.interpolate import interp1d

import torch
import bqplot.pyplot as bqp_plt
from bqplot import ColorScale

from helpers.volcano_data_helpers import get_elevation

blue_cmap = plt.cm.Blues
blue_cmap = blue_cmap(np.arange(blue_cmap.N))
blue_cmap[:, -1] = np.linspace(0, 1, blue_cmap.shape[0])*0.8
blue_cmap[:, :3] = [0.0, 0.0, 1.0]
blue_cmap = ListedColormap(blue_cmap)

red_cmap = plt.cm.Reds
red_cmap = red_cmap(np.arange(red_cmap.N))
red_cmap[:, -1] = np.linspace(0, 1, red_cmap.shape[0])*1.0
red_cmap[:, :3] = [1.0, 0.0, 0.0]
red_cmap = ListedColormap(red_cmap)

binary_cmap = ListedColormap(
    [(0.6, 0, 0, 1), (0, 0, 0, 0)])

def plot_topography(ax, topo_array):
    '''
    Plot topography data in the form of a xarray.DataArray on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot the topography data on.
    topo_ds : xarray.DataArray
        Topography data in the form of a xarray.DataArray. The DataArray should have
        dimensions 'E' and 'N' for the easting and northing coordinates and the values
        should be the elevation in meters.
    
    Returns
    -------
    matplotlib.axes.Axes
        The axis with the topography data plotted on it.    
    '''
    
    E_min, E_max = topo_array.E.min(), topo_array.E.max()
    N_min, N_max = topo_array.N.min(), topo_array.N.max()
    
    ax.imshow(
        topo_array.values.T,
        cmap="Greys",
        extent=[
            E_min * 1e-3, E_max * 1e-3, N_min * 1e-3, N_max * 1e-3,
        ],
        origin='lower',
        alpha=0.8,
    )
    
    ax.pc = ax.contour(
        topo_array.E * 1e-3, topo_array.N * 1e-3,
        topo_array.values.T,
        colors="k",
        zorder=-10,
        # levels divisions of 500 from min to max
        levels=np.arange(
            topo_array.values.min().round(-3), (topo_array.values.max() + 1000).round(-3), 500
        ),
        linewidths=1.0, alpha=1.0,
    )

    ax.cl = ax.clabel(
        ax.pc,
        levels=np.arange(
            topo_array.values.min().round(-3), (topo_array.values.max() + 1000).round(-3), 500
        ),
        inline=True, fontsize=8, fmt="%1.0f", colors="k", use_clabeltext=True,
    )

    ax.set_aspect("equal")

    ax.set_xlabel("E [km]")
    ax.set_ylabel("N [km]")

    return ax

def plot_slice_N(
    ax, volume_data, surface_data, slice_N,
    colorbar=False,
    labels=True,
    type='imshow',
    **kwargs):
    
    aspect = kwargs.pop('aspect', 'auto')

    if type == 'imshow':
        im = ax.imshow(volume_data.sel(
            N=slice_N, method='nearest').T, origin='lower',
            extent=[volume_data['E'].min()*1e-3, volume_data['E'].max()*1e-3,
                    volume_data['Z'].min()*1e-3, volume_data['Z'].max()*1e-3],
            **kwargs)
    elif type == 'contour':
        im = ax.contour(
            volume_data['E']*1e-3, volume_data['Z']*1e-3,
            volume_data.sel(N=slice_N, method='nearest').T,
            **kwargs)
    elif type == 'contourf':
        im = ax.contourf(
            volume_data['E']*1e-3, volume_data['Z']*1e-3,
            volume_data.sel(N=slice_N, method='nearest').T,
            **kwargs)
        
    ax.plot(surface_data['E']*1e-3,
            surface_data['topography'].sel(
                N=slice_N, method='nearest')*1e-3, 'k', lw=1, zorder=20)
    
    if labels:
        ax.set_xlabel('E [km]')
        ax.set_ylabel('Z [km]')
    
    if aspect is not None:
        ax.set_aspect(aspect)

    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.6)

    return ax
    
def plot_slice_E(
    ax, volume_data, surface_data, slice_E,
    colorbar=False,
    labels=True, type='imshow',
    **kwargs):
    
    aspect = kwargs.pop('aspect', 'auto')
    
    if type == 'imshow':
        im = ax.imshow(volume_data.sel(
            E=slice_E, method='nearest').T, origin='lower',
            extent=[volume_data['N'].min()*1e-3, volume_data['N'].max()*1e-3,
                    volume_data['Z'].min()*1e-3, volume_data['Z'].max()*1e-3],
            **kwargs)
    elif type == 'contour':
        im = ax.contour(
            volume_data['N']*1e-3, volume_data['Z']*1e-3,
            volume_data.sel(E=slice_E, method='nearest').T,
            **kwargs)
    elif type == 'contourf':
        im = ax.contourf(
            volume_data['N']*1e-3, volume_data['Z']*1e-3,
            volume_data.sel(E=slice_E, method='nearest').T,
            **kwargs)
    
    ax.plot(surface_data['N']*1e-3,
            surface_data['topography'].sel(
                E=slice_E, method='nearest')*1e-3, 'k', lw=1, zorder=20)
    
    if labels:
        ax.set_xlabel('N [km]')
        ax.set_ylabel('Z [km]')
    
    if aspect is not None:
        ax.set_aspect(aspect)
    
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    return ax

def plot_marginal_Z(
    ax, volume_data, surface_data,
    slice_E=None, slice_N=None,
    colorbar=False,
    labels=True,
    type='imshow',
    **kwargs):
    
    aspect = kwargs.pop('aspect', 'auto')
    
    if type == 'imshow':
        im = ax.imshow(
            volume_data.mean('Z').T, origin='lower',
            extent=[volume_data['E'].min()*1e-3, volume_data['E'].max()*1e-3,
                volume_data['N'].min()*1e-3, volume_data['N'].max()*1e-3],
            **kwargs)
    elif type == 'contour':
        im = ax.contour(
            volume_data['E']*1e-3, volume_data['N']*1e-3,
            volume_data.mean('Z').T, **kwargs)
    elif type == 'contourf':
        im = ax.contourf(
            volume_data['E']*1e-3, volume_data['N']*1e-3,
            volume_data.mean('Z').T, **kwargs)

    if slice_N is not None:
        ax.axhline(slice_N*1e-3, color='k', alpha=0.5, lw=1, zorder=10)

    if slice_E is not None:
        ax.axvline(slice_E*1e-3, color='k', alpha=0.5, lw=1, zorder=10)

    if labels:
        ax.set_xlabel('E [km]')
        ax.set_ylabel('N [km]')
        
    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    if aspect is not None:
        ax.set_aspect(aspect)
        
    return ax

def plot_prior_model(
    PRIOR_DATA, SURFACE_DATA, VOLCANO_DATA,
    slice_E=0, slice_N=0,
    ):
    
    fig, ax_dict = plt.subplot_mosaic(
        [['prior_marginal_Z', 'prior_slice_E'],
        ['prior_marginal_Z', 'prior_slice_N',],], figsize=(10, 4), empty_sentinel=None, dpi=120, 
        gridspec_kw={'width_ratios': [4,6], 'height_ratios': [1, 1]}
        )

    plot_topography(
        ax_dict['prior_marginal_Z'], SURFACE_DATA['topography'])
    plot_marginal_Z(
        ax_dict['prior_marginal_Z'], PRIOR_DATA, SURFACE_DATA,
        slice_E=slice_E, slice_N=slice_N,
        cmap=blue_cmap,
        aspect=1)
    ax_dict['prior_marginal_Z'].set_title('Z marginal')
        
    plot_slice_E(
        ax_dict['prior_slice_E'], PRIOR_DATA, SURFACE_DATA, slice_E,
        cmap=blue_cmap,
        aspect='auto')
    # ax_dict['prior_slice_E'].set_title('E-W slice')
    
    ax_dict['prior_slice_E'].yaxis.set_label_position("right")
    ax_dict['prior_slice_E'].yaxis.set_tick_params(labelright=True, labelleft=False, right=True, left=True, which='both')
    ax_dict['prior_slice_E'].xaxis.set_label_position("top")
    ax_dict['prior_slice_E'].xaxis.tick_top()
    ax_dict['prior_slice_E'].xaxis.set_tick_params(labeltop=True, labelbottom=False, top=True, bottom=True, which='both')
    
    plot_slice_N(
        ax_dict['prior_slice_N'], PRIOR_DATA, SURFACE_DATA, slice_N,
        cmap=blue_cmap,
        aspect='auto')
    # ax_dict['prior_slice_N'].set_title('N-S slice')
    ax_dict['prior_slice_N'].yaxis.tick_right()
    ax_dict['prior_slice_N'].yaxis.set_label_position("right")
    ax_dict['prior_slice_N'].yaxis.set_tick_params(labelright=True, labelleft=False, right=True, left=True, which='both')

    ax_dict['prior_slice_N'].xaxis.tick_bottom()
    ax_dict['prior_slice_N'].xaxis.set_tick_params(labeltop=False, labelbottom=True, top=True, bottom=True, which='both')

    return fig, ax_dict

def plot_posterior_model(
    design, posterior_data, surface_data,
    true_event, std=None,
    E_lim=None, N_lim=None, Z_lim=None,
    show=True):
    
    fig, ax_dict = plt.subplot_mosaic(
        [['prior_marginal_Z', 'prior_slice_E'],
        ['prior_marginal_Z', 'prior_slice_N',],], figsize=(10, 4), empty_sentinel=None, dpi=120, 
        gridspec_kw={'width_ratios': [4 ,6], 'height_ratios': [1, 1]}
        )

    plot_topography(
        ax_dict['prior_marginal_Z'], surface_data['topography'])
    plot_marginal_Z(
        ax_dict['prior_marginal_Z'], posterior_data, surface_data, true_event[0], true_event[1],
        cmap='Reds', type='contour', levels=5, zorder=11,
        aspect=1, )
            
    ax_dict['prior_marginal_Z'].set_title('Z marginal')
    
    # dummy for legend
    ax_dict['prior_marginal_Z'].scatter(
        [], [], s=50, marker='o', linewidth=1.5, alpha=1,
        facecolors='none', edgecolors='darkred',label='posterior pdf'
        )
    
    if std is not None:
        circle = plt.Circle(
            (true_event[0]*1e-3, true_event[1]*1e-3), std*1e-3, color='k', fill=False,
            linestyle='-', linewidth=1.5, zorder=12)
        ax_dict['prior_marginal_Z'].add_artist(circle)
        
        # dummy for legend
        ax_dict['prior_marginal_Z'].scatter(
            [], [], s=50, marker='o',
            linewidth=1.5, alpha=1,
            facecolors='none', edgecolors='k', label='(approx) std'
        )

    for sta_type, sta_data in design:
        ax_dict['prior_marginal_Z'].scatter(
            sta_data[0]*1e-3,
            sta_data[1]*1e-3,
            s=75 if 'array' in sta_type else 100,
            marker='x' if 'array' in sta_type else '^',
            c='k',
            linewidth=2 if 'array' in sta_type else 0,
            alpha=1.0,
            label='array' if 'array' in sta_type else 'node'
        )
    
    if E_lim is not None:
        ax_dict['prior_marginal_Z'].set_xlim(E_lim)
    if N_lim is not None:
        ax_dict['prior_marginal_Z'].set_ylim(N_lim)
    
    ax_dict['prior_slice_E'].scatter(
        [],
        [],
        s=50,
        marker='*',
        c='black',
        linewidth=0,
        alpha=1.0,
        label='true event'
    )
            
    # remove duplicate labels
    handles, labels = ax_dict['prior_marginal_Z'].get_legend_handles_labels()
    by_label = dict(list(zip(labels, handles))[::-1])
    
    # plot in box below
    ax_dict['prior_marginal_Z'].legend(
        by_label.values(), by_label.keys(),
        facecolor='w', edgecolor='k',
        loc='upper center', bbox_to_anchor=(1.5, -0.15),
        ncol=5
    )                                       
        
    plot_slice_E(
        ax_dict['prior_slice_E'], posterior_data, surface_data, true_event[0],
        cmap='Reds', type='contour', levels=5, zorder=9,
        vmin=posterior_data.min()+0.1*(posterior_data.max()-posterior_data.min()),
        vmax=posterior_data.max(),
        aspect='auto')
    
    # ax_dict['prior_slice_E'].set_title('E-W slice')
    ax_dict['prior_slice_E'].yaxis.set_label_position("right")
    ax_dict['prior_slice_E'].yaxis.set_tick_params(labelright=True, labelleft=False, right=True, left=True, which='both')
    ax_dict['prior_slice_E'].xaxis.set_label_position("top")
    ax_dict['prior_slice_E'].xaxis.tick_top()
    ax_dict['prior_slice_E'].xaxis.set_tick_params(labeltop=True, labelbottom=False, top=True, bottom=True, which='both')

    if std is not None:
        circle = plt.Circle(
            (true_event[1]*1e-3, true_event[2]*1e-3), std*1e-3, color='black', fill=False,
            linewidth=1.5, zorder=10)
        ax_dict['prior_slice_E'].add_artist(circle)

    ax_dict['prior_slice_E'].scatter(
        true_event[1]*1e-3,
        true_event[2]*1e-3,
        s=50,
        marker='*',
        c='black',
        linewidth=0,
        alpha=1.0
    )
    if E_lim is not None:
        ax_dict['prior_slice_E'].set_xlim(E_lim)
    if Z_lim is not None:
        ax_dict['prior_slice_E'].set_ylim(Z_lim)

    plot_slice_N(
        ax_dict['prior_slice_N'], posterior_data, surface_data, true_event[1],
        cmap='Reds', type='contour', levels=5, zorder=9,
        vmin=posterior_data.min()+0.1*(posterior_data.max()-posterior_data.min()),
        vmax=posterior_data.max(),
        aspect='auto')
    
    # ax_dict['prior_slice_N'].set_title('N-S slice')
    ax_dict['prior_slice_N'].yaxis.tick_right()
    ax_dict['prior_slice_N'].yaxis.set_label_position("right")
    ax_dict['prior_slice_N'].yaxis.set_tick_params(labelright=True, labelleft=False, right=True, left=True, which='both')

    ax_dict['prior_slice_N'].xaxis.tick_bottom()
    ax_dict['prior_slice_N'].xaxis.set_tick_params(labeltop=False, labelbottom=True, top=True, bottom=True, which='both')
    
    if std is not None:
        circle = plt.Circle(
            (true_event[0]*1e-3, true_event[2]*1e-3), std*1e-3, color='black', fill=False,
            linewidth=1.5, zorder=10)
        ax_dict['prior_slice_N'].add_artist(circle)

    ax_dict['prior_slice_N'].scatter(
        true_event[0]*1e-3,
        true_event[2]*1e-3,
        s=50,
        marker='*',
        c='black',
        linewidth=0,
        alpha=1.0
    )
    if N_lim is not None:
        ax_dict['prior_slice_N'].set_xlim(N_lim)
    if Z_lim is not None:
        ax_dict['prior_slice_N'].set_ylim(Z_lim)

    if show:
        plt.show()
    
    return fig, ax_dict


def plot_design_space_dict(
    design_space_dict, SURFACE_DATA, VOLCANO_DATA, show=True):
    
    # fig, ax_list = plt.subplots(2, 1, figsize=(4, 8), sharex=True, sharey=True)
    fig, ax_list = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    for ax, (ds_name, ds) in zip(ax_list, design_space_dict.items()):

        plot_topography(ax, SURFACE_DATA['topography'])

        ax.imshow(
            ds.values.T,
            cmap=binary_cmap,
            extent=[
                SURFACE_DATA.E.min() * 1e-3 - SURFACE_DATA.E.mean() * 1e-3,
                SURFACE_DATA.E.max() * 1e-3 - SURFACE_DATA.E.mean() * 1e-3,
                SURFACE_DATA.N.min() * 1e-3 - SURFACE_DATA.N.mean() * 1e-3,
                SURFACE_DATA.N.max() * 1e-3 - SURFACE_DATA.N.mean() * 1e-3,
            ],
            origin='lower',
            vmin=0, vmax=1, alpha=0.7
        )

        ax.set_title(f"{VOLCANO_DATA['Volcano Name']}: design space ({ds_name})")

    for i, ax in enumerate(ax_list):
        ax.set_xlabel('E [km]')

        if i == 0:
            ax.set_ylabel('N [km]')

            # a red and a white filled rectangle for not allowed and allowed areas
            handels = [
                plt.Rectangle((0, 0), 1, 1, facecolor='white', alpha=0.7, edgecolor='k'),
                plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, edgecolor='k'),
            ]
            labels = ['allowed', 'not allowed']
            ax.legend(handels, labels, loc='upper left', facecolor='w', edgecolor='k')

    if show:   
        plt.show()

    return fig, ax_list

def plot_design(
    ax, design,
    **kwargs):
    
    unique_types = []

    if ('color' not in kwargs):
        
        print('No color specified, using black')
        kwargs['color'] = 'k'
        
    for sta_type, sta_data in design:
        ax.scatter(
            sta_data[0]*1e-3,
            sta_data[1]*1e-3,
            s=150 if 'array' in sta_type else 200,
            marker='x' if 'array' in sta_type else '^',
            linewidth=4 if 'array' in sta_type else 0,
            **kwargs
        )
        for data_type in sta_type:
            if data_type not in unique_types:
                unique_types.append(data_type)
    
    if 'tt' in sta_type or 'asl' in unique_types:
        ax.scatter(
            [], [], label='broadband', c='k',
            s=120,
            marker='^',
            linewidth=0,
        )
    if 'array' in unique_types:
        ax.scatter(
            [], [], label='array', c='k',
            s=70,
            marker='x',
            linewidth=2,
        )
    # white background
    ax.legend(
        facecolor='w',
        edgecolor='k',
    )
    
    return ax


def interactive_design_plot(
    **kwargs
    ):

    posterior = kwargs.get('posterior', False)
    
    if posterior:
        return interactive_design_plot_posterior(**kwargs)
    else:
        return interactive_design_plot_plain(**kwargs)

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


def interactive_design_plot_plain(
    original_design,
    volcano_data,
    surface_data,
    design_space_dict,
    eig_criterion,
    prior_information,
    **kwargs
    ):
        
    surface_data = add_gradient(surface_data)
    
    changing_design = list(original_design.copy())
    changing_design = [[sta_type, np.array(sta_data)] for sta_type, sta_data in changing_design]

    fig = bqp_plt.figure(figsize=(12, 12), title=f'{volcano_data["Volcano Name"]}: optimal design', min_aspect_ratio=1, max_aspect_ratio=1,
                        #  legend_location="top-right", legend_style={"fill": "white", "stroke": "black"}
    )
    
    fx = interp1d(np.arange(0, surface_data['E'].values.shape[0]), surface_data['E'].values*1e-3)
    fy = interp1d(np.arange(0, surface_data['N'].values.shape[0]), surface_data['N'].values*1e-3)
    
    bqp_plt.heatmap(
        surface_data['topography'].T,
        x=surface_data['E']*1e-3,
        y=surface_data['N']*1e-3,
        scales={'x': bqp_plt.LinearScale(min=surface_data['E'].values.min()*1e-3, max=surface_data['E'].values.max()*1e-3),
                'y': bqp_plt.LinearScale(min=surface_data['N'].values.min()*1e-3, max=surface_data['N'].values.max()*1e-3),
                'color': ColorScale(scheme='Greys')},
        axes_options = {"color": dict(label='z [m]', orientation="horizontal", side="bottom",visible=False),
                        'x': dict(label='E [km]', orientation="horizontal", side="bottom"),
                        'y': dict(label='N [km]', orientation="vertical", side="left"),
                        },
        display_legend=False,
    )
    
    scat_list = []
    unique_types = []
    for i, (sta_type, sta_data) in enumerate(changing_design):
        scat = bqp_plt.scatter(
            [sta_data[0]*1e-3],
            [sta_data[1]*1e-3],
            default_size=300 if 'array' in sta_type else 150,
            marker='crosshair' if 'array' in sta_type else 'triangle-up',
            colors=["black"],
            stroke_width=5 if 'array' in sta_type else 0,
            enable_move=True,
            display_legend=False,
            names=i,
            display_names=False,
        )
        for data_type in sta_type:
            if data_type not in unique_types:
                unique_types.append(data_type)
        scat_list.append(scat)
        
    if 'tt' in unique_types or 'asl' in sta_type:
        bqp_plt.plot(
            [sta_data[0]*1e-3, sta_data[0]*1e-3],
            [sta_data[1]*1e-3, sta_data[1]*1e-3],
            colors=["darkred"],
            stroke_width=5,
        )
        ds_nodes_cs = measure.find_contours(design_space_dict['node'].values, 0.5)
        
        for contour in ds_nodes_cs:
            contour[:,0] = fx(contour[:,0])
            contour[:,1] = fy(contour[:,1])
                
        for c in ds_nodes_cs:
            bqp_plt.plot(c[:,0], c[:,1], colors=['purple'], stroke_width=1, opacity=0.5)
        
    if 'array' in unique_types:
        bqp_plt.scatter(
            [], [], labels='array', colors=["black"],
            s=70,
            marker='cross',
            linewidth=2,
        )
        ds_array_cs = measure.find_contours(design_space_dict['array'].values, 0.5)
        
        for contour in ds_array_cs:
            contour[:,0] = fx(contour[:,0])
            contour[:,1] = fy(contour[:,1])

        for c in  ds_array_cs:
            bqp_plt.plot(c[:,0], c[:,1], colors=['red'], stroke_width=1, opacity=0.5)

    eig = eig_criterion(changing_design)        
    post_information = eig + prior_information

    k = 3
    std = np.exp((post_information + k/2 + k/2 * np.log(2*np.pi)) / (-0.5 * k * 2))

    label  = bqp_plt.label(
        text=['Design Statistics:',
            f'EIG: {eig:.3f} nats',
            f'Approx std: {std:.0f} m',
            '',
            ],
        x=[surface_data['E'].max()*1e-3 + 0.1 * (surface_data['E'].max()*1e-3 - surface_data['E'].min()*1e-3),
           surface_data['E'].max()*1e-3 + 0.1 * (surface_data['E'].max()*1e-3 - surface_data['E'].min()*1e-3),
           surface_data['E'].max()*1e-3 + 0.1 * (surface_data['E'].max()*1e-3 - surface_data['E'].min()*1e-3),
           surface_data['E'].max()*1e-3 + 0.1 * (surface_data['E'].max()*1e-3 - surface_data['E'].min()*1e-3),
        ],
        y=[surface_data['N'].max()*1e-3 - 0.1 * (surface_data['N'].max()*1e-3 - surface_data['N'].min()*1e-3),
           surface_data['N'].max()*1e-3 - 0.2 * (surface_data['N'].max()*1e-3 - surface_data['N'].min()*1e-3),
           surface_data['N'].max()*1e-3 - 0.3 * (surface_data['N'].max()*1e-3 - surface_data['N'].min()*1e-3),
           surface_data['N'].max()*1e-3 - 0.4 * (surface_data['N'].max()*1e-3 - surface_data['N'].min()*1e-3)
        ],
        colors=['black', 'black', 'black', 'red'],
        apply_clip=False
        )
    def update_line(_scatter , event):
    
        # with scat.hold_sync():
        label.text = ['Design Statistics:',
            '',
            '',
            'running...',
            ]
        
        index = _scatter.names.item()
        sta_typ = 'array' if 'array' in changing_design[index][0] else 'node'
                
        new_location = np.array([event['point']['x']*1e3, event['point']['y']*1e3])
        new_elevation = get_elevation(new_location*1e-3, surface_data)
        new_location = np.append(new_location, new_elevation)
        
        if sta_typ == 'array':
            # gradient at new location
            new_gradient = (surface_data['grad_E'].interp(E=new_location[0], N=new_location[1]).values.item(),
                            surface_data['grad_N'].interp(E=new_location[0], N=new_location[1]).values.item())        
            new_normal_vector = np.array([-new_gradient[0], -new_gradient[1], 1])
            new_normal_vector /= np.linalg.norm(new_normal_vector)
            new_location = np.hstack((new_location, new_normal_vector))
            
        # check if new location is in design space
        in_ds = design_space_dict[sta_typ].sel(E=new_location[0], N=new_location[1], method='nearest').values
        
        changing_design[index][1] = new_location
        torch.manual_seed(0)
        eig = eig_criterion(changing_design)
        
        post_information = eig + prior_information
        k = 3
        std = np.exp((post_information + k/2 + k/2 * np.log(2*np.pi)) / (-0.5 * k * 2))
        
        label.text = ['Design Statistics:',
                    f'EIG: {eig:.3f} nats',
                    f'Approx std: {std:.0f} m',
                    '' if in_ds else 'Current location is not in design space',
                    ]

    for i, scat in enumerate(scat_list):
        scat.on_drag_end(update_line)
        
    
    return fig, changing_design, None


def interactive_design_plot_posterior(
    original_design,
    volcano_data,
    surface_data,
    prior_data,
    design_space_dict,
    eig_criterion,
    prior_information,
    forward_function,
    **kwargs
    ):

    from helpers.posterior_helpers import calculate_posterior
    import ipywidgets as widgets

    surface_data = add_gradient(surface_data)

    z_initial = np.average(prior_data.Z.values, weights=prior_data.mean(['E', 'N']).values)
    
    # slider for z position of source
    z_slider = widgets.FloatSlider(
        value=z_initial,
        min=prior_data.Z.min(),
        max=prior_data.Z.max(),
        step=(prior_data.Z.max() - prior_data.Z.min()) / prior_data.Z.shape[0],
        description='Z [m]',
        continuous_update=False,
    )

    initial_model = np.array([0.0, 0.0, np.average(prior_data.Z.values, weights=prior_data.mean(['E', 'N']).values)])

    _, initial_posterior = calculate_posterior(
        original_design,
        initial_model,
        prior_data,
        forward_function
    )
    
    fx_post = interp1d(np.arange(0, initial_posterior['E'].values.shape[0]), initial_posterior['E'].values*1e-3)
    fx_post = interp1d(np.arange(0, initial_posterior['N'].values.shape[0]), initial_posterior['N'].values*1e-3)
    
    fx = interp1d(np.arange(0, surface_data['E'].values.shape[0]), surface_data['E'].values*1e-3)
    fy = interp1d(np.arange(0, surface_data['N'].values.shape[0]), surface_data['N'].values*1e-3)

    changing_design = list(original_design.copy())
    changing_design = [[sta_type, np.array(sta_data)] for sta_type, sta_data in changing_design]

    fig = bqp_plt.figure(figsize=(12, 12), title=f'{volcano_data["Volcano Name"]}: optimal design', min_aspect_ratio=1, max_aspect_ratio=1,
                        #  legend_location="top-right", legend_style={"fill": "white", "stroke": "black"}
    )
    
    bqp_plt.heatmap(
        surface_data['topography'].T,
        x=surface_data['E']*1e-3,
        y=surface_data['N']*1e-3,
        scales={'x': bqp_plt.LinearScale(min=surface_data['E'].values.min()*1e-3, max=surface_data['E'].values.max()*1e-3),
                'y': bqp_plt.LinearScale(min=surface_data['N'].values.min()*1e-3, max=surface_data['N'].values.max()*1e-3),
                'color': ColorScale(scheme='Greys')},
        axes_options = {"color": dict(label='z [m]', orientation="vertical", side="right",visible=True),
                        'x': dict(label='E [km]', orientation="horizontal", side="bottom"),
                        'y': dict(label='N [km]', orientation="vertical", side="left"),
                        },
        display_legend=False,
    )
    
    posterior_plots = []
    
    # post_slice = initial_posterior.mean('Z')
    post_slice_initial = initial_posterior.sel(Z=initial_model[2], method='nearest')
    
    post_diff = post_slice_initial.max().values - post_slice_initial.min().values
    levels = [0.1, 0.5, 0.9]
    
    for i, level in enumerate(
        [post_slice_initial.min().values + levels[i] * post_diff for i in range(len(levels))]):
        
        post_contour = measure.find_contours(post_slice_initial.values, level)
        
        for contour in post_contour:
            contour[:,0] = fx_post(contour[:,0])
            contour[:,1] = fx_post(contour[:,1])
            
        for j, c in enumerate(post_contour):
            posterior_plots.append(
                bqp_plt.plot(c[:,0], c[:,1], colors=['red'],
                stroke_width=2,
                opacity=[0.9, 0.5, 0.1][i]
                ))
        
    scat_list = []
    unique_types = []
    for i, (sta_type, sta_data) in enumerate(changing_design):
        scat = bqp_plt.scatter(
            [sta_data[0]*1e-3],
            [sta_data[1]*1e-3],
            default_size=300 if 'array' in sta_type else 150,
            marker='crosshair' if 'array' in sta_type else 'triangle-up',
            colors=["black"],
            stroke_width=5 if 'array' in sta_type else 0,
            enable_move=True,
            display_legend=False,
            names=i,
            display_names=False,
        )
        for data_type in sta_type:
            if data_type not in unique_types:
                unique_types.append(data_type)
        scat_list.append(scat)
    
    
    src_scat = bqp_plt.scatter(
        [initial_model[0]*1e-3],
        [initial_model[1]*1e-3],
        default_size=50,
        marker='diamond',
        colors=["red"],
        stroke_width=0,
        enable_move=True,
        display_legend=False,
        names='src',
        display_names=False,
    )
    scat_list.append(src_scat)
    
    if 'tt' in unique_types or 'asl' in sta_type:
        bqp_plt.plot(
            [sta_data[0]*1e-3, sta_data[0]*1e-3],
            [sta_data[1]*1e-3, sta_data[1]*1e-3],
            colors=["darkred"],
            stroke_width=5,
        )
        ds_nodes_cs = measure.find_contours(design_space_dict['node'].values, 0.5)

        for contour in ds_nodes_cs:
            contour[:,0] = fx(contour[:,0])
            contour[:,1] = fy(contour[:,1])
                
        for c in ds_nodes_cs:
            bqp_plt.plot(c[:,0], c[:,1], colors=['gray'], stroke_width=1, opacity=0.5)
        
    if 'array' in unique_types:
        bqp_plt.scatter(
            [], [], labels='array', colors=["black"],
            s=70,
            marker='cross',
            linewidth=2,
        )
        ds_array_cs = measure.find_contours(design_space_dict['array'].values, 0.5)
        
        for contour in ds_array_cs:
            contour[:,0] = fx(contour[:,0])
            contour[:,1] = fy(contour[:,1])

        for c in  ds_array_cs:
            bqp_plt.plot(c[:,0], c[:,1], colors=['gray'], stroke_width=1, opacity=0.5)


    eig = eig_criterion(changing_design)        
    post_information = eig + prior_information

    k = 3
    std = np.exp((post_information + k/2 + k/2 * np.log(2*np.pi)) / (-0.5 * k * 2))

    label  = bqp_plt.label(
        text=['Design Statistics:',
            f'EIG: {eig:.3f} nats',
            f'Approx std: {std:.0f} m',
            '',
            ],
        x=[surface_data['E'].max()*1e-3 + 0.4 * (surface_data['E'].max()*1e-3 - surface_data['E'].min()*1e-3),
           surface_data['E'].max()*1e-3 + 0.4 * (surface_data['E'].max()*1e-3 - surface_data['E'].min()*1e-3),
           surface_data['E'].max()*1e-3 + 0.4 * (surface_data['E'].max()*1e-3 - surface_data['E'].min()*1e-3),
           surface_data['E'].max()*1e-3 + 0.4 * (surface_data['E'].max()*1e-3 - surface_data['E'].min()*1e-3),
        ],
        y=[surface_data['N'].max()*1e-3 - 0.1 * (surface_data['N'].max()*1e-3 - surface_data['N'].min()*1e-3),
           surface_data['N'].max()*1e-3 - 0.2 * (surface_data['N'].max()*1e-3 - surface_data['N'].min()*1e-3),
           surface_data['N'].max()*1e-3 - 0.3 * (surface_data['N'].max()*1e-3 - surface_data['N'].min()*1e-3),
           surface_data['N'].max()*1e-3 - 0.4 * (surface_data['N'].max()*1e-3 - surface_data['N'].min()*1e-3),
        ],
        colors=['black', 'black', 'black', 'red'],
        apply_clip=False
        )
    
    def update_line(_scatter , event):
    
        # with scat.hold_sync():
        
        # label.text[-1] = 'running...'
        label.text = ['Design Statistics:',
            '',
            '',
            'running...',
            ]
        
        index = _scatter.names.item()
        
        if index == 'src':
            initial_model[0] = event['point']['x']*1e3
            initial_model[1] = event['point']['y']*1e3            
        else:
            sta_typ = 'array' if 'array' in changing_design[index][0] else 'node'
                    
            new_location = np.array([event['point']['x']*1e3, event['point']['y']*1e3])
            new_elevation = get_elevation(new_location*1e-3, surface_data)
            new_location = np.append(new_location, new_elevation)
            
            if sta_typ == 'array':
                # gradient at new location
                new_gradient = (surface_data['grad_E'].interp(E=new_location[0], N=new_location[1]).values.item(),
                                surface_data['grad_N'].interp(E=new_location[0], N=new_location[1]).values.item())        
                new_normal_vector = np.array([-new_gradient[0], -new_gradient[1], 1])
                new_normal_vector /= np.linalg.norm(new_normal_vector)
                new_location = np.hstack((new_location, new_normal_vector))

            changing_design[index][1] = new_location

        for sta_type, sta_data in changing_design:                
            if 'array' in sta_type:
                in_ds = design_space_dict['array'].sel(E=sta_data[0], N=sta_data[1], method='nearest').values
                if not in_ds:
                    break
                
            if 'tt' in sta_type or 'asl' in sta_type:
                in_ds = design_space_dict['node'].sel(E=sta_data[0], N=sta_data[1], method='nearest').values
                
                if not in_ds:
                    break
                                                
        torch.manual_seed(0)
        eig = eig_criterion(changing_design)
        
        post_information = eig + prior_information
        k = 3
        std = np.exp((post_information + k/2 + k/2 * np.log(2*np.pi)) / (-0.5 * k * 2))
        
        _, new_posterior = calculate_posterior(
            changing_design,
            initial_model,
            prior_data,
            forward_function
        )
        
        # post_slice = new_posterior.mean('Z')
        post_slice = new_posterior.sel(Z=initial_model[2], method='nearest')

        post_diff = post_slice.max().values - post_slice.min().values

        
        for i, level in enumerate(
            [post_slice.min().values + levels[i] * post_diff for i in range(len(levels))]):
            
            post_contour = measure.find_contours(post_slice.values, level)

            for contour in post_contour:
                contour[:,0] = fx_post(contour[:,0])
                contour[:,1] = fx_post(contour[:,1])

            for j, c in enumerate(post_contour):
                # posterior_plots[i*j].x = c[:,0]
                # posterior_plots[i*j].y = c[:,1]
                posterior_plots[j*len(levels) + i].x = c[:,0]
                posterior_plots[j*len(levels) + i].y = c[:,1]
                
        label.text = ['Design Statistics:',
                    f'EIG: {eig:.3f} nats',
                    f'Approx std: {std:.0f} m',
                    '' if in_ds else 'Current location is not in design space',
                    ]

    def slider_update(change):
        initial_model[2] = z_slider.value
        
        label.text = ['Design Statistics:',
            '',
            '',
            'running...',
            ]
        
        for sta_type, sta_data in changing_design:                
            if 'array' in sta_type:
                in_ds = design_space_dict['array'].sel(E=sta_data[0], N=sta_data[1], method='nearest').values
                if not in_ds:
                    break
                
            if 'tt' in sta_type or 'asl' in sta_type:
                in_ds = design_space_dict['node'].sel(E=sta_data[0], N=sta_data[1], method='nearest').values
                
                if not in_ds:
                    break    
    
        torch.manual_seed(0)
        eig = eig_criterion(changing_design)
        
        post_information = eig + prior_information
        k = 3
        std = np.exp((post_information + k/2 + k/2 * np.log(2*np.pi)) / (-0.5 * k * 2))
        
        _, new_posterior = calculate_posterior(
            changing_design,
            initial_model,
            prior_data,
            forward_function
        )
        
        post_slice = new_posterior.sel(Z=initial_model[2], method='nearest')

        post_diff = post_slice.max().values - post_slice.min().values

        for i, level in enumerate(
            [post_slice.min().values + levels[i] * post_diff for i in range(len(levels))]):
            
            post_contour = measure.find_contours(post_slice.values, level)

            for contour in post_contour:
                contour[:,0] = fx_post(contour[:,0])
                contour[:,1] = fx_post(contour[:,1])

            for j, c in enumerate(post_contour):
                posterior_plots[j*len(levels) + i].x = c[:,0]
                posterior_plots[j*len(levels) + i].y = c[:,1]
                
        label.text = ['Design Statistics:',
                    f'EIG: {eig:.3f} nats',
                    f'Approx std: {std:.0f} m',
                    '' if in_ds else 'Current location is not in design space',
                    ]

    for i, scat in enumerate(scat_list):
        scat.on_drag_end(update_line)
        
    z_slider.observe(slider_update, names='value')

    return widgets.VBox([fig, z_slider]) , changing_design, initial_model