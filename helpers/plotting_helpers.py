import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
from skimage import measure
from scipy.interpolate import interp1d

import bqplot.pyplot as bqp_plt
from bqplot import ColorScale

from helpers.vulcano_data_helpers import get_elevation

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

def plot_topography(ax, topo_ds):
    ax.imshow(
        topo_ds.values.T,
        cmap="Greys",
        extent=[
            topo_ds.E.min() * 1e-3 - topo_ds.E.mean() * 1e-3,
            topo_ds.E.max() * 1e-3 - topo_ds.E.mean() * 1e-3,
            topo_ds.N.min() * 1e-3 - topo_ds.N.mean() * 1e-3,
            topo_ds.N.max() * 1e-3 - topo_ds.N.mean() * 1e-3,
        ],
        origin='lower',
        alpha=0.8,
    )
    
    ax.pc = ax.contour(
        topo_ds.E * 1e-3 - topo_ds.E.mean() * 1e-3,
        topo_ds.N * 1e-3 - topo_ds.N.mean() * 1e-3,
        topo_ds.values.T,
        colors="k",
        zorder=-10,
        # levels divisions of 500 from min to max
        levels=np.arange(
            topo_ds.values.min().round(-3), (topo_ds.values.max() + 1000).round(-3), 500
        ),
        linewidths=1.0,
        alpha=1.0,
    )

    ax.cl = ax.clabel(
        ax.pc,
        levels=np.arange(
            topo_ds.values.min().round(-3), (topo_ds.values.max() + 1000).round(-3), 500
        ),
        inline=True,
        fontsize=8,
        fmt="%1.0f",
        colors="k",
        use_clabeltext=True,
    )

    return ax

def plot_slice_N(
    ax, volume_data, surface_data, slice_N, aspect=1,
    colorbar=False,
    labels=True,
    type='imshow',
    **kwargs):
    
    if type == 'imshow':
        im = ax.imshow(volume_data.sel(
            N=slice_N, method='nearest').T, origin='lower', aspect=1,
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
        
    ax.set_aspect(aspect)

    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.6)

    return ax
    
def plot_slice_E(
    ax, volume_data, surface_data, slice_E, aspect=1,
    colorbar=False,
    labels=True, type='imshow',
    **kwargs):
    
    if type == 'imshow':
        im = ax.imshow(volume_data.sel(
            E=slice_E, method='nearest').T, origin='lower', aspect=1,
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
    
    if type == 'imshow':
        im = ax.imshow(
            volume_data.mean('Z').T, origin='lower', aspect=1,
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
        
    return ax

def plot_prior_model(PRIOR_DATA, SURFACE_DATA, VULCANO_DATA, show=True):
    
    fig, ax_dict = plt.subplot_mosaic(
        [['prior_marginal_Z', 'prior_slice_E'],
        ['prior_marginal_Z', 'prior_slice_N',],], figsize=(8, 4), empty_sentinel=None, dpi=120
        )

    plot_topography(
        ax_dict['prior_marginal_Z'], SURFACE_DATA['topography'])
    plot_marginal_Z(
        ax_dict['prior_marginal_Z'], PRIOR_DATA, SURFACE_DATA, 0.0, 0.0, cmap=blue_cmap)
    ax_dict['prior_marginal_Z'].set_title('Z marginal')

    fig.suptitle(f'{VULCANO_DATA["Volcano Name"]}: prior model', fontsize=12)
        
    plot_slice_E(
        ax_dict['prior_slice_E'], PRIOR_DATA, SURFACE_DATA, 0.0,
        aspect=2, cmap=blue_cmap)
    ax_dict['prior_slice_E'].set_title('E-W slice')
    ax_dict['prior_slice_E'].yaxis.tick_right()
    ax_dict['prior_slice_E'].yaxis.set_label_position("right")

    plot_slice_N(
        ax_dict['prior_slice_N'], PRIOR_DATA, SURFACE_DATA, 0.0,
        aspect=2, cmap=blue_cmap)
    ax_dict['prior_slice_N'].set_title('N-S slice')
    ax_dict['prior_slice_N'].yaxis.tick_right()
    ax_dict['prior_slice_N'].yaxis.set_label_position("right")

    if show:
        plt.show()
    
    return fig, ax_dict

def plot_posterior_model(
    design, posterior_data, surface_data, vulcano_data,
    true_event, std=None, show=True):
    
    fig, ax_dict = plt.subplot_mosaic(
        [['prior_marginal_Z', 'prior_slice_E'],
        ['prior_marginal_Z', 'prior_slice_N',],], figsize=(8, 4), empty_sentinel=None, dpi=120
        )

    plot_topography(
        ax_dict['prior_marginal_Z'], surface_data['topography'])
    plot_marginal_Z(
        ax_dict['prior_marginal_Z'], posterior_data, surface_data, true_event[0], true_event[1],
        cmap='Reds', type='contour', levels=3, zorder=11,
        # vmin=posterior_data.min()+0.1*(posterior_data.max()-posterior_data.min()),
        # vmax=posterior_data.max()
        )
            
    ax_dict['prior_marginal_Z'].set_title('Z marginal')
    
    # dummy for legend
    ax_dict['prior_marginal_Z'].scatter(
        [], [], s=50, marker='o', linewidth=1.5, alpha=1,
        facecolors='darkred', edgecolors='darkred',label='posterior pdf'
        )
    
    if std is not None:
        circle = plt.Circle(
            (true_event[0]*1e-3, true_event[1]*1e-3), std*1e-3, color='k', fill=False,
            linestyle='-', linewidth=0.5, zorder=9)
        ax_dict['prior_marginal_Z'].add_artist(circle)
        
        # dummy for legend
        ax_dict['prior_marginal_Z'].scatter(
            [], [], s=50, marker='o',
            linewidth=1, alpha=1,
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
            alpha=0.5,
            label='array' if 'array' in sta_type else 'node'
        )
    
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
        loc='upper center', bbox_to_anchor=(1.0, -0.1),
        ncol=5
    )                                       
                
    fig.suptitle(f'{vulcano_data["Volcano Name"]}: prior model', fontsize=12)
        
    plot_slice_E(
        ax_dict['prior_slice_E'], posterior_data, surface_data, true_event[0],
        aspect=2, cmap='Reds', type='contour', levels=3, zorder=9,
        vmin=posterior_data.min()+0.1*(posterior_data.max()-posterior_data.min()),
        vmax=posterior_data.max())
    
    ax_dict['prior_slice_E'].set_title('E-W slice')
    ax_dict['prior_slice_E'].yaxis.tick_right()
    ax_dict['prior_slice_E'].yaxis.set_label_position("right")

    if std is not None:
        circle = plt.Circle(
            (true_event[1]*1e-3, true_event[2]*1e-3), std*1e-3, color='black', fill=False,
            linewidth=0.5, zorder=10)
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

    plot_slice_N(
        ax_dict['prior_slice_N'], posterior_data, surface_data, true_event[1],
        aspect=2, cmap='Reds', type='contour', levels=3, zorder=9,
        vmin=posterior_data.min()+0.1*(posterior_data.max()-posterior_data.min()),
        vmax=posterior_data.max())
    
    ax_dict['prior_slice_N'].set_title('N-S slice')
    ax_dict['prior_slice_N'].yaxis.tick_right()
    ax_dict['prior_slice_N'].yaxis.set_label_position("right")

    if std is not None:
        circle = plt.Circle(
            (true_event[0]*1e-3, true_event[2]*1e-3), std*1e-3, color='black', fill=False,
            linewidth=0.5, zorder=10)
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

    if show:
        plt.show()
    
    return fig, ax_dict


def plot_design_space_dict(
    design_space_dict, SURFACE_DATA, VULCANO_DATA):
    
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
            vmin=0, vmax=1, alpha=0.5
        )

        ax.set_title(f"{VULCANO_DATA['Volcano Name']}: design space ({ds_name})")

    for i, ax in enumerate(ax_list):
        ax.set_xlabel('E [km]')
        
        if i == 0:
            ax.set_ylabel('N [km]')

    plt.show()

    return fig, ax_list

def plot_design(ax, design):
    
    unique_types = []

    for sta_type, sta_data in design:
        ax.scatter(
            sta_data[0]*1e-3,
            sta_data[1]*1e-3,
            s=150 if 'array' in sta_type else 200,
            marker='x' if 'array' in sta_type else '^',
            c='k',
            linewidth=4 if 'array' in sta_type else 0
        )
        for data_type in sta_type:
            if data_type not in unique_types:
                unique_types.append(data_type)
        
        
    if 'tt' in sta_type or 'asl' in sta_type:
        ax.scatter(
            [], [], label='broadband', c='k',
            s=120,
            marker='^',
            linewidth=0,
        )
    if 'array' in sta_type:
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
    original_design,
    vulcano_data,
    surface_data,
    design_space_dict,
    eig_criterion,
    prior_information
    ):
    
    changing_design = list(original_design.copy())
    changing_design = [[sta_type, np.array(sta_data)] for sta_type, sta_data in changing_design]

    fig = bqp_plt.figure(figsize=(12, 12), title=f'{vulcano_data["Volcano Name"]}: optimal design', min_aspect_ratio=1, max_aspect_ratio=1,
                        #  legend_location="top-right", legend_style={"fill": "white", "stroke": "black"}
    )
    
    
    bqp_plt.heatmap(
        surface_data['topography'].T,
        x=surface_data['E']*1e-3,
        y=surface_data['N']*1e-3,
        scales={'x': bqp_plt.LinearScale(min=-20, max=20),
                'y': bqp_plt.LinearScale(min=-20, max=20),
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
        fx = interp1d(np.arange(0, design_space_dict['node']['E'].values.shape[0]), design_space_dict['node']['E'].values*1e-3)
        fy = interp1d(np.arange(0, design_space_dict['node']['N'].values.shape[0]), design_space_dict['node']['N'].values*1e-3)
        
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
        fx = interp1d(np.arange(0, design_space_dict['array']['E'].values.shape[0]), design_space_dict['array']['E'].values*1e-3)
        fy = interp1d(np.arange(0, design_space_dict['array']['N'].values.shape[0]), design_space_dict['array']['N'].values*1e-3)
        
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
    
        with scat.hold_sync():
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
        
    
    return fig, changing_design
    