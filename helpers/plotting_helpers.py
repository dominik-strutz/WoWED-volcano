import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from skimage import measure
from scipy.interpolate import interp1d

import bqplot.pyplot as bqp_plt
from bqplot import ColorScale
from ipyleaflet import Map, basemaps, Rectangle
from ipywidgets import Layout

import utm

from helpers.volcano_data_helpers import get_elevation

blue_cmap = plt.cm.Blues
blue_cmap = blue_cmap(np.arange(blue_cmap.N))
blue_cmap[:, -1] = np.linspace(0, 1, blue_cmap.shape[0]) * 0.8
blue_cmap[:, :3] = [0.0, 0.0, 1.0]
blue_cmap = ListedColormap(blue_cmap)

red_cmap = plt.cm.Reds
red_cmap = red_cmap(np.arange(red_cmap.N))
red_cmap[:, -1] = np.linspace(0, 1, red_cmap.shape[0]) * 1.0
red_cmap[:, :3] = [1.0, 0.0, 0.0]
red_cmap = ListedColormap(red_cmap)

binary_cmap = ListedColormap([(0.6, 0, 0, 1), (0, 0, 0, 0)])


def _bb_km2latlon(bounding_box, center):
    """
    Convert bounding box in km to lat/lon coordinates.

    Parameters
    ----------
    bounding_box : dict
        Dictionary with the keys 'extent_south', 'extent_north', 'extent_west', 'extent_east'
        and the values in km.
    center : tuple
        Tuple with the lat/lon coordinates of the center of the bounding box.

    Returns
    -------
    dict
        Dictionary with the keys 'min_lat', 'max_lat', 'min_lon', 'max_lon' and the values in degrees.
    """
    center_utm = utm.from_latlon(center[0], center[1])

    min_E = center_utm[0] - bounding_box["extent_west"] * 1e3
    max_E = center_utm[0] + bounding_box["extent_east"] * 1e3
    min_N = center_utm[1] - bounding_box["extent_south"] * 1e3
    max_N = center_utm[1] + bounding_box["extent_north"] * 1e3

    min_lat, min_lon = utm.to_latlon(min_E, min_N, center_utm[2], center_utm[3])
    max_lat, max_lon = utm.to_latlon(max_E, max_N, center_utm[2], center_utm[3])

    return dict(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)


def draw_bounding_box(
    volcano_data,
    bounding_box=None,
    basemap=basemaps.Esri.WorldTopoMap,
):
    """
    Draw a bounding box on a map centered around a volcano. The function returns the map and the bounding box in lat/lon coordinates. The map can be used to interactively draw a bounding box.

    Parameters
    ----------
    volcano_data : dict
        Dictionary with the volcano data. The dictionary should have the keys 'lat' and 'lon' with the latitude and longitude of the volcano.
    bounding_box : dict, optional
        Dictionary with either the keys 'extent_south', 'extent_north', 'extent_west', 'extent_east' or 'min_lat', 'max_lat', 'min_lon', 'max_lon'. If the keys are 'extent_south', 'extent_north', 'extent_west', 'extent_east' the values should be in km. If the keys are 'min_lat', 'max_lat', 'min_lon', 'max_lon' the values should be in degrees. If None, a default bounding box of 20 km in each direction is used. The default is None.
    basemap : ipyleaflet.basemaps, optional
        Basemap to use for the map. The default is basemaps.Esri.WorldTopoMap.

    Returns
    -------
    ipyleaflet.Map, dict
        The map with the bounding box drawn on it and the bounding box in lat/lon coordinates.

    """
    center = (volcano_data["lat"], volcano_data["lon"])

    if bounding_box is not None:
        if all(
            [
                key in bounding_box.keys()
                for key in [
                    "extent_south",
                    "extent_north",
                    "extent_west",
                    "extent_east",
                ]
            ]
        ):
            bounding_box = _bb_km2latlon(bounding_box, center)
        elif all(
            [
                key in bounding_box.keys()
                for key in ["min_lat", "max_lat", "min_lon", "max_lon"]
            ]
        ):
            pass
        else:
            raise ValueError(
                'Invalid bounding box format. Keys should be either "extent_south", "extent_north", "extent_west", "extent_east" or "min_lat", "max_lat", "min_lon", "max_lon"'
            )
    else:
        bounding_box = dict(
            extent_south=20.0,  # in km
            extent_north=20.0,  # in km
            extent_west=20.0,  # in km
            extent_east=20.0,  # in km
        )
        bounding_box = _bb_km2latlon(bounding_box, center)

    defaultLayout = Layout(height="480px", width="960px")

    m = Map(
        basemap=basemap,
        center=center,
        # scroll_wheel_zoom=True,
        zoom_snap=0.2,
        layout=defaultLayout,
    )

    bound_rect = np.array(
        [
            [bounding_box["min_lat"], bounding_box["min_lon"]],
            [bounding_box["max_lat"], bounding_box["max_lon"]],
        ]
    )

    m.fit_bounds(bound_rect)

    m.add_layer(Rectangle(bounds=bound_rect.tolist(), color="black", fill_opacity=0.0))

    try:
        # try to use the geoman draw control, if not available use the deprecated draw control
        from ipyleaflet import GeomanDrawControl
        draw_control = GeomanDrawControl()
        draw_control_type = 'geoman'
    except ImportError:
        from ipyleaflet import DrawControl
        draw_control = DrawControl()
        draw_control_type = 'deprecated'
        
    draw_control.rectangle = {"shapeOptions": {"color": "black", "fillOpacity": 0.0}}
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circlemarker = {}

    draw_control.edit = False
    draw_control.drag = False
    draw_control.cut = False
    draw_control.rotate = False
    draw_control.remove = False

    m.add(draw_control)
        
    def handle_draw(self, action, geo_json):
        """Do something with the GeoJSON when it's drawn on the map"""

        if draw_control_type == 'deprecated':
            coordinates = geo_json["geometry"]["coordinates"][-1]
        else:
            coordinates = geo_json[-1]["geometry"]["coordinates"][0]
        
        bounding_box["min_lat"] = min([coord[1] for coord in coordinates])
        bounding_box["max_lat"] = max([coord[1] for coord in coordinates])
        bounding_box["min_lon"] = min([coord[0] for coord in coordinates])
        bounding_box["max_lon"] = max([coord[0] for coord in coordinates])
    
        #remove original rectangle layers but the map
        m.layers = m.layers[:1]
        
        draw_control.clear()
        m.add_layer(
            Rectangle(bounds=[[bounding_box["min_lat"], bounding_box["min_lon"]], [bounding_box["max_lat"], bounding_box["max_lon"]]], color="black", fill_opacity=0.0))
        
            

    draw_control.on_draw(handle_draw)

    return m, bounding_box


def plot_topography(ax, topo_array):
    """
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
    """

    E_min, E_max = topo_array.E.values.min(), topo_array.E.values.max()
    N_min, N_max = topo_array.N.values.min(), topo_array.N.values.max()

    ax.imshow(
        topo_array.values.T,
        cmap="Greys",
        extent=[
            E_min * 1e-3,
            E_max * 1e-3,
            N_min * 1e-3,
            N_max * 1e-3,
        ],
        origin="lower",
        alpha=0.8,
    )

    ax.pc = ax.contour(
        topo_array.E.values * 1e-3,
        topo_array.N.values * 1e-3,
        topo_array.values.T,
        colors="k",
        zorder=-10,
        # levels divisions of 500 from min to max
        levels=np.arange(
            topo_array.values.min().round(-3),
            (topo_array.values.max() + 1000).round(-3),
            500,
        ),
        linewidths=1.0,
        alpha=1.0,
    )

    ax.cl = ax.clabel(
        ax.pc,
        levels=np.arange(
            topo_array.values.min().round(-3),
            (topo_array.values.max() + 1000).round(-3),
            500,
        ),
        inline=True,
        fontsize=8,
        fmt="%1.0f",
        colors="k",
        use_clabeltext=True,
    )

    ax.set_aspect("equal")

    ax.set_xlabel("E [km]")
    ax.set_ylabel("N [km]")

    return ax


def plot_slice_N(
    ax,
    volume_data,
    surface_data,
    slice_N,
    colorbar=False,
    labels=True,
    type="imshow",
    **kwargs,
):
    aspect = kwargs.pop("aspect", "auto")

    if type == "imshow":
        im = ax.imshow(
            volume_data.sel(N=slice_N, method="nearest").T,
            origin="lower",
            extent=[
                volume_data["E"].min() * 1e-3,
                volume_data["E"].max() * 1e-3,
                volume_data["Z"].min() * 1e-3,
                volume_data["Z"].max() * 1e-3,
            ],
            **kwargs,
        )
    elif type == "contour":
        im = ax.contour(
            volume_data["E"] * 1e-3,
            volume_data["Z"] * 1e-3,
            volume_data.sel(N=slice_N, method="nearest").T,
            **kwargs,
        )
    elif type == "contourf":
        im = ax.contourf(
            volume_data["E"] * 1e-3,
            volume_data["Z"] * 1e-3,
            volume_data.sel(N=slice_N, method="nearest").T,
            **kwargs,
        )

    ax.plot(
        surface_data["E"] * 1e-3,
        surface_data["topography"].sel(N=slice_N, method="nearest") * 1e-3,
        "k",
        lw=1,
        zorder=20,
    )

    if labels:
        ax.set_xlabel("E [km]")
        ax.set_ylabel("Z [km]")

    if aspect is not None:
        ax.set_aspect(aspect)

    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.6)

    return ax


def plot_slice_E(
    ax,
    volume_data,
    surface_data,
    slice_E,
    colorbar=False,
    labels=True,
    type="imshow",
    **kwargs,
):
    aspect = kwargs.pop("aspect", "auto")

    if type == "imshow":
        im = ax.imshow(
            volume_data.sel(E=slice_E, method="nearest").T,
            origin="lower",
            extent=[
                volume_data["N"].min() * 1e-3,
                volume_data["N"].max() * 1e-3,
                volume_data["Z"].min() * 1e-3,
                volume_data["Z"].max() * 1e-3,
            ],
            **kwargs,
        )
    elif type == "contour":
        im = ax.contour(
            volume_data["N"] * 1e-3,
            volume_data["Z"] * 1e-3,
            volume_data.sel(E=slice_E, method="nearest").T,
            **kwargs,
        )
    elif type == "contourf":
        im = ax.contourf(
            volume_data["N"] * 1e-3,
            volume_data["Z"] * 1e-3,
            volume_data.sel(E=slice_E, method="nearest").T,
            **kwargs,
        )

    ax.plot(
        surface_data["N"] * 1e-3,
        surface_data["topography"].sel(E=slice_E, method="nearest") * 1e-3,
        "k",
        lw=1,
        zorder=20,
    )

    if labels:
        ax.set_xlabel("N [km]")
        ax.set_ylabel("Z [km]")

    if aspect is not None:
        ax.set_aspect(aspect)

    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.6)

    return ax


def plot_marginal_Z(
    ax,
    volume_data,
    surface_data,
    slice_E=None,
    slice_N=None,
    colorbar=False,
    labels=True,
    type="imshow",
    **kwargs,
):
    aspect = kwargs.pop("aspect", "auto")

    if type == "imshow":
        im = ax.imshow(
            volume_data.mean("Z").T,
            origin="lower",
            extent=[
                volume_data["E"].min() * 1e-3,
                volume_data["E"].max() * 1e-3,
                volume_data["N"].min() * 1e-3,
                volume_data["N"].max() * 1e-3,
            ],
            **kwargs,
        )
    elif type == "contour":
        im = ax.contour(
            volume_data["E"] * 1e-3,
            volume_data["N"] * 1e-3,
            volume_data.mean("Z").T,
            **kwargs,
        )
    elif type == "contourf":
        im = ax.contourf(
            volume_data["E"] * 1e-3,
            volume_data["N"] * 1e-3,
            volume_data.mean("Z").T,
            **kwargs,
        )

    if slice_N is not None:
        ax.axhline(slice_N * 1e-3, color="k", alpha=0.5, lw=1, zorder=10)

    if slice_E is not None:
        ax.axvline(slice_E * 1e-3, color="k", alpha=0.5, lw=1, zorder=10)

    if labels:
        ax.set_xlabel("E [km]")
        ax.set_ylabel("N [km]")

    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.6)

    if aspect is not None:
        ax.set_aspect(aspect)

    return ax


def plot_prior_model(
    PRIOR_DATA,
    SURFACE_DATA,
    VOLCANO_DATA,
    slice_E=0,
    slice_N=0,
):
    fig, ax_dict = plt.subplot_mosaic(
        [
            ["prior_marginal_Z", "prior_slice_E"],
            [
                "prior_marginal_Z",
                "prior_slice_N",
            ],
        ],
        figsize=(10, 4),
        empty_sentinel=None,
        dpi=120,
        gridspec_kw={"width_ratios": [4, 6], "height_ratios": [1, 1]},
    )

    plot_topography(ax_dict["prior_marginal_Z"], SURFACE_DATA["topography"])
    plot_marginal_Z(
        ax_dict["prior_marginal_Z"],
        PRIOR_DATA,
        SURFACE_DATA,
        slice_E=slice_E,
        slice_N=slice_N,
        cmap=blue_cmap,
        aspect=1,
    )
    ax_dict["prior_marginal_Z"].set_title("Z marginal")

    plot_slice_E(
        ax_dict["prior_slice_E"],
        PRIOR_DATA,
        SURFACE_DATA,
        slice_E,
        cmap=blue_cmap,
        aspect="auto",
    )
    # ax_dict['prior_slice_E'].set_title('E-W slice')

    ax_dict["prior_slice_E"].yaxis.set_label_position("right")
    ax_dict["prior_slice_E"].yaxis.set_tick_params(
        labelright=True, labelleft=False, right=True, left=True, which="both"
    )
    ax_dict["prior_slice_E"].xaxis.set_label_position("top")
    ax_dict["prior_slice_E"].xaxis.tick_top()
    ax_dict["prior_slice_E"].xaxis.set_tick_params(
        labeltop=True, labelbottom=False, top=True, bottom=True, which="both"
    )

    plot_slice_N(
        ax_dict["prior_slice_N"],
        PRIOR_DATA,
        SURFACE_DATA,
        slice_N,
        cmap=blue_cmap,
        aspect="auto",
    )
    # ax_dict['prior_slice_N'].set_title('N-S slice')
    ax_dict["prior_slice_N"].yaxis.tick_right()
    ax_dict["prior_slice_N"].yaxis.set_label_position("right")
    ax_dict["prior_slice_N"].yaxis.set_tick_params(
        labelright=True, labelleft=False, right=True, left=True, which="both"
    )

    ax_dict["prior_slice_N"].xaxis.tick_bottom()
    ax_dict["prior_slice_N"].xaxis.set_tick_params(
        labeltop=False, labelbottom=True, top=True, bottom=True, which="both"
    )

    return fig, ax_dict


def plot_posterior_model(
    design,
    posterior_data,
    surface_data,
    true_event,
    std=None,
    E_lim=None,
    N_lim=None,
    Z_lim=None,
    show=True,
):
    fig, ax_dict = plt.subplot_mosaic(
        [
            ["prior_marginal_Z", "prior_slice_E"],
            [
                "prior_marginal_Z",
                "prior_slice_N",
            ],
        ],
        figsize=(10, 4),
        empty_sentinel=None,
        dpi=120,
        gridspec_kw={"width_ratios": [4, 6], "height_ratios": [1, 1]},
    )

    plot_topography(ax_dict["prior_marginal_Z"], surface_data["topography"])
    plot_marginal_Z(
        ax_dict["prior_marginal_Z"],
        posterior_data,
        surface_data,
        true_event[0],
        true_event[1],
        cmap="Reds",
        type="contour",
        levels=5,
        zorder=11,
        aspect=1,
    )

    ax_dict["prior_marginal_Z"].set_title("Z marginal")

    # dummy for legend
    import matplotlib.path as mpath
    circle = mpath.Path.unit_circle()
    verts = np.copy(circle.vertices)
    verts[:, 0] *= 2
    ellipse_marker = mpath.Path(verts, circle.codes)

    
    ax_dict["prior_marginal_Z"].scatter(
        [],
        [],
        s=80,
        marker=ellipse_marker,
        linewidth=1.5,
        alpha=1,
        facecolors="none",
        edgecolors="darkred",
        label="posterior pdf",
    )

    if std is not None:
        circle = plt.Circle(
            (true_event[0] * 1e-3, true_event[1] * 1e-3),
            std * 1e-3,
            color="k",
            fill=False,
            linestyle="-",
            linewidth=1.5,
            zorder=12,
        )
        ax_dict["prior_marginal_Z"].add_artist(circle)

        # dummy for legend
        ax_dict["prior_marginal_Z"].scatter(
            [],
            [],
            s=50,
            marker="o",
            linewidth=1.5,
            alpha=1,
            facecolors="none",
            edgecolors="k",
            label="2x approx. standard deviation",
        )

    for sta_type, sta_data in design:
        ax_dict["prior_marginal_Z"].scatter(
            sta_data[0] * 1e-3,
            sta_data[1] * 1e-3,
            s=75 if "array" in sta_type else 100,
            marker="x" if "array" in sta_type else "^",
            c="k",
            linewidth=2 if "array" in sta_type else 0,
            alpha=1.0,
            label="array" if "array" in sta_type else "node",
        )

    if E_lim is not None:
        ax_dict["prior_marginal_Z"].set_xlim(E_lim)
    if N_lim is not None:
        ax_dict["prior_marginal_Z"].set_ylim(N_lim)

    ax_dict["prior_marginal_Z"].scatter(
        [], [], s=50, marker="*", c="black", linewidth=0, alpha=1.0, label="true event"
    )

    # remove duplicate labels
    handles, labels = ax_dict["prior_marginal_Z"].get_legend_handles_labels()
    by_label = dict(list(zip(labels, handles))[::-1])

    # plot in box below
    ax_dict["prior_marginal_Z"].legend(
        by_label.values(),
        by_label.keys(),
        facecolor="w",
        edgecolor="k",
        loc="upper center",
        bbox_to_anchor=(1.5, -0.15),
        ncol=5,
    )

    plot_slice_E(
        ax_dict["prior_slice_E"],
        posterior_data,
        surface_data,
        true_event[0],
        cmap="Reds",
        type="contour",
        levels=5,
        zorder=9,
        vmin=posterior_data.min() + 0.1 * (posterior_data.max() - posterior_data.min()),
        vmax=posterior_data.max(),
        aspect="auto",
    )

    # ax_dict['prior_slice_E'].set_title('E-W slice')
    ax_dict["prior_slice_E"].yaxis.set_label_position("right")
    ax_dict["prior_slice_E"].yaxis.set_tick_params(
        labelright=True, labelleft=False, right=True, left=True, which="both"
    )
    ax_dict["prior_slice_E"].xaxis.set_label_position("top")
    ax_dict["prior_slice_E"].xaxis.tick_top()
    ax_dict["prior_slice_E"].xaxis.set_tick_params(
        labeltop=True, labelbottom=False, top=True, bottom=True, which="both"
    )

    if std is not None:
        circle = plt.Circle(
            (true_event[1] * 1e-3, true_event[2] * 1e-3),
            std * 1e-3,
            color="black",
            fill=False,
            linewidth=1.5,
            zorder=10,
        )
        ax_dict["prior_slice_E"].add_artist(circle)

    ax_dict["prior_slice_E"].scatter(
        true_event[1] * 1e-3,
        true_event[2] * 1e-3,
        s=50,
        marker="*",
        c="black",
        linewidth=0,
        alpha=1.0,
    )
    if E_lim is not None:
        ax_dict["prior_slice_E"].set_xlim(E_lim)
    if Z_lim is not None:
        ax_dict["prior_slice_E"].set_ylim(Z_lim)

    plot_slice_N(
        ax_dict["prior_slice_N"],
        posterior_data,
        surface_data,
        true_event[1],
        cmap="Reds",
        type="contour",
        levels=5,
        zorder=9,
        vmin=posterior_data.min() + 0.1 * (posterior_data.max() - posterior_data.min()),
        vmax=posterior_data.max(),
        aspect="auto",
    )

    # ax_dict['prior_slice_N'].set_title('N-S slice')
    ax_dict["prior_slice_N"].yaxis.tick_right()
    ax_dict["prior_slice_N"].yaxis.set_label_position("right")
    ax_dict["prior_slice_N"].yaxis.set_tick_params(
        labelright=True, labelleft=False, right=True, left=True, which="both"
    )

    ax_dict["prior_slice_N"].xaxis.tick_bottom()
    ax_dict["prior_slice_N"].xaxis.set_tick_params(
        labeltop=False, labelbottom=True, top=True, bottom=True, which="both"
    )

    if std is not None:
        circle = plt.Circle(
            (true_event[0] * 1e-3, true_event[2] * 1e-3),
            std * 1e-3,
            color="black",
            fill=False,
            linewidth=1.5,
            zorder=10,
        )
        ax_dict["prior_slice_N"].add_artist(circle)

    ax_dict["prior_slice_N"].scatter(
        true_event[0] * 1e-3,
        true_event[2] * 1e-3,
        s=50,
        marker="*",
        c="black",
        linewidth=0,
        alpha=1.0,
    )
    if N_lim is not None:
        ax_dict["prior_slice_N"].set_xlim(N_lim)
    if Z_lim is not None:
        ax_dict["prior_slice_N"].set_ylim(Z_lim)

    if show:
        plt.show()

    return fig, ax_dict


def plot_design_space_dict(design_space_dict, SURFACE_DATA, VOLCANO_DATA, show=True):
    fig, ax_list = plt.subplots(
        1, len(design_space_dict.items()), figsize=(8, 4), sharex=True, sharey=True
    )
    if len(design_space_dict.items()) == 1:
        ax_list = [ax_list]

    for ax, (ds_name, ds) in zip(ax_list, design_space_dict.items()):
        plot_topography(ax, SURFACE_DATA["topography"])

        ax.imshow(
            ds.values.T,
            cmap=binary_cmap,
            extent=[
                SURFACE_DATA.E.min() * 1e-3 - SURFACE_DATA.E.mean() * 1e-3,
                SURFACE_DATA.E.max() * 1e-3 - SURFACE_DATA.E.mean() * 1e-3,
                SURFACE_DATA.N.min() * 1e-3 - SURFACE_DATA.N.mean() * 1e-3,
                SURFACE_DATA.N.max() * 1e-3 - SURFACE_DATA.N.mean() * 1e-3,
            ],
            origin="lower",
            vmin=0,
            vmax=1,
            alpha=0.7,
        )

        ax.set_title(f"{VOLCANO_DATA['Volcano Name']}: design space ({ds_name})")

    for i, ax in enumerate(ax_list):
        ax.set_xlabel("E [km]")

        if i == 0:
            ax.set_ylabel("N [km]")

            # a red and a white filled rectangle for not allowed and allowed areas
            handels = [
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor="white", alpha=0.7, edgecolor="k"
                ),
                plt.Rectangle((0, 0), 1, 1, facecolor="red", alpha=0.7, edgecolor="k"),
            ]
            labels = ["allowed", "not allowed"]
            ax.legend(handels, labels, loc="upper left", facecolor="w", edgecolor="k")

    if show:
        plt.show()

    return fig, ax_list


def plot_design(ax, design, **kwargs):
    unique_types = []

    if "color" not in kwargs:
        kwargs["color"] = "k"

    for sta_type, sta_data in design:
        ax.scatter(
            sta_data[0] * 1e-3,
            sta_data[1] * 1e-3,
            s=150 if "array" in sta_type else 200,
            marker="x" if "array" in sta_type else "^",
            linewidth=4 if "array" in sta_type else 0,
            **kwargs,
        )
        for data_type in sta_type:
            if data_type not in unique_types:
                unique_types.append(data_type)

    # if any of the types is not array, add a legend entry
    if ["array" == ut for ut in unique_types].count(True) != len(unique_types):
        ax.scatter(
            [],
            [],
            label="broadband",
            c="k",
            s=120,
            marker="^",
            linewidth=0,
        )
    if "array" in unique_types:
        ax.scatter(
            [],
            [],
            label="array",
            c="k",
            s=70,
            marker="x",
            linewidth=2,
        )
    # white background
    ax.legend(
        facecolor="w",
        edgecolor="k",
        loc="upper right",
    )

    return ax

def plot_design_statistics(ax, best_design, nmc_method, prior_information):
    eig = nmc_method(best_design)
    post_information = eig + prior_information

    k = 3
    std = np.exp((-post_information - k/2 * (1 + np.log(2*np.pi))) / (0.5 * k * 2))

    t = ax.text(
        x=1.1, y=0.95,
        s=rf'''Design Statistics:
        
    EIG:        {eig:.1f} nats
    Approx std: {std:.0f} m
    $_\mathrm{{std = standard \, deviation}}$ ''',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes,
        fontsize=8,
        fontfamily='monospace',
    )
    t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='black'))

    ax.set_xlabel("Easting [km]")
    ax.set_ylabel("Northing [km]")
    
    return eig, std


def interactive_design_plot(**kwargs):
    """
    Interactive plot of the design space and the design. The design can be interactively changed by dragging the stations around. The plot shows the topography, the design space and the design. The design is plotted as a scatter plot with the nodes as triangles and the arrays as crosses. The design can be changed by dragging the stations around. The plot also shows the expected information gain (EIG) and the standard deviation of the posterior distribution. The EIG is calculated using the given eig_criterion function. The standard deviation is calculated using the given prior information and the EIG.

    Parameters
    ----------
    original_design : list
        List of lists with the station types and the station data. The station type determines which data types the station records. The station data should be a list with the easting, northing and elevation of the station.
    volcano_data : dict
        Dictionary with the volcano data. The dictionary should have the keys 'lat' and 'lon' with the latitude and longitude of the volcano.
    surface_data : xarray.DataArray
        DataArray with the topography data. The DataArray should have dimensions 'E' and 'N' for the easting and northing coordinates and the values should be the elevation in meters.
    design_space_dict : dict
        Dictionary with the design space data. The dictionary should have the keys 'node' and 'array' with the design space data for the nodes and arrays. The design space data should be a xarray.DataArray with dimensions 'E' and 'N' for the easting and northing coordinates and the values should be 1 for allowed and 0 for not allowed.
    eig_criterion : function
        Function that calculates the expected information gain (EIG) for a given design. The function should take the design as input and return the EIG in nats.
    prior_information : float
        The prior information in nats.
    posterior : bool, optional
        If True, the plot shows the posterior distribution. The default is False.

    Returns
    -------
    bqplot.Figure
        The interactive plot. Use display() to show the plot.
    list of lists
        The changed design.
    np.array, optional
        The true event location used for the posterior plot. Only returned if posterior=True.
    """

    posterior = kwargs.get("posterior", False)

    if posterior:
        return _interactive_design_plot_posterior(**kwargs)
    else:
        return _interactive_design_plot_plain(**kwargs)


def _interactive_design_plot_plain(
    original_design,
    volcano_data,
    surface_data,
    design_space_dict,
    eig_criterion,
    prior_information,
    **kwargs,
    ):
    
    import ipywidgets as widgets

    changing_design = list(original_design.copy())
    changing_design = [
        [sta_type, np.array(sta_data)] for sta_type, sta_data in changing_design
    ]

    fig = bqp_plt.figure(
        figsize=(12, 12),
        title=f'{volcano_data["Volcano Name"]}: optimal design',
        min_aspect_ratio=1,
        max_aspect_ratio=1,
        #  legend_location="top-right", legend_style={"fill": "white", "stroke": "black"}
    )

    fx = interp1d(
        np.arange(0, surface_data["E"].values.shape[0]), surface_data["E"].values * 1e-3
    )
    fy = interp1d(
        np.arange(0, surface_data["N"].values.shape[0]), surface_data["N"].values * 1e-3
    )

    bqp_plt.heatmap(
        surface_data["topography"].T,
        x=surface_data["E"] * 1e-3,
        y=surface_data["N"] * 1e-3,
        scales={
            "x": bqp_plt.LinearScale(
                min=float(surface_data["E"].values.min()) * 1e-3,
                max=float(surface_data["E"].values.max()) * 1e-3,
            ),
            "y": bqp_plt.LinearScale(
                min=float(surface_data["N"].values.min()) * 1e-3,
                max=float(surface_data["N"].values.max()) * 1e-3,
            ),
            "color": ColorScale(scheme="Greys"),
        },
        axes_options={
            "color": dict(
                label="z [m]", orientation="horizontal", side="bottom", visible=False
            ),
            "x": dict(label="E [km]", orientation="horizontal", side="bottom"),
            "y": dict(label="N [km]", orientation="vertical", side="left"),
        },
        display_legend=False,
    )

    scat_list = []
    unique_types = []
    for i, (sta_type, sta_data) in enumerate(changing_design):
        scat = bqp_plt.scatter(
            [sta_data[0] * 1e-3],   
            [sta_data[1] * 1e-3],
            default_size=300 if "array" in sta_type else 150,
            marker="crosshair" if "array" in sta_type else "triangle-up",
            colors=["black"],
            stroke_width=5 if "array" in sta_type else 0,
            enable_move=True,
            display_legend=False,
            names=[i],
            display_names=False,
        )
        for data_type in sta_type:
            if data_type not in unique_types:
                unique_types.append(data_type)
        scat_list.append(scat)

    # if any of the types is not array, add a legend entry
    if ["array" == ut for ut in unique_types].count(True) != len(unique_types):
        bqp_plt.plot(
            [sta_data[0] * 1e-3, sta_data[0] * 1e-3],
            [sta_data[1] * 1e-3, sta_data[1] * 1e-3],
            colors=["darkred"],
            stroke_width=5,
        )
        ds_nodes_cs = measure.find_contours(design_space_dict["node"].values, 0.5)

        for contour in ds_nodes_cs:
            contour[:, 0] = fx(contour[:, 0])
            contour[:, 1] = fy(contour[:, 1])

        for c in ds_nodes_cs:
            bqp_plt.plot(
                c[:, 0], c[:, 1], colors=["#FFC300"],
                stroke_width=1, opacity=0.8, line_style="solid"
            )

    if "array" in unique_types:
        bqp_plt.scatter(
            [],
            [],
            labels="array",
            colors=["black"],
            s=70,
            marker="cross",
            linewidth=2,
        )
        ds_array_cs = measure.find_contours(design_space_dict["array"].values, 0.5)

        for contour in ds_array_cs:
            contour[:, 0] = fx(contour[:, 0])
            contour[:, 1] = fy(contour[:, 1])

        for c in ds_array_cs:
            bqp_plt.plot(
                c[:, 0], c[:, 1], colors=["#009933"],
                stroke_width=1, opacity=0.8, line_style="solid")

    eig = eig_criterion(changing_design)
    post_information = eig + prior_information

    k = 3
    std = np.exp(
        (post_information + k / 2 + k / 2 * np.log(2 * np.pi)) / (-0.5 * k * 2)
    )

    label = bqp_plt.label(
        text=[
            "Design Statistics:",
            f"EIG: {eig:.3f} nats",
            f"Approx std: {std:.0f} m",
            "",
        ],
        x=[
            surface_data["E"].max() * 1e-3
            + 0.1 * (surface_data["E"].max() * 1e-3 - surface_data["E"].min() * 1e-3),
            surface_data["E"].max() * 1e-3
            + 0.1 * (surface_data["E"].max() * 1e-3 - surface_data["E"].min() * 1e-3),
            surface_data["E"].max() * 1e-3
            + 0.1 * (surface_data["E"].max() * 1e-3 - surface_data["E"].min() * 1e-3),
            surface_data["E"].max() * 1e-3
            + 0.1 * (surface_data["E"].max() * 1e-3 - surface_data["E"].min() * 1e-3),
        ],
        y=[
            surface_data["N"].max() * 1e-3
            - 0.1 * (surface_data["N"].max() * 1e-3 - surface_data["N"].min() * 1e-3),
            surface_data["N"].max() * 1e-3
            - 0.2 * (surface_data["N"].max() * 1e-3 - surface_data["N"].min() * 1e-3),
            surface_data["N"].max() * 1e-3
            - 0.3 * (surface_data["N"].max() * 1e-3 - surface_data["N"].min() * 1e-3),
            surface_data["N"].max() * 1e-3
            - 0.4 * (surface_data["N"].max() * 1e-3 - surface_data["N"].min() * 1e-3),
        ],
        colors=["black", "black", "black", "red"],
        apply_clip=False,
    )

    def update_line(_scatter, event):
        # with scat.hold_sync():
        label.text = [
            "Design Statistics:",
            "",
            "",
            "running...",
        ]

        index = _scatter.names.item()
        sta_typ = "array" if "array" in changing_design[index][0] else "node"

        new_location = np.array(
            [event["point"]["x"] * 1e3, event["point"]["y"] * 1e3]
        )
        new_elevation = get_elevation(new_location * 1e-3, surface_data)
        new_location = np.append(new_location, new_elevation)

        changing_design[index][1] = new_location

        for sta_type, sta_data in changing_design:
            if "array" in sta_type:
                in_ds = (
                    design_space_dict["array"]
                    .sel(E=sta_data[0], N=sta_data[1], method="nearest")
                    .values
                )
                if not in_ds:
                    break
            # if its not an array, it must be a node
            else:
                in_ds = (
                    design_space_dict["node"]
                    .sel(E=sta_data[0], N=sta_data[1], method="nearest")
                    .values
                )

                if not in_ds:
                    break

        eig = eig_criterion(changing_design)

        post_information = eig + prior_information
        k = 3
        std = np.exp(
            (post_information + k / 2 + k / 2 * np.log(2 * np.pi)) / (-0.5 * k * 2)
        )

        label.text = [
            "Design Statistics:",
            f"EIG: {eig:.3f} nats",
            f"Approx std: {std:.0f} m",
            "" if in_ds else "Current location is not in design space",
        ]

    for i, scat in enumerate(scat_list):
        scat.on_drag_end(update_line)



    return widgets.VBox([fig]), changing_design, None


def _interactive_design_plot_posterior(
    original_design,
    volcano_data,
    surface_data,
    prior_data,
    design_space_dict,
    eig_criterion,
    prior_information,
    forward_function,
    **kwargs,
):
    from helpers.posterior_helpers import calculate_posterior
    import ipywidgets as widgets

    z_initial = np.average(
        prior_data.Z.values, weights=prior_data.mean(["E", "N"]).values
    )

    # slider for z position of source
    z_slider = widgets.FloatSlider(
        value=z_initial,
        min=prior_data.Z.values.min(),
        max=prior_data.Z.values.max(),
        step=(prior_data.Z.values.max() - prior_data.Z.min()) / prior_data.Z.values.shape[0],
        description="Z [m]",
        continuous_update=False,
    )

    initial_model = np.array(
        [
            0.0,
            0.0,
            np.average(prior_data.Z.values, weights=prior_data.mean(["E", "N"]).values),
        ]
    )

    _, initial_posterior = calculate_posterior(
        original_design, initial_model, prior_data, forward_function
    )

    fx_post = interp1d(
        np.arange(0, initial_posterior["E"].values.shape[0]),
        initial_posterior["E"].values * 1e-3,
    )
    fx_post = interp1d(
        np.arange(0, initial_posterior["N"].values.shape[0]),
        initial_posterior["N"].values * 1e-3,
    )

    fx = interp1d(
        np.arange(0, surface_data["E"].values.shape[0]), surface_data["E"].values * 1e-3
    )
    fy = interp1d(
        np.arange(0, surface_data["N"].values.shape[0]), surface_data["N"].values * 1e-3
    )

    changing_design = list(original_design.copy())
    changing_design = [
        [sta_type, np.array(sta_data)] for sta_type, sta_data in changing_design
    ]

    fig = bqp_plt.figure(
        figsize=(12, 12),
        title=f'{volcano_data["Volcano Name"]}: optimal design',
        min_aspect_ratio=1,
        max_aspect_ratio=1,
        #  legend_location="top-right", legend_style={"fill": "white", "stroke": "black"}
    )

    bqp_plt.heatmap(
        surface_data["topography"].T,
        x=surface_data["E"] * 1e-3,
        y=surface_data["N"] * 1e-3,
        scales={
            "x": bqp_plt.LinearScale(
                min=float(surface_data["E"].values.min()) * 1e-3,
                max=float(surface_data["E"].values.max()) * 1e-3,
            ),
            "y": bqp_plt.LinearScale(
                min=float(surface_data["N"].values.min()) * 1e-3,
                max=float(surface_data["N"].values.max()) * 1e-3,
            ),
            "color": ColorScale(scheme="Greys"),
        },
        axes_options={
            "color": dict(
                label="z [m]", orientation="vertical", side="right", visible=True
            ),
            "x": dict(label="E [km]", orientation="horizontal", side="bottom"),
            "y": dict(label="N [km]", orientation="vertical", side="left"),
        },
        display_legend=False,
    )

    posterior_plots = []

    # post_slice = initial_posterior.mean('Z')
    post_slice_initial = initial_posterior.sel(Z=initial_model[2], method="nearest")

    post_diff = post_slice_initial.max().values - post_slice_initial.min().values
    levels = [0.1, 0.5, 0.9]

    for i, level in enumerate(
        [
            post_slice_initial.min().values + levels[i] * post_diff
            for i in range(len(levels))
        ]
    ):
        post_contour = measure.find_contours(post_slice_initial.values, level)

        for contour in post_contour:
            contour[:, 0] = fx_post(contour[:, 0])
            contour[:, 1] = fx_post(contour[:, 1])

        for j, c in enumerate(post_contour):
            posterior_plots.append(
                bqp_plt.plot(
                    c[:, 0],
                    c[:, 1],
                    colors=["red"],
                    stroke_width=2,
                    opacity=[0.9, 0.5, 0.1][i],
                )
            )

    scat_list = []
    unique_types = []
    for i, (sta_type, sta_data) in enumerate(changing_design):
        scat = bqp_plt.scatter(
            [sta_data[0] * 1e-3],
            [sta_data[1] * 1e-3],
            default_size=300 if "array" in sta_type else 150,
            marker="crosshair" if "array" in sta_type else "triangle-up",
            colors=["black"],
            stroke_width=5 if "array" in sta_type else 0,
            enable_move=True,
            display_legend=False,
            names=[i],
            display_names=False,
        )
        for data_type in sta_type:
            if data_type not in unique_types:
                unique_types.append(data_type)
        scat_list.append(scat)

    src_scat = bqp_plt.scatter(
        [initial_model[0] * 1e-3],
        [initial_model[1] * 1e-3],
        default_size=50,
        marker="diamond",
        colors=["red"],
        stroke_width=0,
        enable_move=True,
        display_legend=False,
        names="src",
        display_names=False,
    )
    scat_list.append(src_scat)

    # if any of the types is not array, add a legend entry
    if ["array" == ut for ut in unique_types].count(True) != len(unique_types):
        bqp_plt.plot(
            [sta_data[0] * 1e-3, sta_data[0] * 1e-3],
            [sta_data[1] * 1e-3, sta_data[1] * 1e-3],
            colors=["darkred"],
            stroke_width=5,
        )
        ds_nodes_cs = measure.find_contours(design_space_dict["node"].values, 0.5)

        for contour in ds_nodes_cs:
            contour[:, 0] = fx(contour[:, 0])
            contour[:, 1] = fy(contour[:, 1])

        for c in ds_nodes_cs:
            bqp_plt.plot(c[:, 0], c[:, 1], colors=["#FFC300"],
                         stroke_width=1, opacity=0.8, line_style="solid")

    if "array" in unique_types:
        bqp_plt.scatter(
            [],
            [],
            labels="array",
            colors=["black"],
            s=70,
            marker="cross",
            linewidth=2,
        )
        ds_array_cs = measure.find_contours(design_space_dict["array"].values, 0.5)

        for contour in ds_array_cs:
            contour[:, 0] = fx(contour[:, 0])
            contour[:, 1] = fy(contour[:, 1])

        for c in ds_array_cs:
            bqp_plt.plot(
                c[:, 0], c[:, 1], colors=["#009933"],
                stroke_width=1, opacity=0.8, line_style="solid")

    eig = eig_criterion(changing_design)
    post_information = eig + prior_information

    k = 3
    std = np.exp(
        (post_information + k / 2 + k / 2 * np.log(2 * np.pi)) / (-0.5 * k * 2)
    )

    label = bqp_plt.label(
        text=[
            "Design Statistics:",
            f"EIG: {eig:.3f} nats",
            f"Approx std: {std:.0f} m",
            "",
        ],
        x=[
            surface_data["E"].max() * 1e-3
            + 0.4 * (surface_data["E"].max() * 1e-3 - surface_data["E"].min() * 1e-3),
            surface_data["E"].max() * 1e-3
            + 0.4 * (surface_data["E"].max() * 1e-3 - surface_data["E"].min() * 1e-3),
            surface_data["E"].max() * 1e-3
            + 0.4 * (surface_data["E"].max() * 1e-3 - surface_data["E"].min() * 1e-3),
            surface_data["E"].max() * 1e-3
            + 0.4 * (surface_data["E"].max() * 1e-3 - surface_data["E"].min() * 1e-3),
        ],
        y=[
            surface_data["N"].max() * 1e-3
            - 0.1 * (surface_data["N"].max() * 1e-3 - surface_data["N"].min() * 1e-3),
            surface_data["N"].max() * 1e-3
            - 0.2 * (surface_data["N"].max() * 1e-3 - surface_data["N"].min() * 1e-3),
            surface_data["N"].max() * 1e-3
            - 0.3 * (surface_data["N"].max() * 1e-3 - surface_data["N"].min() * 1e-3),
            surface_data["N"].max() * 1e-3
            - 0.4 * (surface_data["N"].max() * 1e-3 - surface_data["N"].min() * 1e-3),
        ],
        colors=["black", "black", "black", "red"],
        apply_clip=False,
    )

    def update_line(_scatter, event):
        # with scat.hold_sync():

        # label.text[-1] = 'running...'
        label.text = [
            "Design Statistics:",
            "",
            "",
            "running...",
        ]

        index = _scatter.names.item()

        if index == "src":
            initial_model[0] = event["point"]["x"] * 1e3
            initial_model[1] = event["point"]["y"] * 1e3
        else:
            new_location = np.array(
                [event["point"]["x"] * 1e3, event["point"]["y"] * 1e3]
            )
            new_elevation = get_elevation(new_location * 1e-3, surface_data)
            new_location = np.append(new_location, new_elevation)

            changing_design[index][1] = new_location

        for sta_type, sta_data in changing_design:
            if "array" in sta_type:
                in_ds = (
                    design_space_dict["array"]
                    .sel(E=sta_data[0], N=sta_data[1], method="nearest")
                    .values
                )
                if not in_ds:
                    break
            # if its not an array, it must be a node
            else:
                in_ds = (
                    design_space_dict["node"]
                    .sel(E=sta_data[0], N=sta_data[1], method="nearest")
                    .values
                )

                if not in_ds:
                    break

        eig = eig_criterion(changing_design)

        post_information = eig + prior_information
        k = 3
        std = np.exp(
            (post_information + k / 2 + k / 2 * np.log(2 * np.pi)) / (-0.5 * k * 2)
        )

        _, new_posterior = calculate_posterior(
            changing_design, initial_model, prior_data, forward_function
        )

        # post_slice = new_posterior.mean('Z')
        post_slice = new_posterior.sel(Z=initial_model[2], method="nearest")

        post_diff = post_slice.max().values - post_slice.min().values

        for i, level in enumerate(
            [
                post_slice.min().values + levels[i] * post_diff
                for i in range(len(levels))
            ]
        ):
            post_contour = measure.find_contours(post_slice.values, level)

            for contour in post_contour:
                contour[:, 0] = fx_post(contour[:, 0])
                contour[:, 1] = fx_post(contour[:, 1])

            for j, c in enumerate(post_contour):
                # posterior_plots[i*j].x = c[:,0]
                # posterior_plots[i*j].y = c[:,1]
                try:
                    posterior_plots[j * len(levels) + i].x = c[:, 0]
                    posterior_plots[j * len(levels) + i].y = c[:, 1]
                except IndexError:
                    pass

        label.text = [
            "Design Statistics:",
            f"EIG: {eig:.3f} nats",
            f"Approx std: {std:.0f} m",
            "" if in_ds else "Current location is not in design space",
        ]

    def slider_update(change):
        initial_model[2] = z_slider.value

        label.text = [
            "Design Statistics:",
            "",
            "",
            "running...",
        ]

        for sta_type, sta_data in changing_design:
            if "array" in sta_type:
                in_ds = (
                    design_space_dict["array"]
                    .sel(E=sta_data[0], N=sta_data[1], method="nearest")
                    .values
                )
                if not in_ds:
                    break
            # if its not an array, it must be a node
            else:
                in_ds = (
                    design_space_dict["node"]
                    .sel(E=sta_data[0], N=sta_data[1], method="nearest")
                    .values
                )

                if not in_ds:
                    break

        eig = eig_criterion(changing_design)

        post_information = eig + prior_information
        k = 3
        std = np.exp(
            (post_information + k / 2 + k / 2 * np.log(2 * np.pi)) / (-0.5 * k * 2)
        )

        _, new_posterior = calculate_posterior(
            changing_design, initial_model, prior_data, forward_function
        )

        post_slice = new_posterior.sel(Z=initial_model[2], method="nearest")

        post_diff = post_slice.max().values - post_slice.min().values

        for i, level in enumerate(
            [
                post_slice.min().values + levels[i] * post_diff
                for i in range(len(levels))
            ]
        ):
            post_contour = measure.find_contours(post_slice.values, level)

            for contour in post_contour:
                contour[:, 0] = fx_post(contour[:, 0])
                contour[:, 1] = fx_post(contour[:, 1])

            for j, c in enumerate(post_contour):
                try:
                    posterior_plots[j * len(levels) + i].x = c[:, 0]
                    posterior_plots[j * len(levels) + i].y = c[:, 1]
                except IndexError:
                    pass

        label.text = [
            "Design Statistics:",
            f"EIG: {eig:.3f} nats",
            f"Approx std: {std:.0f} m",
            "" if in_ds else "Current location is not in design space",
        ]

    for i, scat in enumerate(scat_list):
        scat.on_drag_end(update_line)

    z_slider.observe(slider_update, names="value")

    return widgets.VBox([fig, z_slider]), changing_design, initial_model
