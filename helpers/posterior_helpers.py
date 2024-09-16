import numpy as np
import xarray as xr


def calculate_posterior(
    design, true_event, prior_data, forward_function, downsample_size=128
):
    tt_obs, _ = forward_function(design, true_event[None])

    if downsample_size is not None:
        if not downsample_size == prior_data.E.size:
            prior_data = prior_data.interp(
                E=np.linspace(prior_data.E.min(), prior_data.E.max(), downsample_size),
                N=np.linspace(prior_data.N.min(), prior_data.N.max(), downsample_size),
                Z=np.linspace(prior_data.Z.min(), prior_data.Z.max(), downsample_size),
                method="linear",
            )

    source_locations = np.meshgrid(
        prior_data.E.values, prior_data.N.values, prior_data.Z.values, indexing="ij"
    )
    src_mask = prior_data.values > 0
    src_loc_grid = np.stack(source_locations, axis=-1).reshape(-1, 3)[src_mask.ravel()]

    tt, cov = forward_function(design, src_loc_grid)
    log_data_likelihood = np.nan * np.ones(prior_data.values.shape)
    log_data_likelihood[src_mask] = -0.5 * np.sum(
        np.log(2 * np.pi * cov) + (tt - tt_obs) ** 2 / cov, axis=-1
    )

    log_prior = np.nan * np.ones(prior_data.values.shape)
    log_prior[src_mask] = np.log(prior_data.values[src_mask])

    log_evidence = np.nanmean(log_data_likelihood + log_prior)
    log_posterior = log_data_likelihood + log_prior - log_evidence
    log_posterior = np.nan_to_num(log_posterior, nan=-np.inf)

    return xr.DataArray(
        np.exp(log_prior),
        coords=dict(
            E=prior_data.E,
            N=prior_data.N,
            Z=prior_data.Z,
        ),
        dims=["E", "N", "Z"],
        name="log_prior",
    ), xr.DataArray(
        np.exp(log_posterior),
        coords=dict(
            E=prior_data.E,
            N=prior_data.N,
            Z=prior_data.Z,
        ),
        dims=["E", "N", "Z"],
        name="log_posterior",
    )

def get_posterior_statistics(posterior_data):
    posterior_mode = np.unravel_index(
        np.argmax(posterior_data.values), posterior_data.shape
    )
    posterior_mode = np.array(
        [
            posterior_data["E"].values[posterior_mode[0]],
            posterior_data["N"].values[posterior_mode[1]],
            posterior_data["Z"].values[posterior_mode[2]],
        ]
    )

    E, N, Z = np.meshgrid(
        posterior_data["E"].values,
        posterior_data["N"].values,
        posterior_data["Z"].values,
        indexing="ij",
    )

    posterior_mean = np.array(
        [
            (E * posterior_data.values).sum() / posterior_data.values.sum(),
            (N * posterior_data.values).sum() / posterior_data.values.sum(),
            (Z * posterior_data.values).sum() / posterior_data.values.sum(),
        ]
    )

    posterior_std = np.sqrt(
        np.array(
            [
                ((E - posterior_mean[0]) ** 2 * posterior_data.values).sum()
                / posterior_data.values.sum(),
                ((N - posterior_mean[1]) ** 2 * posterior_data.values).sum()
                / posterior_data.values.sum(),
                ((Z - posterior_mean[2]) ** 2 * posterior_data.values).sum()
                / posterior_data.values.sum(),
            ]
        )
    )

    return posterior_mode, posterior_mean, posterior_std

def display_posterior_statistics(posterior_mode, posterior_mean, posterior_std):
    return '''
Posterior Mode:          Posterior Mean:          Posterior Std:
    E: {0[0]:10.2f} m \t\t E: {1[0]:10.2f} m \t\t E: {2[0]:10.2f} m 
    N: {0[1]:10.2f} m \t\t N: {1[1]:10.2f} m \t\t N: {2[1]:10.2f} m
    Z: {0[2]:10.2f} m \t\t Z: {1[2]:10.2f} m \t\t Z: {2[2]:10.2f} m
    '''.format(posterior_mode, posterior_mean, posterior_std)

