import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import math
from sklearn.utils.extmath import fast_logdet
from scipy.special import logsumexp


def get_prior_samples(prior_data, n_samples):
    """
    Helper function that generates model samples from a prior distribution described by a 3D grid of probabilities.

    Parameters
    ----------
    prior_data : xr.Dataset
        The prior data that describes the prior distribution of the source locations. The data must have E, N, and Z coordinates.
    n_samples : int
        The number of samples to generate.
    """

    dE = prior_data["E"].values[1] - prior_data["E"].values[0]
    dN = prior_data["N"].values[1] - prior_data["N"].values[0]
    dZ = prior_data["Z"].values[1] - prior_data["Z"].values[0]

    source_location_mask = prior_data.values > 0.0
    source_locations = np.meshgrid(
        prior_data["E"].values,
        prior_data["N"].values,
        prior_data["Z"].values,
        indexing="ij",
    )
    source_locations = np.stack(source_locations, axis=-1)[source_location_mask]

    np.random.seed(0)
    model_samples = source_locations[
        np.random.choice(
            len(source_locations),
            n_samples,
            p=prior_data.values[source_location_mask].flatten(),
        )
    ]

    # add uniform noise to vary source locations within their grid cell
    model_samples += np.random.uniform(-0.5, 0.5, model_samples.shape) * np.array(
        [dE, dN, dZ]
    )

    return model_samples


class NMC_method:
    def __init__(self, forward_function, prior_data, n_model_samples=1000):
        """
        Method to calculate the NMC criterion for a given design. During the initialisation, prior model samples are drawn from the prior distribution described by a 3D grid of probabilities.

        Parameters
        ----------
        forward_function : callable
            The forward function that takes a design and a set of model samples and returns the travel times and their covariances.
        prior_data : xr.Dataset
            The prior data that describes the prior distribution of the source locations. The data must have E, N, and Z coordinates.
        n_model_samples : int
            The number of model samples to draw from the prior distribution and use in the NMC criterion calculation.
        """
        self.forward_function = forward_function

        self.model_samples = self._construct_model_samples(prior_data, n_model_samples)

    def __call__(self, design):
        """
        Method to calculate the NMC criterion for a given design.

        Parameters
        ----------
        design : iterable of iterables
            A tuple/list that contains the station type and the station coordinates for each station in the design.

        Returns
        -------
        float
            The NMC criterion value for the given design.

        """

        np.random.seed(0)
        tt, cov = self.forward_function(design, self.model_samples)

        N, M = tt.shape[0], tt.shape[0]

        tt_noise = tt + np.random.normal(0, 1, tt.shape) * np.sqrt(cov)
        likelihood_term = -0.5 * np.sum(
            np.log(2 * np.pi * cov) + (tt_noise - tt) ** 2 / cov, axis=-1
        )

        log_scale = np.log(np.sqrt(cov.reshape(N, 1, -1)))
        evidence_term = logsumexp(
            np.sum(
                -((tt_noise.reshape(N, 1, -1) - tt.reshape(1, N, -1)) ** 2)
                / (2 * cov.reshape(N, 1, -1))
                - log_scale
                - np.log(np.sqrt(2 * np.pi)),
                axis=-1,
            ),
            axis=-1,
        ) - np.log(M)

        eig = (likelihood_term - evidence_term).sum(0) / N

        return eig.item()

    @staticmethod
    def _construct_model_samples(prior_data, n_model_samples):
        return get_prior_samples(prior_data, n_model_samples)


class DN_method:
    def __init__(self, forward_function, prior_data, n_model_samples=1000):
        """
        Method to calculate the DN criterion for a given design. During the initialisation, prior model samples are drawn from the prior distribution described by a 3D grid of probabilities.

        Parameters
        ----------
        forward_function : callable
            The forward function that takes a design and a set of model samples and returns the travel times and their covariances.
        prior_data : xr.Dataset
            The prior data that describes the prior distribution of the source locations. The data must have E, N, and Z coordinates.
        n_model_samples : int
            The number of model samples to draw from the prior distribution and use in the DN criterion calculation.
        """

        self.forward_function = forward_function

        self.model_samples = self._construct_model_samples(prior_data, n_model_samples)

    def __call__(self, design):
        """
        Method to calculate the DN criterion for a given design.

        Parameters
        ----------
        design : iterable of iterables
            A tuple/list that contains the station type and the station coordinates for each station in the design.

        Returns
        -------
        float
            The DN criterion value for the given design.
        """
        np.random.seed(0)
        tt, cov = self.forward_function(design, self.model_samples)

        tt_noise = tt + np.random.normal(0, 1, tt.shape) * np.sqrt(cov)
        likelihood_term = -0.5 * np.sum(
            np.log(2 * np.pi * cov) + (tt_noise - tt) ** 2 / cov, axis=-1
        )
        likelihood_term = likelihood_term.sum(0) / tt.shape[0]

        if tt_noise.shape[-2] >= 2:
            evidence_covariance = np.cov(tt_noise.T)
        else:
            evidence_covariance = np.cov(tt_noise.T)[None, None]

        evidence_cov = fast_logdet(evidence_covariance)

        k = evidence_covariance.shape[-1]
        evidence_term = -(k / 2 + k / 2 * np.log(2 * np.pi) + 0.5 * evidence_cov)

        eig = likelihood_term - evidence_term

        return eig.item()

    @staticmethod
    def _construct_model_samples(prior_data, n_model_samples):
        return get_prior_samples(prior_data, n_model_samples)


class Design_Optimisation:
    def __init__(
        self, design_criterion, surface_data, design_space_dict, preexisting_design=None
    ):
        """
        Class to perform design optimisation for a given design criterion.

        Parameters
        ----------
        design_criterion : callable
            The design criterion function that takes a tuple/list of tuples/lists, where each tuple contains the station type and the station coordinates, and returns the value of the design criterion.
        surface_data : xr.Dataset
            The surface data that describes the topography of the surface. The data must have E, N, and Z coordinates.
        design_space_dict : xr.Dataset
            A dictionary that contains the design space for node and array stations. The design space values is an xarray DataArray with E, N, and Z coordinates and a boolean mask that describes the design space.
        preexisting_design : iterable of iterables, optional
            A tuple/list of tuples/lists that contain the station type and the station coordinates for preexisting stations. Default is None.

        Examples
        --------
        >>> design_criterion = NMC_method(forward_function, prior_data, n_model_samples=1000)
        >>> design_space_dict = {
        ...     'node': design_space_node,
        ...     'array': design_space_array,
        ... }
        >>> preexisting_design = [
        ...     ('arrival_p', np.array([0.0, 0.0, 0.0])),
        ...     ('array', np.array([0.0, 0.0, 0.0])),
        ... ]
        >>> design_optimisation = Design_Optimisation(
        ...     design_criterion, surface_data, design_space_dict, preexisting_design)
        """

        self.design_criterion = design_criterion
        self.surface_data = surface_data

        # convert all tuples to lists
        if preexisting_design is not None:
            self.preexisting_design = [
                [d[0], np.array(d[1])] for d in preexisting_design
            ]
        else:
            self.preexisting_design = None

        design_points_dict = {}
        design_points_mask_dict = {}

        for receiver_type in ["node", "array"]:
            design_points, design_points_mask = self._construct_design_points(
                design_space_dict[receiver_type], self.surface_data, receiver_type
            )
            design_points_dict[receiver_type] = design_points
            design_points_mask_dict[receiver_type] = design_points_mask

        self.design_points_dict = design_points_dict
        self.design_points_mask_dict = design_points_mask_dict

    def _construct_design_points(
        self, design_space, surface_data, receiver_type="node"
    ):
        """
        Helper function to construct the design points for a given receiver type.
        """

        design_points_hor = np.stack(
            np.meshgrid(design_space.E, design_space.N, indexing="ij"), axis=-1
        ).reshape(-1, 2)

        design_points_vert = surface_data["topography"].values.flatten()[..., None]
        design_points = np.hstack(
            [design_points_hor, design_points_vert],
        )

        design_points_mask = design_space.values.flatten()
        design_points = design_points[design_points_mask]

        if receiver_type == "array":
            dE = surface_data.E.values[1] - surface_data.E.values[0]
            dN = surface_data.N.values[1] - surface_data.N.values[0]

            ds_array_grad = np.gradient(surface_data["topography"], dE, dN)
            ds_array_normal = np.stack(
                [
                    -ds_array_grad[0][
                        design_points_mask.reshape(*surface_data["topography"].shape)
                    ],
                    -ds_array_grad[1][
                        design_points_mask.reshape(*surface_data["topography"].shape)
                    ],
                    np.ones_like(
                        ds_array_grad[0][
                            design_points_mask.reshape(
                                *surface_data["topography"].shape
                            )
                        ]
                    ),
                ],
                axis=-1,
            )

            ds_array_normal /= np.linalg.norm(ds_array_normal, axis=-1, keepdims=True)

            # add normal vector of the surface to the design space
            design_points = np.hstack([design_points, ds_array_normal])

        return design_points, design_points_mask

    def get_optimal_design(
        self,
        available_stations,
        optimisation_algorithm="genetic",
        optimisation_kwargs={},
    ):
        """
        Method to perform design optimisation for a given design criterion.

        Parameters
        ----------
        available_stations : tuple or dict
            A tuple or a dictionary that contains the available station types for the design optimisation. If a tuple is provided, each element of the tuple must be a tuple that contains the station types. If a dictionary is provided, the dictionary must have the keys 'n_node' and 'n_array' that describe the number of node and array stations, respectively.
        optimisation_algorithm : str, optional
            The optimisation algorithm to use. Default is 'genetic'.
        optimisation_kwargs : dict, optional
            A dictionary that contains the keyword arguments for the optimisation algorithm. Default is {}. For a list of available keyword arguments, see the documentation of the optimisation algorithm.

        Returns
        -------
        list
            A list that contains the optimal design. Each element of the list is a tuple that contains the station type and the station coordinates.
        dict or class
            A dictionary or a class that contains additional information about the optimisation process.

        Examples
        --------
        >>> available_stations = [('arrival_p', 'amplitude_s'), ('arrival_p', 'amplitude_s', 'array')]
        >>> optimisation_algorithm = 'genetic'
        >>> optimisation_kwargs = {
        ...     'num_generations': 100,
        ...     'num_parents_mating': 4,
        ...     'plot_fitness': True,
        ...     'random_seed': 0,
        ... }
        """

        optimisation_kwargs = dict(optimisation_kwargs)

        if isinstance(available_stations, dict):
            N_node = available_stations.get("n_node", 0)
            N_array = available_stations.get("n_array", 0)
            available_stations = [
                ("arrival_p", "amplitude_s"),
            ] * N_node
            available_stations += [
                ("arrival_p", "amplitude_s", "array"),
            ] * N_array
        elif isinstance(available_stations, tuple) or isinstance(available_stations, list):
            for station in available_stations:
                assert isinstance(station, tuple) or isinstance(station, list)
                for dtype in station:
                    assert (
                        dtype[:7] == "arrival"
                        or dtype[:9] == "amplitude"
                        or dtype[:5] == "array"
                    )
        else:
            raise ValueError("available_stations must be either a tuple/list or a dict")

        if optimisation_algorithm == "differential_evolution":
            best_design, info = self._differential_evolution_optimisation(
                available_stations, optimisation_kwargs
            )
        elif optimisation_algorithm == "genetic":
            best_design, info = self._genetic_optimisation(
                available_stations, optimisation_kwargs
            )
        elif optimisation_algorithm == "sequential":
            raise NotImplementedError
        elif optimisation_algorithm == "random":
            raise NotImplementedError
        else:
            raise ValueError(
                "optimisation_algorithm must be one of ['differential_evolution', 'genetic']"
            )

        # convert all tuples to lists
        best_design = list(best_design.copy())
        best_design = [
            [sta_type, np.array(sta_data)] for sta_type, sta_data in best_design
        ]

        # remove array normal vector from the design, rather recompute it than expose it to the user
        for d in best_design:
            if "array" in d[0]:
                d[1] = d[1][:3]

        return best_design, info

    def _differential_evolution_optimisation(
        self, available_stations, optimisation_kwargs
    ):
        """
        Method to perform design optimisation using the differential evolution algorithm. For more information and a list of available keyword arguments, see the documentation of scipy.optimize.differential_evolution.

        Parameters
        ----------
        available_stations : tuple or list
            A tuple or a list that contains the available station types for the design optimisation. Each element of the tuple/list must be a tuple that contains the station types.
        optimisation_kwargs : dict
            A dictionary that contains the keyword arguments for the differential evolution algorithm. Default is {}. For a list of available keyword arguments, see the documentation of scipy.optimize.differential_evolution.
        """

        optimisation_kwargs.setdefault("maxiter", 100)
        optimisation_kwargs.setdefault("popsize", 15)
        optimisation_kwargs.setdefault("tol", 1e-3)
        optimisation_kwargs.setdefault("seed", 0)

        plot_fitness = optimisation_kwargs.pop("plot_fitness", False)
        progress_bar = optimisation_kwargs.pop("progress_bar", True)

        def fitness_function(design, *args):
            design_with_type = []
            for i, st in enumerate(available_stations):
                if "array" in st:
                    st_coords = self.design_points_dict["array"][int(design[i])]
                    design_with_type += [(st, st_coords)]
                else:
                    st_coords = self.design_points_dict["node"][int(design[i])]
                    design_with_type += [(st, st_coords)]

            if self.preexisting_design is not None:
                design_with_type = self.preexisting_design + design_with_type

            eig = self.design_criterion(design_with_type)

            return -eig

        from scipy.optimize import differential_evolution

        bounds = []
        for st in available_stations:
            if "array" in st:
                bounds.append((0, len(self.design_points_dict["array"]) - 1))
            else:
                bounds.append((0, len(self.design_points_dict["node"]) - 1))

        with tqdm(
            total=optimisation_kwargs["maxiter"],
            desc="GA progress",
            postfix={"EIG": 0.0},
            disable=not progress_bar,
        ) as pbar:
            EIG_history = []

            def callback(xk, convergence):
                pbar.set_postfix({"EIG": -fitness_function(xk)})
                pbar.update(1)
                EIG_history.append(-fitness_function(xk))

            result = differential_evolution(
                fitness_function,
                bounds=bounds,
                integrality=[True] * len(available_stations),
                callback=callback,
                **optimisation_kwargs,
            )

        best_design = []
        for i, st in enumerate(available_stations):
            if "array" in st:
                st_coords = self.design_points_dict["array"][int(result.x[i])]
                best_design.append((st, st_coords))
            else:
                st_coords = self.design_points_dict["node"][int(result.x[i])]
                best_design.append((st, st_coords))

        if plot_fitness:
            fig, ax = plt.subplots()
            ax.plot(EIG_history)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Fitness")
            ax.set_title("Fitness over iterations")
            plt.show()

        result.EIG_history = EIG_history

        return best_design, result

    def _genetic_optimisation(
        self,
        available_stations,
        optimisation_kwargs,
    ):
        """
        Method to perform design optimisation using the genetic algorithm. For more information and a list of available keyword arguments, see the documentation of pygad.GA.

        Parameters
        ----------
        available_stations : tuple or list
            A tuple or a list that contains the available station types for the design optimisation. Each element of the tuple/list must be a tuple that contains the station types.
        optimisation_kwargs : dict
            A dictionary that contains the keyword arguments for the genetic algorithm. Default is {}. For a list of available keyword arguments, see the documentation of pygad.GA.
        """

        try:
            import pygad
        except ImportError:
            raise ImportError("pygad is required for genetic optimisation")

        optimisation_kwargs.setdefault("num_generations", 100)
        optimisation_kwargs.setdefault("num_parents_mating", 4)
        optimisation_kwargs.setdefault("mutation_percent_genes", "default")
        optimisation_kwargs.setdefault("sol_per_pop", 64)
        optimisation_kwargs.setdefault("gene_type", int)
        optimisation_kwargs.setdefault("allow_duplicate_genes", False)
        optimisation_kwargs.setdefault("mutation_type", "random")
        optimisation_kwargs.setdefault("suppress_warnings", True)
        optimisation_kwargs.setdefault("random_seed", 1)

        optimisation_kwargs["num_genes"] = len(available_stations)

        plot_fitness = optimisation_kwargs.pop("plot_fitness", False)
        progress_bar = optimisation_kwargs.pop("progress_bar", True)

        gene_space = []
        for station in available_stations:
            if "array" in station:
                gene_space.append(
                    np.arange(len(self.design_points_dict["array"]), dtype=int).tolist()
                )
            else:
                gene_space.append(
                    np.arange(len(self.design_points_dict["node"]), dtype=int).tolist()
                )

        def fitness_function(ga_instance, solution, solution_idx):
            design_with_type = []
            for i, st in enumerate(available_stations):
                if "array" in st:
                    st_coords = self.design_points_dict["array"][solution[i]]
                    design_with_type += [(st, st_coords)]
                else:
                    st_coords = self.design_points_dict["node"][solution[i]]
                    design_with_type += [(st, st_coords)]

            if self.preexisting_design is not None:
                design_with_type = self.preexisting_design + design_with_type

            eig = self.design_criterion(design_with_type)

            return eig

        with tqdm(
            total=optimisation_kwargs["num_generations"],
            desc="GA progress",
            postfix={"EIG": 0.0},
            disable=not progress_bar,
        ) as pbar:

            def on_generation(ga_instance):
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "EIG": ga_instance.last_generation_fitness.max(),
                    }
                )

            ga_instance = pygad.GA(
                fitness_func=fitness_function,
                gene_space=gene_space,
                on_generation=on_generation,
                **optimisation_kwargs,
            )

            ga_instance.run()

        best_design_idx = ga_instance.best_solution()[0]
        best_design_EIG = ga_instance.last_generation_fitness.max()

        best_design = []
        for i, st in enumerate(available_stations):
            if "array" in st:
                best_design.append(
                    (st, self.design_points_dict["array"][best_design_idx[i]])
                )
            else:
                best_design.append(
                    (st, self.design_points_dict["node"][best_design_idx[i]])
                )

        if self.preexisting_design is not None:
            best_design = self.preexisting_design + best_design

        if plot_fitness:
            fitness = np.array(ga_instance.best_solutions_fitness)

            fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
            ax.plot(fitness, color="black", linewidth=2)
            ax.set_xlabel("Generation")
            ax.set_ylabel("DN criterion")

            ax.set_title("Fitness over generations")

            plt.show()

        out_info = dict(
            best_design_EIG=best_design_EIG,
            ga_instance=ga_instance,
        )

        return best_design, out_info
