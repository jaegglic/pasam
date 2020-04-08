# -*- coding: utf-8 -*-
"""This module defines the trajectory sampling algorithms.
"""

# -------------------------------------------------------------------------
#   Authors: Stefanie Marti and Christoph Jaeggli
#   Institute: Insel Data Science Center, Insel Gruppe AG
#
#   MIT License
#   Copyright (c) 2020 Stefanie Marti, Christoph Jaeggli
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

# Standard library
import functools
# Third party requirements
# Local imports
from pasam._settings import DIM_GANTRY, DIM_TABLE
from pasam.lattice import *

# Constants
_rlib = reprlib.Repr()
_rlib.maxlist = RLIB_MAXLIST


# Sampling Algorithm
class Sampler(abc.ABC):
    """Base class for sampling algorithms.
    """

    @abc.abstractmethod
    def __init__(self, lattice: Lattice):
        self._lattice = lattice
        ones = np.ones(lattice.nnodes_dim)
        self._prior_prob = LatticeMap(lattice, ones, dtype=int)
        self._prior_cond = LatticeMap(lattice, ones, dtype=bool)

    @abc.abstractmethod
    def __call__(self, conditions=None, validate=False, seed=None):
        """Executes the sampling algorithm and produces a possible trajectory.

        Args:
            conditions (list, optional): Set of conditions of either strings
               that indicate the path of a conditioning file or array_like
               objects that represent conditioning points.
            validate (bool, optional): Inspect the given conditioning with
                respect to the sampler settings and if necessary correct it.
            seed (int, optional): Seed used by the random procedure.

        Returns:
            Trajectory: Sampled trajectory
        """

    @property
    def prior_cond(self):
        """Returns the prior conditioning map.

        Returns:
            LatticeMap
        """
        return self._prior_cond

    @property
    def prior_prob(self):
        """Returns the prior probability map.

        Returns:
            LatticeMap
        """
        return self._prior_prob

    @abc.abstractmethod
    def set_prior_cond(self, conditions=None, validate=False) -> None:
        """Sets the prior conditioning map.

        Args:
            conditions (list, optional): Set of conditions.
            validate (bool, optional): Inspect the given conditioning with
                respect to the sampler settings and if necessary correct it.

        Returns:
            None
        """

    @abc.abstractmethod
    def set_prior_prob(self, map_: LatticeMap, energy: bool = False) -> None:
        """Set's the prior probability map.

        Args:
            map_ (LatticeMap): Prior probability or energy map.
            energy (bool): Indicates whether the energy (True) or the
                probability (False) values are provided.

        Returns:
            None
        """


class GantryDominant2D(Sampler):
    """`GantryDominant2D` is a 2D gantry dominated trajectory movement sampler.
    """

    def __call__(self, conditions=None, validate=False, seed=None):
        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Combine the `conditions` with the prior conditioning
        if not conditions:
            conditions = []
        conditions.append(self._prior_cond)
        cond_map = self.compute_condition_map(conditions, validate)

        # Sample path and generate a `Trajectory` object
        points = self._sample_trajectory_points(cond_map)
        trajectory = Trajectory(points)

        return trajectory

    def __init__(self, lattice, ratio=1.0):
        # Check the ratio with respect to the lattice, for more information
        # consult the docstring in :meth:`_check_ratio`
        self._check_ratio(lattice, ratio)

        # Initialize the values
        super().__init__(lattice)
        self._ratio = ratio
        self._graph = None

    def __repr__(self):
        cls_name = type(self).__name__
        return f'{cls_name}(lattice={self._lattice}, ' \
               f'ratio={repr(self._ratio)})'

    def __str__(self):
        return self.__repr__()

    # Public Methods
    def compute_adjacency_graph(self):
        """Computes and stores the adjacency graph of the sampler.

        Not that we consider a directed graph where the edges always go from
        the lower to the higher nodes such that the graph::

                2---3
               / \  |
              /   \ |
             /     \|
            1---4---5

        would give the adjacency matrix::

              | 1   2   3   4   5
            --|--------------------
            1 |     *       *
            2 |         *       *
            3 |                 *
            4 |                 *
            5 |

        In this matrix, the rows represent 'arriving' and the columns
        'departing' edges.
        """
        graph = self._adjacency_graph()
        self._graph = graph

    def compute_condition_map(self, conditions=None, validate=False):
        """Computes a lattice map representing the union of a set of
        conditions.

        Args:
            conditions (list, optional): Set of conditions of either strings
               (that indicate the path of a conditioning file), array_like
               objects (that represent conditioning points), or conditioning
               lattice maps.
            validate (bool, optional): Inspect the given conditioning with
                respect to the sampler settings and if necessary correct it.

        Returns:
            LatticeMap: Conditioning map.
        """
        if not conditions:
            ones = np.ones(self._lattice.nnodes_dim, dtype=bool)
            return LatticeMap(self._lattice, ones)

        # Iteratively transform all conditions into lattice maps
        cond_maps = []
        for cond in conditions:
            if isinstance(cond, str):               # conditioning FILE
                file = cond
                cond_map = self._cond_map_from_str(file)
            elif isinstance(cond, LatticeMap):      # conditioning MAP
                cond_map = cond
            elif len(cond) == self._lattice.ndim:   # conditioning POINT
                components = np.asarray(cond)
                cond_map = self._cond_map_from_point(components)
            else:
                raise ValueError(msg.err3004(cond))
            cond_maps.append(cond_map)

        # Reduce set of conditioning by multplication of all maps
        map_ = functools.reduce(lambda a, b: a * b, cond_maps)

        # Validate final conditioning map
        if validate:
            values = self._validate_cond_values(map_.values)
            map_.values = values

        return map_

    def set_prior_cond(self, conditions=None, validate=False):
        """Sets the prior conditioning attribute of the sampler with the
        possibility to validate it.

        Args:
            conditions (list, optional): Set of conditions of either strings
               (that indicate the path of a conditioning file), array_like
               objects (that represent conditioning points), or conditioning
               lattice maps.
            validate (bool, optional): Inspect the given conditioning with
                respect to the sampler settings and if necessary correct it.

        Returns:
            None
        """
        # Get conditioning map
        map_ = self.compute_condition_map(conditions, validate)

        # Minor consistency check
        if self._lattice != map_.lattice:
            raise ValueError(msg.err3000(self._lattice, map_.lattice))

        # Set value within self
        self._prior_cond = map_

    def set_prior_prob(self, map_, energy=False):
        """Sets the prior probability attribute of the sampler.

        It is possible to provide prior probability values (`energy=False`) or
        probability energy values (`energy=True`). In the latter case, the
        values are transformed to un-normalized probability measures by::

            prior_prob.values = exp(-map_.values)

        where `exp` denotes the exponential function.

        Args:
            map_ (LatticeMap): Prior map
            energy (bool): Indicates if the prior map represents probability
                values (false) or probability energy (true).

        Returns:
            None
        """
        # Minor consistency checks
        if self._lattice != map_.lattice:
            raise ValueError(msg.err3000(self._lattice, map_.lattice))

        # If the map represents the energy, transform it to probability values
        if energy:
            values = np.exp(-map_.values)
            map_ = LatticeMap(self._lattice, values)

        # Set value within self
        self._prior_prob = map_

    # Private Methods
    def _adjacency_graph(self):
        """Computes an adjacency graph based on a computational lattice.

        The entry (i, j) of the graph matrix is non-zero if there is a
        connection starting from node i and joining j. For more information see
        docstring of :meth:`compute_adjacency_graph`.

        Returns
            ndarray: Matrix representing the directed graph
        """
        # Gantry/Table nodes
        lattice = self._lattice
        nodes_gantry = lattice.nodes[DIM_GANTRY]
        nodes_table = lattice.nodes[DIM_TABLE]

        # Get an array containing the linear indices
        lin_ind = self._linear_indices()
        shape, dim = lin_ind.shape, (len(nodes_gantry), len(nodes_table))
        if shape != dim:
            raise ValueError(msg.err3002(shape, dim))

        # Iteratively go through the nodes and generate adjacency graph
        ratio = 2 * self._ratio  # The opening is symmetric
        pts = nodes_table
        graph = np.zeros((lattice.nnodes,) * 2, dtype=bool)
        for igan, node_g in enumerate(nodes_gantry[:-1]):

            # Conical opening for the permitted area of the trajectory
            dist = abs(node_g - nodes_gantry[igan + 1])
            for itab, node_t in enumerate(nodes_table):
                cntr = node_t
                allowed = utl.within_conical_opening(cntr, dist, ratio, pts)

                from_node = lin_ind[igan, itab]
                to_nodes = lin_ind[igan + 1, allowed]
                graph[from_node, to_nodes] = True

        return graph

    # TODO Refactor `_check_ratio`
    @staticmethod
    def _check_ratio(lattice, ratio):
        """If the ratio is too small, the trajectory can not jump to upper or
        lower neighbors but only straight right (or left) one::

                     ________________________________
                    |       |       |       |       |
                    |       |   B   |   E   |   H   |
                    |_______|_______|_______|_______|
                    |       |       |       |       |
         DIM_TABLE  |   A   |   C   |   F   |   I   |
                    |_______|_______|_______|_______|
                    |       |       |       |       |
                    |       |   D   |   G   |   J   |
                    |_______|_______|_______|_______|

                              DIM_GANTRY

        If the nodes have horizontal and vertical spread of 1.0 and the ratio
        is 0.5 we see that it is not possible to jump from A to B or D but only
        to C. But also to jump from A to C to E is not very meaningful because
        it would mean that between C and E we have a virtual ratio of 1.0. In
        these cases it is better to adapt the node spreading to the ratio and
        take a lattice such that::

                     ________________________________
                    |               |               |
                    |               |       B       |
                    |_______________|_______________|
                    |               |               |
                    |       A       |       C       |
                    |_______________|_______________|
                    |               |               |
                    |               |       D       |
                    |_______________|_______________|

        and then still take a ratio of 0.5. In the latter setup it would
        perfectly be fine to jump from A to B or from A to D.
        """
        # TODO write this issue in the documentation
        nodes = lattice.nodes
        spacing_gantry = nodes[DIM_GANTRY][1:] - nodes[DIM_GANTRY][:-1]
        spacing_table = nodes[DIM_TABLE][1:] - nodes[DIM_TABLE][:-1]

        ratio_min = np.max(spacing_table) / np.min(spacing_gantry)
        if ratio < ratio_min:
            raise ValueError(msg.err3003)

    # TODO Refactor `_cond_map_from_point`
    def _cond_map_from_point(self, components):
        # Initialization
        lattice = self._lattice
        if components is None:
            return np.ones(lattice.nnodes_dim, dtype=bool)
        nodes = lattice.nodes
        dim = (lattice.nnodes_dim[DIM_GANTRY], lattice.nnodes_dim[DIM_TABLE])
        values = np.zeros(dim, dtype=bool)

        # Conical opening for the permitted area of the trajectory
        cntr = components[DIM_TABLE]
        ratio = 2 * self._ratio  # The opening is symmetric
        pts = nodes[DIM_TABLE]

        # Loop in gantry direction through the lattice
        for inode, node in enumerate(nodes[DIM_GANTRY]):
            dist = abs(node - components[DIM_GANTRY])
            allowed = utl.within_conical_opening(cntr, dist, ratio, pts)
            values[inode, allowed] = True

        # Correct dimension ordering if needed
        if DIM_GANTRY > DIM_TABLE:
            values = values.transpose()

        return LatticeMap(lattice, values.ravel(order=NP_ORDER))

    # TODO Refactor `_cond_map_from_str`
    def _cond_map_from_str(self, file):
        # Read the conditioning file and invert the values
        nodes, vals = readfile_nodes_values(file)
        values = utl.ams_val_map_to_bool_map(vals)  # !! Values are inverted !!

        # Check consistency of the lattices
        if Lattice(nodes) != self._lattice:
            raise ValueError(msg.err3001(file, self._lattice))

        return LatticeMap(nodes, values)

    # TODO Refactor `_linear_indices`
    def _linear_indices(self):
        """Returns the two dimensional array of the linear indices where the
        lower dimension always considers the GANTRY dimension."""
        nnodes_dim = self._lattice.nnodes_dim
        nnodes = self._lattice.nnodes
        lin_ind = np.arange(nnodes).reshape(nnodes_dim, order=NP_ORDER)

        if DIM_GANTRY > DIM_TABLE:
            lin_ind = lin_ind.transpose()

        return np.asarray(lin_ind, dtype='int32')

    # TODO Refactor `_validate_cond_values`
    # TODO maybe rename it into `_valide_condition_array`
    def _validate_cond_values(self, values):
        """Returns a valid condition array respecting graph connectivity and
        restrictions from a set of values."""

        values = np.array(values, dtype=bool, copy=True).ravel(order=NP_ORDER)
        if not self._graph:
            self.compute_adjacency_graph()
        graph = np.array(self._graph, copy=True)

        # Initialize set of nodes to be checked
        inodes_control = set(np.where(np.logical_not(values))[0])

        # Boundary nodes
        lin_ind = self._linear_indices()
        no_outgoing = lin_ind[-1, :]
        no_incoming = lin_ind[0, :]

        # Loop until there is no potential issue anymore
        while len(inodes_control) > 0:
            inode = inodes_control.pop()

            outgoing = np.where(graph[inode, :])[0]
            incoming = np.where(graph[:, inode])[0]

            # Remove node if there is no path passing through it
            outgoing_issue = len(outgoing) == 0 and inode not in no_outgoing
            incoming_issue = len(incoming) == 0 and inode not in no_incoming
            if not values[inode] or outgoing_issue or incoming_issue:
                # Remove node edges from graph
                graph[inode, :] = False
                graph[:, inode] = False

                # Add the modified nodes to the list to be checked
                inodes_control = inodes_control.union(outgoing)
                inodes_control = inodes_control.union(incoming)

                # Make it unpermitted
                values[inode] = False

        return values

    # TODO Refactor `_sample_trajectory_points`
    def _sample_trajectory_points(self, perm_map):
        lattice = self._lattice
        ndim = lattice.ndim
        ntraj = lattice.nnodes_dim[DIM_GANTRY]
        traj_points = np.array([None for _ in range(ntraj)])
        ind_to_do = np.array([True for _ in range(ntraj)], dtype='bool')

        gantry_ind = np.arange(len(ind_to_do))
        rem_nodes = [n for i, n in enumerate(lattice.nodes)
                     if i != DIM_GANTRY]
        rem_nodes = utl.cartesian_product(*rem_nodes)
        while np.any(ind_to_do):
            ind_gantry_pos = np.random.choice(gantry_ind[ind_to_do])

            prior_slice = self._prior_prob.slice(DIM_GANTRY, ind_gantry_pos)
            perm_slice = perm_map.slice(DIM_GANTRY, ind_gantry_pos)

            # import matplotlib.pyplot as plt
            # plt.imshow(np.reshape((self._prior_prob * perm_map).values, (180, 90),
            #                       order='F').transpose(), origin='lower')
            # plt.plot([ind_gantry_pos,]*2, [0, 89], 'r--')
            # plt.show()

            if np.sum(perm_slice.values) >= 1:
                distribution = (prior_slice * perm_slice).values
                try:
                    distribution = distribution / np.sum(distribution)
                except (ZeroDivisionError, FloatingPointError,
                        RuntimeWarning, RuntimeError):
                    # TODO do something about that case!!
                    raise ValueError('HERE WE SHOULD TAKE THE PERMITTED REGION WITHOUT'
                                     'THE PRIOR DISTRIBUTION AND SAMPLE UNIFORMLY IN THERE')
                distribution = distribution / np.sum(distribution)
            else:
                # TODO also react to this situation
                import matplotlib.pyplot as plt
                plt.imshow(np.reshape((self._prior_prob * perm_map).values, (180, 90),
                                      order='F').transpose(), origin='lower')
                plt.show()
                raise ValueError('TODO Error message for not possible settings '
                                 '(because there is no connected trajectory possible anymore)')

            ind = np.random.choice(np.arange(len(distribution)),
                                   p=distribution)
            pos_slice = rem_nodes[ind]

            traj_point = np.array([None,] * ndim)
            traj_point[DIM_GANTRY] = lattice.nodes[DIM_GANTRY][ind_gantry_pos]
            traj_point[[i for i in range(ndim) if i != DIM_GANTRY]] = pos_slice

            components = tuple(traj_point)
            cond_map = self._cond_map_from_point(components)
            perm_map *= cond_map

            traj_points[ind_gantry_pos] = traj_point
            ind_to_do[ind_gantry_pos] = False
        return traj_points


# Trajectory Object
class Trajectory:
    """Definition of dynamic trajectories.

    Args:
        points (array_like of array_like): Sequence of trajectory points.

    Attributes:
        points (list of tuple): Sequence of trajectory points.

    """

    def __getitem__(self, key):
        return self.points[key]

    def __init__(self, points):
        self.points = list(tuple(p) for p in points)

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        cls_name = type(self).__name__
        points_repr = _rlib.repr(self.points)
        return f'{cls_name}(points={points_repr})'

    def __str__(self):
        return self.__repr__()

    def to_txt(self, file):
        self._ams_write_trajectory_to_txt(file, self.points)

    @staticmethod
    def _ams_write_trajectory_to_txt(fname, points):
        """Write a trajectory to a txt file according to the AMS guidelines.
        """
        with open(fname, 'w+') as tfile:
            tfile.write(f'{len(points)}\n')
            for pt in points:
                tfile.write('\t'.join([f'{p}' for p in pt]) + '\n')
