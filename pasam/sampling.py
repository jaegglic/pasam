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


# TODO refactor this entire module
# Sampling Algorithm
class Sampler(abc.ABC):
    """Base class for sampling algorithms.
    """

    @abc.abstractmethod
    def __init__(self, lattice: Lattice):
        self._lattice = lattice
        values = np.ones(lattice.nnodes_dim)
        self._prior_prob = LatticeMap(lattice, values)
        self._prior_cond = LatticeMap(lattice, values)

    @abc.abstractmethod
    def __call__(self, conditions=None, inspect=False):
        """Executes the sampling algorithm and produces a possible trajectory.

        Args:
            conditions (list, optional): Set of conditions of either strings
               that indicate the path of a conditioning file or array_like
               objects that represent conditioning points.
            inspect (bool, optional): Inspect/transform the given conditioning.

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
    def set_prior_cond(self, conditions=None, inspect=False) -> None:
        """Sets the prior conditioning map used in the sampling algorithm.

        Args:
            conditions (list, optional): Set of conditions.
            inspect (bool, optional): Inspect/transform the given conditioning.

        Returns:
            None
        """

    @abc.abstractmethod
    def set_prior_prob(self, map_: LatticeMap, energy: bool = False) -> None:
        """Set's the prior probability map used in the sampling algorithm.

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

    def __call__(self, conditions=None, inspect=False):
        # Get complete conditioning map
        if not conditions:
            conditions = []
        conditions.append(self._prior_cond)
        cond_map = self.cond_map(conditions, inspect)

        # Sample path return it as a `Trajectory` object
        points = self._sample_trajectory_points(cond_map)
        trajectory = Trajectory(points)
        return trajectory

    def __init__(self, lattice, ratio=1.0):
        self._check_ratio(lattice, ratio)

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

    def cond_map(self, conditions, inspect):
        """Computes a lattice map representing the union of all conditions.

        Args:
            conditions (list, optional): Set of conditions of either strings
               that indicate the path of a conditioning file or array_like
               objects that represent conditioning points.
            inspect (bool, optional): Inspect/transform the given conditioning.

        Returns:
            LatticeMap: Conditioning map.
        """

        if conditions:
            # Iteratively load all the different conditioning maps
            cond_maps = []
            for cond in conditions:
                if isinstance(cond, str):
                    file = cond
                    cond_map = self._cond_map_from_str(file)
                elif isinstance(cond, LatticeMap):
                    cond_map = cond
                else:
                    components = np.asarray(cond)
                    cond_map = self._cond_map_from_point(components)
                cond_maps.append(cond_map)

            # Multiply through the set of conditioning maps
            map_ = functools.reduce(lambda a, b: a * b, cond_maps)

            # Check/transform to a valid conditioning
            if inspect:
                values = self._validate_cond_values(map_.values)
                map_.values = values
        else:
            values = np.ones(self._lattice.nnodes_dim, dtype=bool)
            map_ = LatticeMap(self._lattice, values)
        return map_

    def set_adjacency_graph(self):
        """Computes and stores the adjacency graph of the sampler."""
        self._graph = self._adjacency_graph()

    def set_prior_cond(self, conditions=None, inspect=True):
        # Get conditioning map
        map_ = self.cond_map(conditions, inspect)

        # Minor consistency check
        if self._lattice != map_.lattice:
            raise ValueError(msg.err3000(self._lattice, map_.lattice))

        # Set value within self
        self._prior_cond = map_

    def set_prior_prob(self, map_, energy=False):
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
        connection starting from node i and joining j. The linear indices i, j
        are ordered according to the order specified in `_settings`.
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

    def _cond_map_from_str(self, file):
        # Read the conditioning file and invert the values
        nodes, vals = readfile_nodes_values(file)
        values = utl.ams_val_map_to_bool_map(vals)  # !! Values are inverted !!

        # Check consistency of the lattices
        if Lattice(nodes) != self._lattice:
            raise ValueError(msg.err3001(file, self._lattice))

        return LatticeMap(nodes, values)

    def _linear_indices(self):
        """Returns the two dimensional array of the linear indices where the
        lower dimension always considers the GANTRY dimension."""
        nnodes_dim = self._lattice.nnodes_dim
        nnodes = self._lattice.nnodes
        lin_ind = np.arange(nnodes).reshape(nnodes_dim, order=NP_ORDER)

        if DIM_GANTRY > DIM_TABLE:
            lin_ind = lin_ind.transpose()

        return np.asarray(lin_ind, dtype='int32')

    def _validate_cond_values(self, values):
        """Returns a valid array respecting graph connectivity and
        restrictions from a set of values."""

        values = np.array(values, dtype=bool, copy=True).ravel(order=NP_ORDER)
        if not self._graph:
            self.set_adjacency_graph()
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
                    raise ValueError('HERE WE SHOULD TAKE THE PERMITTED REGION WITHOUT'
                                     'THE PRIOR DISTRIBUTION AND SAMPLE UNIFORMLY IN THERE')
                distribution = distribution / np.sum(distribution)
            else:
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
        points (array_like): Sequence of trajectory points.

    Attributes:
        points (list): Sequence of trajectory points.

    """

    def __init__(self, points):
        self.points = list(points)

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
