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
import abc
import os
import reprlib
import functools
import logging
# Third party requirements
import numpy as np
# Local imports
import pasam._messages as msg
from pasam._settings import NP_ORDER, RLIB_MAXLIST, DIM_GANTRY, DIM_TABLE
from pasam._settings import LOG_FILE_NAME, LOG_MAP_NAME
from pasam.lattice import Lattice, LatticeMap, readfile_nodes_values
import pasam.utils as utl

# Constants
_rlib = reprlib.Repr()
_rlib.maxlist = RLIB_MAXLIST
_LOC_PASAM = os.path.dirname(os.path.abspath(__file__))
_LOC_FOLDER = os.path.join(_LOC_PASAM, '.log')
_LOG_FILE = os.path.join(_LOC_FOLDER, LOG_FILE_NAME)
logging.basicConfig(filename=_LOG_FILE, filemode='w', level=logging.DEBUG)


# Sampling Algorithm
class Sampler(abc.ABC):
    """Base class for sampling algorithms.

    Args
        lattice (Lattice or list of array_like): Object defining the
            computational lattice.
    """

    @abc.abstractmethod
    def __init__(self, lattice):
        # Also accept objects that can be used to generate a lattice
        if not isinstance(lattice, Lattice):
            lattice = Lattice(lattice)
        self._lattice = lattice

        # Make default prior maps (all equal probability, all allowed)
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
    """`GantryDominant2D` is a 2D gantry dominated trajectory sampler.

    It is designed to sample trajectories in two-dimensional domains as e.g.::

                --------------------------------------
                |            ****        0000000000  |
                | ***      **    *       0000000000  |
                **   ******       *        000000    |
                |                  *         00      |
      DIM_TABLE |      00           *                |
                |    000000          *****        **** Trajectory
                |  0000000000             ***  ***   |
                |  0000000000                **      |
                --------------------------------------
                             DIM_GANTRY

    where the zeros indicate prior conditioning. The trajectories are guided
    by a prior probability map describing higher and lower passage frequencies.

    Notes
        If the ratio is too small, the trajectory can not jump to upper or
        lower neighbors but only straight right (or left). Consider the
        following setup::

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

        where the nodes are chosen to have horizontal and vertical spread of
        1.0. Let us fix a ratio value of 0.5. This means that when moving from
        the first row to the second (distance=1.0) the trajectory is free to
        move up or down anything between +0.5 and -0.5 (distance*ratio = 0.5).
        We observe that in this case, it is not possible to jump from A to B or
        D, because the vertical difference is 1.0 which is higher than 0.5.
        Therefore, from A it is only possible to move to C.

        Considering the spread across two rows (distance=2.0), we have a
        vertical freedom of 1.0. This would technically allow to
        start at A and end on E. However, due to the discretization of the grid
        it would request to go the path A-C-E or A-B-E what would violate at
        least once the maximum ratio of 0.5.

        In such cases it is better to adapt the grid and design it more
        according to the desired ratio as e.g::

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

        In the latter setup we can keep a ratio of 0.5 and it would be allowed
        to jump from A to B or from A to D.

    Args
        lattice (Lattice or list of array_like): Object defining the
            computational lattice.
        ratio (float): Ratio describing the agility of the table with respect
            to the gantry.
        order (str {'random', 'max_val', 'max_random'}, optional): Order in
            which the trajectory points are sampled. 'random' randomly selects
            one gantry position after the other. 'max_val' treats the gantry
            positions according to the maximum value in the prior probability
            map. 'max_random' samples positions according to their maximum
            value in the prior probability map.
    """

    def __call__(self, conditions=None, validate=False, seed=None):
        logging.info(f'Start Path Sampling')

        # Set the random seed for reproducibility and write it to the logger
        if seed is None:
            seed = np.random.get_state()[1][0]
        np.random.seed(seed)
        logging.info(f'Numpy seed is {seed}')

        # Sample path and generate a `Trajectory` object
        points = self._compute_trajectory_points(conditions, validate)

        return Trajectory(points)

    def __init__(self, lattice, ratio=1.0, order='random'):
        # Check the ratio with respect to the lattice, for more information
        # consult the docstring in :meth:`_check_ratio`
        self._validate_ratio(lattice, ratio)

        # Initialize the values
        super().__init__(lattice)
        self._ratio = ratio
        if order not in ['random', 'max_val', 'max_random']:
            raise ValueError(msg.err3005(order))
        self._order = order
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
            self._validate_conditioning(map_)

        return map_

    def set_prior_cond(self, conditions=None, validate=False):
        """Sets the prior conditioning map of the sampler with the possibility
        to validate it.

        The prior probability map is updated whenever a new prior conditioning
        map is set.

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
        values = np.copy(map_.values)
        if energy:
            values = np.exp(-values)

        # Set value within self
        self._prior_prob = LatticeMap(self._lattice, values)

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

    def _compute_trajectory_points(self, conditions=None, validate=False):
        """Samples the trajectory points to generate a trajectory reaching
        through the gantry dimension of the lattice.

        Args:
            conditions (list, optional): Set of conditions of either strings
               (that indicate the path of a conditioning file), array_like
               objects (that represent conditioning points), or conditioning
               lattice maps.
            validate (bool, optional): Inspect the given conditioning with
                respect to the sampler settings and if necessary correct it.

        Returns:
            list of tuples: List of sampled points
        """
        # Initialization
        nodes_gantry = self._lattice.nodes[DIM_GANTRY]
        nodes_table  = self._lattice.nodes[DIM_TABLE]
        points = [None, ] * len(nodes_gantry)

        # Compute (and eventually validate) the global condition map
        if not conditions:
            conditions = []
        conditions.append(self.prior_cond)
        cond_map = self.compute_condition_map(conditions, validate)

        # Iteratively sample the points following a given gantry order
        for ipos in self._sampling_indices():
            # Compute the distribution in the slice at position `ipos`
            prior_slice = self._prior_prob.slice(DIM_GANTRY, ipos)
            cond_slice = cond_map.slice(DIM_GANTRY, ipos)
            distribution = prior_slice * cond_slice

            # Normalize the distribution values
            try:
                distribution = distribution.normalize_sum()
            except (ValueError, TypeError) as e:
                # Write logging file entry
                logging.error(f'No possible path passage at ipos={ipos}')
                # Save current permission map to a text file
                map_ = self._prior_prob * cond_map
                map_.to_txt(os.path.join(_LOC_FOLDER, LOG_MAP_NAME))
                raise e

            # Select table trajectory point according to the distribution
            p_choice = distribution.values
            pos_table = np.random.choice(nodes_table, p=p_choice)

            # Generate and set new trajectory point
            new_point = [None, ] * self._lattice.ndim
            new_point[DIM_GANTRY] = nodes_gantry[ipos]
            new_point[DIM_TABLE] = pos_table
            points[ipos] = tuple(new_point)

            # Update the restriction map by conditioning the new point
            cond_map *= self._cond_map_from_point(new_point)

        return points

    def _cond_map_from_point(self, components):
        """Generate a conditioning lattice map from a conditioning point."""
        # Initialization
        nodes = self._lattice.nodes
        nnodes_dim = self._lattice.nnodes_dim

        # Generate array with lowest dimension corresponding to the gantry
        dim = (nnodes_dim[DIM_GANTRY], nnodes_dim[DIM_TABLE])
        values = np.zeros(dim, dtype=bool)

        # Conical opening for the permitted area of the trajectory
        center = components[DIM_TABLE]
        con_rat = 2 * self._ratio  # The conical opening is symmetric
        pts = nodes[DIM_TABLE]

        # Loop in gantry direction through the lattice
        for inode, node in enumerate(nodes[DIM_GANTRY]):
            dist = abs(node - components[DIM_GANTRY])
            allowed = utl.within_conical_opening(center, dist, con_rat, pts)
            values[inode, allowed] = True

        # Correct dimension ordering if needed
        if DIM_GANTRY > DIM_TABLE:
            values = values.transpose()
        values = values.ravel(order=NP_ORDER)

        return LatticeMap(self._lattice, values)

    def _cond_map_from_str(self, file):
        """Loads a conditioning map from a saved file.

        !!! ATTENTION !!!
        This map invertes the values. AMS is used to indicate blocked nodes
        by `1` and allowed nodes by `0`. However, for the sampling it is more
        convenient to use it the other way around. Therefore, the values are
        inverted withing this function.
        """
        # Read the conditioning file
        nodes, values = readfile_nodes_values(file)

        # Invert the values (c.f. docstring)
        values = utl.ams_val_map_to_bool_map(values)

        # Check consistency of the lattices
        if Lattice(nodes) != self._lattice:
            raise ValueError(msg.err3001(file, self._lattice))

        return LatticeMap(nodes, values)

    def _linear_indices(self):
        """Returns the two dimensional array of the linear indices. The lower
        dimension of the produced array always considers the GANTRY dimension.

        The setting (DIM_GANTRY < DIM_TABLE and NP_ORDER='F') then produces the
        same outcome as (DIM_GANTRY > DIM_TABLE and NP_ORDER='C') because in
        both cases, the linear counter first considers (and increases) along
        the gantry dimension.

        Similar reasoning goes for the two settings (DIM_GANTRY > DIM_TABLE and
        NP_ORDER='F') and (DIM_GANTRY < DIM_TABLE and NP_ORDER='C') because the
        linear counter first considers (and increases) along the table
        dimension.
        """
        linear_indices = self._lattice.linear_indices()

        # As mentioned in the docstring, lower dimension corresponds to gantry
        if DIM_GANTRY > DIM_TABLE:
            linear_indices = linear_indices.transpose()

        return linear_indices

    def _sampling_indices(self):
        """This method is used to define the order in which the trajectory
        points are sampled.

        There are the following possibilities:
            - 'random': Randomly choose positions indices,
            - 'max_val': For each position, compute the maximum value in the
                prior probability map (i.e. np.max(..., axis=DIM_TABLE) and
                sort positions indices accordingly in descending order,
            - 'max_random': As above, but rather than ordering, we sample
               position indices according to these values.

        Returns:
            ndarray: Position indices.
        """
        order = self._order
        lattice = self._lattice

        def _max_vals():
            """Computing the maximum probability values in table direction.
            """
            values = (self._prior_prob * self.prior_cond).values
            values = values.reshape(lattice.nnodes_dim, order=NP_ORDER)
            return np.max(values, axis=DIM_TABLE)

        if order == 'random':
            pos = np.arange(lattice.nnodes_dim[DIM_GANTRY])
            np.random.shuffle(pos)

        elif order == 'max_val':
            pos = np.argsort(_max_vals())[::-1]

        elif order == 'max_random':
            prob = _max_vals()
            prob /= np.sum(prob)

            npos = lattice.nnodes_dim[DIM_GANTRY]
            pos = np.arange(npos)
            pos = np.random.choice(pos, size=npos, replace=False, p=prob)
        else:
            raise ValueError(msg.err3005(order))

        return pos

    def _validate_conditioning(self, map_):
        """Checks if a given conditioning map is valid with respect to the
        sampler settings.

        For doing so, we check if each node is potentially reachable by a
        trajectory that goes through the entire gantry range. If not, the node
        is set to be 'blocked'.

        Args:
            map_ (LatticeMap): Conditioning map
        """
        # The validity of the conditioning us checked by using an adjacency
        # graph to see if all nodes can be reached f
        if not self._graph:
            self.compute_adjacency_graph()
        graph = np.array(self._graph, copy=True)

        # Make the changes to an shallow copy of map_.values
        values = map_.values

        # Initialize set of nodes to be checked
        inodes_control = set(np.where(np.logical_not(values))[0])

        # Boundary nodes
        lin_ind = self._linear_indices()
        no_out = lin_ind[-1, :]     # boundary nodes with no outgoing edges
        no_in = lin_ind[0, :]       # boundary nodes with no incoming edges

        # Loop until there is no potential issue anymore
        while len(inodes_control) > 0:
            # Pop next node index to be checked
            inode = inodes_control.pop()

            # Get incoming and outgoing edges for the given node index `inode`
            outgoing = np.where(graph[inode, :])[0]
            incoming = np.where(graph[:, inode])[0]

            # Block node if there is no path passing through it
            issue_out = len(outgoing) == 0 and inode not in no_out
            issue_in = len(incoming) == 0 and inode not in no_in
            if not values[inode] or issue_out or issue_in:
                # Block the specific node
                values[inode] = False

                # Remove node edges from graph
                graph[inode, :] = False
                graph[:, inode] = False

                # Add the modified nodes to the control set
                inodes_control = inodes_control.union(outgoing)
                inodes_control = inodes_control.union(incoming)

    @staticmethod
    def _validate_ratio(lattice, ratio):
        """Checks if the ratio corresponds well with the lattice definitions.
        For mor information see the docstring of this class. The ratio can be
        interpreted as:

            ratio = d_table / d_gantry.

        In this function we test if the given ratio is sufficient to make
        the largest table gap when facing the smallest gantry gap.
        """
        nodes = lattice.nodes
        spacing_gantry = nodes[DIM_GANTRY][1:] - nodes[DIM_GANTRY][:-1]
        spacing_table = nodes[DIM_TABLE][1:] - nodes[DIM_TABLE][:-1]

        ratio_max = np.max(spacing_table) / np.min(spacing_gantry)
        if ratio < ratio_max:
            raise ValueError(msg.err3003)


# Trajectory Object
class Trajectory:
    """Definition of dynamic trajectories.

    Args:
        points (array_like of array_like): Sequence of trajectory points.

    Attributes:
        points (list of tuple): Sequence of trajectory points.

    """
    _TXT_SEP = '\t'

    def __eq__(self, other):
        return all([s == o for s, o in zip(self, other)])

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

    @classmethod
    def from_txt(cls, file):
        """Generates a trajectory object from .txt files.

        The form of the text file must be as follows::

            -------------------------
            |  <len>                |
            |  points[0]            |
            |  points[1]            |
            |     ...               |
            |  points[-1]           |
            -------------------------

        Args:
            file (str or pathlib.Path): File or filename.

        Returns:
            Trajectory: Trajectory from file.
        """
        return cls._ams_read_trajectory_from_txt(file)

    def to_txt(self, file):
        """Writes a trajectory to a text file.

        The form of the text file is as follows::

            -------------------------
            |  <len>                |
            |  points[0]            |
            |  points[1]            |
            |     ...               |
            |  points[-1]           |
            -------------------------

        Args:
            file (str or pathlib.Path): File or filename.

        Returns:
            None
        """
        self._ams_write_trajectory_to_txt(file, self.points)

    def to_latticemap(self, lattice, dtype=None):
        """Generates a boolean lattice map indicating the trajectory.

        Args:
            lattice (Lattice): Object defining the computational lattice.
        dtype (data-type, optional): The desired data-type for the map values.
            If not given, then the type will be determined as the minimum
            requirement by `numpy`.

        Returns:
            LatticeMap: Boolean trajectory according to a lattice grid.
        """
        values = np.zeros(lattice.nnodes, dtype=bool)

        # Make similar (closest node) computation iteratively for each point
        for components in self:
            linear_index = lattice.linear_index_from_point(components)
            values[linear_index] = True

        return LatticeMap(lattice, values, dtype=dtype)

    @classmethod
    def _ams_read_trajectory_from_txt(cls, file):
        """Read a trajectory from a .txt file according to the AMS guidelines.
        """

        lines = utl.readlines_(file, remove_blank_lines=True)

        # Number of points
        npts = int(lines[0])
        if npts != len(lines) - 1:
            raise ValueError(msg.err2000(file))

        # Iteratively read the lines and transform them to tuple of floats
        points = []
        for line in lines[1:]:
            points.append(tuple(map(float, line.split(cls._TXT_SEP))))

        return cls(points)

    def _ams_write_trajectory_to_txt(self, file, points):
        """Write a trajectory to a txt file according to the AMS guidelines.
        """
        with open(file, 'w+') as tfile:
            tfile.write(f'{len(points)}\n')
            for pt in points:
                tfile.write(self._TXT_SEP.join([f'{p}' for p in pt]) + '\n')
