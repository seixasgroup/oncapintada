# -*- coding: utf-8 -*-
# file: bonds_model.py

# This code is part of Onça-pintada.
# MIT License
#
# Copyright (c) 2025 Leandro Seixas Rocha <leandro.seixas@proton.me> 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ase import Atoms
from typing import Optional, List, Dict, Tuple, Any

class Lattice2D:
    """
    A class to represent a 2D lattice with various topologies.
    
    This class builds a 2D lattice, establishes nearest-neighbor
    connections with periodic boundary conditions (PBC), and stores 
    atom information in a Pandas DataFrame.
    
    Topologies:
    - 'square', 'rectangular': (N*M) atoms, 4-connected.
    - 'triangular', 'distorted_triangular': (N*M) atoms, 6-connected.
    - 'honeycomb', 'distorted_honeycomb': (N*M*2) atoms, 3-connected.
      'a' and 'b' are used for bond lengths.
    
    Attributes:
        N (int): Number of grid units in the first dimension.
        M (int): Number of grid units in the second dimension.
        topology (str): The lattice topology.
        pbc (bool): Whether to use Periodic Boundary Conditions.
        a (float): Primary lattice constant or bond length.
        b (float): Secondary lattice constant or bond length.
        num_atoms_per_cell (int): 1 for simple lattices, 2 for honeycomb.
        total_atoms (int): Total number of atoms in the system.
        nodes_df (pd.DataFrame): DataFrame holding all atom info.
        bonds (List[Tuple[int, int, str]]): List of unique bonds.
        atoms_obj (ase.Atoms): An ASE Atoms object.
        cell_vectors (np.ndarray): The 2D cell vectors for PBC.
    """
    
    def __init__(self, 
                 N: int, 
                 M: int, 
                 topology: str = 'square', 
                 pbc: bool = True, 
                 a: float = 1.0, 
                 b: float = 1.0):
        """
        Initializes the 2D lattice.

        Args:
            N (int): Grid units in the i-direction.
            M (int): Grid units in the j-direction.
            topology (str): The lattice topology.
            pbc (bool): Use Periodic Boundary Conditions. Default is True.
            a (float): Primary lattice constant/bond length.
            b (float): Secondary lattice constant/bond length.
        """
        self.N = N
        self.M = M
        self.pbc = pbc
        self.a = a
        self.b = b
        
        valid_topologies = [
            'square', 'rectangular', 
            'triangular', 'distorted_triangular',
            'honeycomb', 'distorted_honeycomb'
        ]
        if topology not in valid_topologies:
            raise ValueError(f"Topology must be one of {valid_topologies}")
        self.topology = topology

        # Honeycomb lattices have a 2-atom basis
        if 'honeycomb' in self.topology:
            self.num_atoms_per_cell = 2
        else:
            self.num_atoms_per_cell = 1
            
        self.total_atoms = self.N * self.M * self.num_atoms_per_cell
        
        # --- Auto-set constants for regular lattices if not specified ---
        if self.topology == 'square':
            self.b = self.a
        elif self.topology == 'triangular':
            # Use 'a' as bond length, 'b' as row separation
            self.b = self.a * np.sqrt(3) / 2.0
        elif self.topology == 'honeycomb':
            # 'a' and 'b' are bond lengths. For regular, they are equal.
            self.b = self.a 

        self.nodes_df: Optional[pd.DataFrame] = None
        self.bonds: List[Tuple[int, int, str]] = []
        self.atoms_obj: Optional[Atoms] = None
        self.cell_vectors: Optional[np.ndarray] = None

    def _get_index(self, i: int, j: int, sublat: int = 0) -> int:
        """
        Converts (row, col, sublattice) to a flat atom index.
        Wraps i and j indices based on PBC.
        """
        # Apply periodic boundary conditions to grid indices
        if self.pbc:
            i_wrapped = i % self.N
            j_wrapped = j % self.M
        else:
            i_wrapped, j_wrapped = i, j
            
        return (i_wrapped * self.M + j_wrapped) * self.num_atoms_per_cell + sublat

    def _get_coordinates(self, i: int, j: int, sublat: int = 0) -> Tuple[float, float]:
        """
        Gets (x, y) coordinates for a given (i, j, sublat) node index
        based on the lattice topology and constants.
        """
        if self.topology in ['square', 'rectangular']:
            return (j * self.a, i * self.b)
            
        elif self.topology in ['triangular', 'distorted_triangular']:
            # Skewed grid mapping
            return (j * self.a + 0.5 * i * self.a, i * self.b)
            
        elif self.topology in ['honeycomb', 'distorted_honeycomb']:
            # Graphene-like 2-atom basis on a triangular lattice
            # 'a' is h-bond, 'b' is v-bond (for distorted)
            # v1 = (1.5a, 0.5 * sqrt(3) * b)
            # v2 = (1.5a, -0.5 * sqrt(3) * b)
            
            # This is a robust coordinate mapping
            x_base = j * 1.5 * self.a
            y_base = i * self.b * np.sqrt(3)
            
            if j % 2 == 1:
                y_base += self.b * np.sqrt(3) / 2.0

            if sublat == 0: # Atom A
                return (x_base, y_base)
            else: # Atom B (bonded to A)
                return (x_base + self.a, y_base)

    def _get_neighbors(self, i: int, j: int, sublat: int = 0) -> List[Tuple[int, int, int]]:
        """
        Gets valid (i, j, sublat) neighbor indices for a given node
        based on the defined topology. Does NOT wrap indices here.
        """
        neighbors_ijs = []
        
        if self.topology in ['square', 'rectangular']:
            neighbors_ijs = [(i+1, j, 0), (i-1, j, 0), (i, j+1, 0), (i, j-1, 0)]
            
        elif self.topology in ['triangular', 'distorted_triangular']:
            neighbors_ijs = [
                (i, j+1, 0), (i, j-1, 0),     # Horizontal
                (i-1, j, 0), (i-1, j+1, 0), # Diagonal-Up
                (i+1, j, 0), (i+1, j-1, 0)  # Diagonal-Down
            ]
            
        elif self.topology in ['honeycomb', 'distorted_honeycomb']:
            # 3-connected, 2-atom basis.
            # A (sublat 0) connects to 3 B's (sublat 1)
            # B (sublat 1) connects to 3 A's (sublat 0)
            
            # This logic matches the _get_coordinates
            if j % 2 == 0:
                if sublat == 0: # Atom A
                    neighbors_ijs = [(i, j, 1), (i-1, j, 1), (i, j-1, 1)]
                else: # Atom B
                    neighbors_ijs = [(i, j, 0), (i+1, j, 0), (i, j+1, 0)]
            else: # j is odd
                if sublat == 0: # Atom A
                    neighbors_ijs = [(i, j, 1), (i, j-1, 1), (i+1, j, 1)]
                else: # Atom B
                    neighbors_ijs = [(i, j, 0), (i, j+1, 0), (i-1, j, 0)]
                    
        return neighbors_ijs

    def build_lattice(self, atom_types: Optional[np.ndarray] = None):
        """
        Builds the lattice nodes and their nearest-neighbor connections.
        
        Populates self.nodes_df, self.bonds, and self.atoms_obj.
        Assigns specific bond types ('_h', '_d', '_v') for distorted lattices.
        """
        
        # Handle atom_types array
        symbols = np.full(self.total_atoms, 'X', dtype=object)
        if atom_types is not None:
            if atom_types.shape == (self.N, self.M):
                # Apply (N,M) array to all atoms in unit cells
                for i in range(self.N):
                    for j in range(self.M):
                        for sublat in range(self.num_atoms_per_cell):
                            idx = self._get_index(i, j, sublat)
                            symbols[idx] = atom_types[i, j]
            elif atom_types.shape == (self.total_atoms,):
                symbols = atom_types
            else:
                raise ValueError("atom_types shape is incompatible")

        # 1. Create all nodes (sites)
        nodes_data = []
        for i in range(self.N):
            for j in range(self.M):
                for sublat in range(self.num_atoms_per_cell):
                    idx = self._get_index(i, j, sublat)
                    x, y = self._get_coordinates(i, j, sublat)
                    nodes_data.append({
                        'site_index': idx,
                        'atom_type': symbols[idx],
                        'i': i, 'j': j, 'sublat': sublat,
                        'x': x, 'y': y,
                        'connections': []
                    })
        self.nodes_df = pd.DataFrame(nodes_data).set_index('site_index')

        # 2. Build connections (bonds)
        self.bonds = [] 
        all_bonds_set = set() 
        
        for i in range(self.N):
            for j in range(self.M):
                for sublat in range(self.num_atoms_per_cell):
                    
                    current_idx = self._get_index(i, j, sublat)
                    current_atom_type = self.nodes_df.loc[current_idx, 'atom_type']
                    
                    # Get neighbors defined by the topology
                    neighbors_ijs = self._get_neighbors(i, j, sublat)
                    
                    node_connections_list = []
                    
                    for ni, nj, n_sublat in neighbors_ijs:
                        
                        # --- Check for finite boundaries (if pbc=False) ---
                        if not self.pbc:
                            if not (0 <= ni < self.N and 0 <= nj < self.M):
                                continue # This neighbor is outside the finite grid
                        
                        # Get flat index, which wraps if pbc=True
                        neighbor_idx = self._get_index(ni, nj, n_sublat)
                        
                        neighbor_atom_type = self.nodes_df.loc[neighbor_idx, 'atom_type']
                        
                        node_connections_list.append(neighbor_idx)
                        bond_tuple = tuple(sorted((current_idx, neighbor_idx)))
                        
                        if bond_tuple not in all_bonds_set:
                            base_bond_type = "-".join(sorted([current_atom_type, neighbor_atom_type]))
                            bond_type = base_bond_type
                            
                            # --- Add direction-specific tags for distorted lattices ---
                            if self.topology == 'distorted_triangular':
                                if ni == i: bond_type = f"{base_bond_type}_h" # Horizontal
                                else: bond_type = f"{base_bond_type}_d" # Diagonal
                                    
                            elif self.topology == 'distorted_honeycomb':
                                # Check if it's the A-B bond within the (i,j) cell
                                # This is the "vertical" bond in our coordinate system
                                if ni == i and nj == j: bond_type = f"{base_bond_type}_v" 
                                else: bond_type = f"{base_bond_type}_h"
                            
                            self.bonds.append((bond_tuple[0], bond_tuple[1], bond_type))
                            all_bonds_set.add(bond_tuple)
                    
                    self.nodes_df.at[current_idx, 'connections'] = node_connections_list
                
        # 3. Create ASE Atoms object
        positions = self.nodes_df[['x', 'y']].values
        positions_3d = np.hstack([positions, np.zeros((len(positions), 1))])
        
        # --- Define Cell Vectors ---
        cell_z = 10.0 # Default 2D cell height
        v1, v2, v3 = [0,0,0], [0,0,0], [0,0,cell_z]
        
        if self.topology in ['square', 'rectangular']:
            v1 = [self.M * self.a, 0, 0]
            v2 = [0, self.N * self.b, 0]
        elif self.topology in ['triangular', 'distorted_triangular']:
            v1_cell = np.array([self.M * self.a, 0])
            v2_cell = np.array([0.5 * self.N * self.a, self.N * self.b])
            v1 = [v1_cell[0], v1_cell[1], 0]
            v2 = [v2_cell[0], v2_cell[1], 0]
        elif self.topology in ['honeycomb', 'distorted_honeycomb']:
            # Get max coordinates to define the cell
            max_x = self.nodes_df['x'].max()
            max_y = self.nodes_df['y'].max()
            min_x = self.nodes_df['x'].min()
            min_y = self.nodes_df['y'].min()
            # This is complex. We'll use the coordinate function.
            # Vector for M units in j
            v1_coords = self._get_coordinates(0, self.M, 0) 
            # Vector for N units in i
            v2_coords = self._get_coordinates(self.N, 0, 0)
            v1 = [v1_coords[0], v1_coords[1], 0]
            v2 = [v2_coords[0], v2_coords[1], 0]
        
        cell = [v1, v2, v3]
        
        self.atoms_obj = Atoms(symbols=symbols, 
                               positions=positions_3d,
                               cell=cell,
                               pbc=[self.pbc, self.pbc, False])
        
        self.atoms_obj.center(about=None)
        self.nodes_df[['x', 'y']] = self.atoms_obj.get_positions()[:, :2]
        self.cell_vectors = self.atoms_obj.get_cell()[:2, :2]


    def get_all_bonds(self) -> pd.DataFrame:
        """
        Lists all unique bonds in the lattice with their types.
        """
        # ... (same as before) ...
        if not self.bonds:
            print("No bonds built. Run build_lattice() first.")
            return pd.DataFrame(columns=['atom_index_1', 'atom_index_2', 'bond_type'])
        return pd.DataFrame(self.bonds, columns=['atom_index_1', 'atom_index_2', 'bond_type'])

    def calculate_total_energy(self, bond_energies: Dict[str, float]) -> float:
        """
        Calculates the total energy of the lattice by summing
        nearest-neighbor bond energies.
        """
        # ... (same as before) ...
        if not self.bonds:
            raise RuntimeError("Lattice has no bonds. Run build_lattice() first.")
        total_energy = 0.0
        for _, _, bond_type in self.bonds:
            if bond_type not in bond_energies:
                raise ValueError(
                    f"Energy for bond type '{bond_type}' not found in bond_energies."
                )
            total_energy += bond_energies[bond_type]
        return total_energy

    def plot_lattice(self, 
                     show_atom_indices: bool = False, 
                     show_bonds: bool = True,
                     show_periodic_images: bool = False):
        """
        Plots the 2D lattice using Matplotlib and Seaborn.
        """
        if self.nodes_df is None:
            raise RuntimeError("Lattice not built. Run build_lattice() first.")
            
        sns.set_theme(style="whitegrid", palette="muted")
        fig, ax = plt.subplots(figsize=(max(self.M, 8), max(self.N, 6)))
        
        # Bond length heuristic
        max_bond_dist_sq = (1.5 * max(self.a, self.b) * 2)**2 # Larger buffer
        
        # --- PERIODIC Image Plotting Logic ---
        if self.pbc and show_periodic_images:
            if self.cell_vectors is None:
                print("Warning: Cell vectors not defined. Cannot show periodic images.")
                return
            
            v1, v2 = self.cell_vectors[0], self.cell_vectors[1]
            
            for i_shift in [-1, 0, 1]:
                for j_shift in [-1, 0, 1]:
                    displacement = (i_shift * v2) + (j_shift * v1)
                    is_central_cell = (i_shift == 0 and j_shift == 0)
                    
                    df_ghost = self.nodes_df.copy()
                    df_ghost['x'] = self.nodes_df['x'] + displacement[0]
                    df_ghost['y'] = self.nodes_df['y'] + displacement[1]
                    
                    self._plot_single_cell_data(ax, df_ghost,
                                                show_atom_indices=show_atom_indices if is_central_cell else False,
                                                show_bonds=show_bonds,
                                                max_bond_dist_sq=max_bond_dist_sq,
                                                is_ghost=not is_central_cell)
        else:
            # --- SINGLE Cell Plotting Logic ---
            self._plot_single_cell_data(ax, self.nodes_df,
                                        show_atom_indices=show_atom_indices,
                                        show_bonds=show_bonds,
                                        max_bond_dist_sq=max_bond_dist_sq,
                                        is_ghost=False)
                         
        plt.title(f"{self.N}x{self.M} Lattice (Topology: {self.topology}, PBC: {self.pbc})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.axis('equal') 
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def _plot_single_cell_data(self, ax, 
                               df: pd.DataFrame, 
                               show_atom_indices: bool, 
                               show_bonds: bool, 
                               max_bond_dist_sq: float,
                               is_ghost: bool):
        """Helper method to plot atoms and bonds for a given DataFrame."""
        # ... (same as before) ...
        alpha = 0.3 if is_ghost else 1.0
        if show_bonds:
            for idx1, idx2, _ in self.bonds:
                if idx1 not in df.index or idx2 not in df.index:
                    continue
                p1, p2 = df.loc[idx1], df.loc[idx2]
                dist_sq = (p1.x - p2.x)**2 + (p1.y - p2.y)**2
                if not is_ghost and self.pbc and dist_sq > max_bond_dist_sq:
                    continue 
                ax.plot([p1.x, p2.x], [p1.y, p2.y], color='gray', zorder=1, 
                        alpha=0.2 if is_ghost else 0.7)
        sns.scatterplot(
            ax=ax, x='x', y='y', hue='atom_type', data=df,
            s=50 if is_ghost else 150, zorder=2, alpha=alpha,
            legend=False if is_ghost else 'full')
        if show_atom_indices:
            for idx, row in df.iterrows():
                ax.text(row.x, row.y, str(idx),
                         ha='center', va='center',
                         fontweight='bold', color='white', fontsize=7)


if __name__ == "__main__":
    
    # --- Example 1: Regular Honeycomb Lattice (4x3 cells) ---
    print("--- Example 1: 5x5 Square lattice (a=1) ---")
    square_lattice = Lattice2D(N=3, M=3, topology='square', a=1.0)
    square_lattice.build_lattice()
    print(f"Total atoms: {square_lattice.total_atoms}")
    square_lattice.plot_lattice(show_atom_indices=True,
                                show_periodic_images=True)

    # # --- Example 2: Distorted Honeycomb (A-B Alloy) ---
    # print("\n--- Example 2: 3x3 Distorted Honeycomb (a=1, b=1.5) ---")
    # N_h, M_h = 3, 3
    # # N*M array, will be applied to A and B sublattices
    # random_alloy = np.random.choice(['Au', 'Pt'], size=(N_h, M_h))
    
    # dist_honey = Lattice2D(N=N_h, M=M_h, 
    #                        topology='distorted_honeycomb', 
    #                        a=1.0, b=1.5) # a=h-bonds, b=v-bonds
    # dist_honey.build_lattice(atom_types=random_alloy)
    # print(f"Total atoms: {dist_honey.total_atoms}")
    # dist_honey.plot_lattice(show_atom_indices=False, show_bonds=True)

    # # --- Example 3: Energy Calculation for Distorted Honeycomb ---
    # print("\n--- Example 3: Energy Calculation for Distorted Honeycomb ---")
    
    # # 'a' (h-bonds) and 'b' (v-bonds)
    # energies = {
    #     'Au-Au_h': -1.0, 'Au-Au_v': -1.2,
    #     'Pt-Pt_h': -1.0, 'Pt-Pt_v': -1.2,
    #     'Au-Pt_h': -2.0, 'Au-Pt_v': -2.5, # 'b' bonds (vertical) are stronger
    # }
    
    # try:
    #     total_energy = dist_honey.calculate_total_energy(bond_energies=energies)
    #     print(f"Total Lattice Energy (Distorted): {total_energy:.2f}")
        
    #     print("\nBond type counts:")
    #     print(dist_honey.get_all_bonds()['bond_type'].value_counts())
        
    # except ValueError as e:
    #     print(f"Energy calculation failed: {e}")

    # # --- Example 4: Square Lattice with PBC (Default) ---
    # print("\n--- Example 4: 4x4 Square with PBC (Default) ---")
    # square_pbc = Lattice2D(N=4, M=4, topology='square')
    # square_pbc.build_lattice()
    # print("Plotting 3x3 periodic images to show wrapping:")
    # square_pbc.plot_lattice(show_periodic_images=True, show_atom_indices=True)
