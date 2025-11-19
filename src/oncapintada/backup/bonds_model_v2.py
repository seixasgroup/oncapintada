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
    
    This class builds a 2D lattice of N x M nodes, establishes nearest-neighbor
    connections based on a specified topology, and stores atom information
    in a Pandas DataFrame. It supports periodic boundary conditions (PBC),
    custom lattice constants, and calculation of a simple bond-energy Hamiltonian.

    Supported topologies:
    - 'square': 4-connected grid.
    - 'rectangular': 4-connected grid, 'a' and 'b' define aspect ratio.
    - 'triangular': 6-connected grid (regular).
    - 'distorted_triangular': 6-connected grid, 'a' and 'b' define aspect ratio.
                              Bonds are typed as '_h' (horizontal) or '_d' (diagonal).
    - 'honeycomb': 3-connected grid (regular).
    - 'distorted_honeycomb': 3-connected grid, 'a' and 'b' define aspect ratio.
                             Bonds are typed as '_h' (horizontal) or '_v' (vertical).
    
    Attributes:
        N (int): Number of nodes in the first dimension (rows).
        M (int): Number of nodes in the second dimension (columns).
        topology (str): The lattice topology.
        pbc (bool): Whether to use Periodic Boundary Conditions.
        a (float): Primary lattice constant (typically x-direction).
        b (float): Secondary lattice constant (typically y-direction).
        nodes_df (pd.DataFrame): DataFrame holding node info.
        bonds (List[Tuple[int, int, str]]): List of unique bonds (idx1, idx2, bond_type).
        atoms_obj (ase.Atoms): An ASE Atoms object representing the lattice.
        cell_vectors (np.ndarray): The 2D cell vectors for PBC.
    """
    
    def __init__(self, 
                 N: int, 
                 M: int, 
                 topology: str = 'square', 
                 pbc: bool = False, 
                 a: float = 1.0, 
                 b: float = 1.0):
        """
        Initializes the 2D lattice.

        Args:
            N (int): Number of rows.
            M (int): Number of columns.
            topology (str): The lattice topology.
            pbc (bool): If True, use Periodic Boundary Conditions.
            a (float): Primary lattice constant (e.g., x-spacing).
            b (float): Secondary lattice constant (e.g., y-spacing).
        """
        self.N = N
        self.M = M
        
        valid_topologies = [
            'square', 'rectangular', 
            'triangular', 'distorted_triangular',
            'honeycomb', 'distorted_honeycomb'
        ]
        if topology not in valid_topologies:
            raise ValueError(f"Topology must be one of {valid_topologies}")
        self.topology = topology
        
        self.pbc = pbc
        self.a = a
        self.b = b
        
        # Adjust 'b' for regular topologies if 'a' is the only parameter
        if self.topology == 'square':
            self.b = a
        elif self.topology == 'triangular':
            self.b = a * np.sqrt(3) / 2.0
        elif self.topology == 'honeycomb':
            # For the (i+j)%2 neighbor logic, we use a simple grid
            # where b is the row separation.
            # For a regular honeycomb, b = a * sqrt(3) / 2
            # (assuming 'a' is the x-dist between j and j+1)
            # This is a bit abstract, 'a' and 'b' are just parameters
            # for the coordinate mapping.
            self.b = a * np.sqrt(3) / 2.0

        
        self.nodes_df: Optional[pd.DataFrame] = None
        self.bonds: List[Tuple[int, int, str]] = []
        self.atoms_obj: Optional[Atoms] = None
        self.cell_vectors: Optional[np.ndarray] = None

    def _get_index(self, i: int, j: int) -> int:
        """Converts (row, col) to a flat site index."""
        return i * self.M + j

    def _get_coordinates(self, i: int, j: int) -> Tuple[float, float]:
        """
        Gets (x, y) coordinates for a given (i, j) node index
        based on the lattice topology and constants 'a' and 'b'.
        """
        x, y = 0.0, 0.0
        
        if self.topology in ['square', 'rectangular']:
            x = j * self.a
            y = i * self.b
            
        elif self.topology in ['triangular', 'distorted_triangular']:
            # Skewed grid mapping
            x = j * self.a + 0.5 * i * self.a
            y = i * self.b
            
        elif self.topology in ['honeycomb', 'distorted_honeycomb']:
            # Use a simple rectangular grid for coordinates.
            # The bond-drawing will reveal the honeycomb topology.
            x = j * self.a
            y = i * self.b
            
        return (x, y)

    def _get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """
        Gets valid (i, j) neighbor indices for a given node
        based on the defined topology.
        """
        potential_neighbors = []
        
        if self.topology in ['square', 'rectangular']:
            # 4-connectivity
            potential_neighbors = [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]
            
        elif self.topology in ['triangular', 'distorted_triangular']:
            # 6-connectivity on the skewed grid
            potential_neighbors = [
                (i, j+1), (i, j-1),     # Horizontal
                (i-1, j), (i-1, j+1), # Diagonal-Up
                (i+1, j), (i+1, j-1)  # Diagonal-Down
            ]
            
        elif self.topology in ['honeycomb', 'distorted_honeycomb']:
            # 3-connectivity ("brick wall" logic on i, j grid)
            potential_neighbors = [(i, j-1), (i, j+1)] # Horizontal
            
            if (i + j) % 2 == 0:
                potential_neighbors.append((i-1, j)) # Vertical-Up
            else:
                potential_neighbors.append((i+1, j)) # Vertical-Down

        # --- Handle Boundary Conditions ---
        neighbors_ij = []
        if not self.pbc:
            # Finite (non-periodic)
            for ni, nj in potential_neighbors:
                if 0 <= ni < self.N and 0 <= nj < self.M:
                    neighbors_ij.append((ni, nj))
        else:
            # Periodic
            for ni, nj in potential_neighbors:
                ni_wrapped = ni % self.N
                nj_wrapped = nj % self.M
                
                if ni_wrapped == i and nj_wrapped == j:
                    continue
                    
                neighbors_ij.append((ni_wrapped, nj_wrapped))
                
        return list(set(neighbors_ij))

    def build_lattice(self, atom_types: Optional[np.ndarray] = None):
        """
        Builds the lattice nodes and their nearest-neighbor connections.
        
        Populates self.nodes_df, self.bonds, and self.atoms_obj.
        Assigns specific bond types ('_h', '_d', '_v') for distorted lattices.
        """
        if atom_types is None:
            atom_types = np.full((self.N, self.M), 'X', dtype=object)
        
        # 1. Create all nodes (sites)
        nodes_data = []
        for i in range(self.N):
            for j in range(self.M):
                x, y = self._get_coordinates(i, j)
                nodes_data.append({
                    'site_index': self._get_index(i, j),
                    'atom_type': atom_types[i, j],
                    'i': i, 'j': j, 'x': x, 'y': y,
                    'connections': []
                })
        self.nodes_df = pd.DataFrame(nodes_data).set_index('site_index')

        # 2. Build connections (bonds)
        self.bonds = [] 
        all_bonds_set = set() 
        
        for i in range(self.N):
            for j in range(self.M):
                current_idx = self._get_index(i, j)
                current_atom_type = self.nodes_df.loc[current_idx, 'atom_type']
                
                neighbors_ij = self._get_neighbors(i, j)
                node_connections_list = []
                
                for ni, nj in neighbors_ij:
                    neighbor_idx = self._get_index(ni, nj)
                    neighbor_atom_type = self.nodes_df.loc[neighbor_idx, 'atom_type']
                    
                    node_connections_list.append(neighbor_idx)
                    bond_tuple = tuple(sorted((current_idx, neighbor_idx)))
                    
                    if bond_tuple not in all_bonds_set:
                        base_bond_type = "-".join(sorted([current_atom_type, neighbor_atom_type]))
                        bond_type = base_bond_type # Default
                        
                        # --- Add direction-specific tags for distorted lattices ---
                        if self.topology == 'distorted_triangular':
                            if ni == i:
                                bond_type = f"{base_bond_type}_h" # Horizontal
                            else:
                                bond_type = f"{base_bond_type}_d" # Diagonal
                                
                        elif self.topology == 'distorted_honeycomb':
                            if ni == i:
                                bond_type = f"{base_bond_type}_h" # Horizontal
                            else:
                                bond_type = f"{base_bond_type}_v" # Vertical

                        self.bonds.append((bond_tuple[0], bond_tuple[1], bond_type))
                        all_bonds_set.add(bond_tuple)
                
                self.nodes_df.at[current_idx, 'connections'] = node_connections_list
                
        # 3. Create ASE Atoms object
        symbols = self.nodes_df['atom_type'].tolist()
        positions = self.nodes_df[['x', 'y']].values
        positions_3d = np.hstack([positions, np.zeros((len(positions), 1))])
        
        cell_z = 10.0 # Default 2D cell height
        v1, v2, v3 = [0,0,0], [0,0,0], [0,0,cell_z]
        
        # Define cell vectors based on coordinate system
        if self.topology in ['square', 'rectangular', 'honeycomb', 'distorted_honeycomb']:
            # Rectangular cell
            v1 = [self.M * self.a, 0, 0]
            v2 = [0, self.N * self.b, 0]
            
        elif self.topology in ['triangular', 'distorted_triangular']:
            # Skewed cell
            v_i_basis = np.array([0.5 * self.a, self.b])
            v_j_basis = np.array([self.a, 0])
            v1_cell = self.M * v_j_basis
            v2_cell = self.N * v_i_basis
            v1 = [v1_cell[0], v1_cell[1], 0]
            v2 = [v2_cell[0], v2_cell[1], 0]

        cell = [v1, v2, v3]
        
        self.atoms_obj = Atoms(symbols=symbols, 
                               positions=positions_3d,
                               cell=cell,
                               pbc=[self.pbc, self.pbc, False])
        
        self.atoms_obj.center(about=None)
        self.nodes_df[['x', 'y']] = self.atoms_obj.get_positions()[:, :2]
        
        # Store 2D cell vectors
        self.cell_vectors = self.atoms_obj.get_cell()[:2, :2]


    def get_all_bonds(self) -> pd.DataFrame:
        """
        Lists all unique bonds in the lattice with their types.

        Returns:
            pd.DataFrame: A DataFrame with columns 
                          ['atom_index_1', 'atom_index_2', 'bond_type'].
        """
        if not self.bonds:
            print("No bonds built. Run build_lattice() first.")
            return pd.DataFrame(columns=['atom_index_1', 'atom_index_2', 'bond_type'])
            
        return pd.DataFrame(self.bonds, columns=['atom_index_1', 'atom_index_2', 'bond_type'])

    def calculate_total_energy(self, bond_energies: Dict[str, float]) -> float:
        """
        Calculates the total energy of the lattice by summing
        nearest-neighbor bond energies.

        Args:
            bond_energies (Dict[str, float]): A dictionary mapping bond types
                (e.g., 'A-A', 'A-B_h') to their energy values.

        Returns:
            float: The total energy of the lattice.
        """
        if not self.bonds:
            raise RuntimeError("Lattice has no bonds. Run build_lattice() first.")
            
        total_energy = 0.0
        
        for _, _, bond_type in self.bonds:
            if bond_type not in bond_energies:
                raise ValueError(
                    f"Energy for bond type '{bond_type}' not found in bond_energies. "
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
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Heuristic for skipping "wrapped" bonds in single-cell PBC plots
        max_bond_dist_sq = (1.5 * max(self.a, self.b))**2
        
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
        
        alpha = 0.3 if is_ghost else 1.0
        
        # 1. Plot bonds
        if show_bonds:
            for idx1, idx2, _ in self.bonds:
                # Check if both atoms are in the plotting df
                if idx1 not in df.index or idx2 not in df.index:
                    continue
                    
                p1 = df.loc[idx1]
                p2 = df.loc[idx2]
                
                dist_sq = (p1.x - p2.x)**2 + (p1.y - p2.y)**2
                
                # If PBC is ON, skip drawing long "wrapped" lines
                # (This check is mainly for the single-cell PBC plot)
                if not is_ghost and self.pbc and dist_sq > max_bond_dist_sq:
                    continue 
                    
                ax.plot([p1.x, p2.x], [p1.y, p2.y], color='gray', zorder=1, 
                        alpha=0.2 if is_ghost else 0.7)

        # 2. Plot nodes (atoms)
        sns.scatterplot(
            ax=ax, x='x', y='y', hue='atom_type',
            data=df,
            s=50 if is_ghost else 150, 
            zorder=2,
            alpha=alpha,
            legend=False if is_ghost else 'full'
        )
        
        # 3. Annotate
        if show_atom_indices:
            for idx, row in df.iterrows():
                ax.text(row.x, row.y, str(idx),
                         ha='center', va='center',
                         fontweight='bold', color='white', fontsize=7)


if __name__ == "__main__":
    
    # # --- Example 1: Rectangular Lattice ---
    # print("--- Example 1: 8x6 Rectangular Lattice (a=2, b=1.5) ---")
    # rect_lattice = Lattice2D(N=8, M=6, topology='rectangular', a=2.0, b=1.5, pbc=True)
    # rect_lattice.build_lattice()
    # rect_lattice.plot_lattice(show_atom_indices=True)

    # # --- Example 2: Regular Honeycomb Lattice ---
    # print("\n--- Example 2: 6x5 Honeycomb Lattice (a=1) ---")
    # # 'a' defines the x-spacing, 'b' is set automatically
    # honey_lattice = Lattice2D(N=6, M=5, topology='honeycomb', a=1.0)
    # honey_lattice.build_lattice()
    # # The plot shows the 3-connected topology
    # honey_lattice.plot_lattice(show_atom_indices=True)

    # # --- Example 3: Triangular lattice (A-B Alloy) ---
    print("\n--- Example 3: 2x2 Triangular lattice (a=2.0) ---")
    N_tri, M_tri = 2, 2
    random_alloy = np.random.choice(['Au', 'Pt'], size=(N_tri, M_tri))
    
    dist_tri_lattice = Lattice2D(N=N_tri, M=M_tri, 
                                 topology='triangular', 
                                 a=2.0, pbc=True)
    dist_tri_lattice.build_lattice(atom_types=random_alloy)
    dist_tri_lattice.plot_lattice(show_atom_indices=False, show_bonds=True, show_periodic_images=True)

    # --- Example 4: Energy Calculation for Triangular Lattice ---
    print("\n--- Example 4: Energy Calculation for Triangular Lattice ---")
    
    # Define different energies for horizontal and diagonal bonds
    # energies = {
    #     'Au-Au_h': -1.0, 'Au-Au_d': -0.8,
    #     'Pt-Pt_h': -0.9, 'Pt-Pt_d': -0.5,
    #     'Au-Pt_h': -2.0, 'Au-Pt_d': -1.8, # A-B bonds are favorable
    # }

    energies = {
        'Au-Au': -1.0,
        'Pt-Pt': -0.9,
        'Au-Pt': -2.0,
    }
    
    try:
        total_energy = dist_tri_lattice.calculate_total_energy(bond_energies=energies)
        print(f"Total Lattice Energy (Distorted): {total_energy:.2f}")
        
        print("\nBond type counts:")
        print(dist_tri_lattice.get_all_bonds()['bond_type'].value_counts())
        
    except ValueError as e:
        print(f"Energy calculation failed: {e}")

    # --- Example 5: Distorted Honeycomb with PBC ---
    # print("\n--- Example 5: 4x4 Distorted Honeycomb (a=1, b=2) with PBC ---")
    # dist_honey_pbc = Lattice2D(N=4, M=4, 
    #                            topology='distorted_honeycomb', 
    #                            pbc=True, a=1.0, b=2.0)
    # dist_honey_pbc.build_lattice()
    
    # print("Plotting 3x3 periodic images to show wrapping:")
    # dist_honey_pbc.plot_lattice(show_periodic_images=True)