import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import write


class BondsModel:
    """
    Classical bond-interaction Hamiltonian on a 2D lattice with PBC.

    - Square lattice with z = 4 neighbors per site (up, down, left, right).
    - Binary or multicomponent alloys with specified concentrations.
    - Nearest-neighbor pair interactions J_{alpha,beta}.
    """

    def __init__(
        self,
        nx,
        ny,
        a=2.0,
        topology="square",
        species=("Au", "Pt"),
        concentrations=(0.5, 0.5),
        J=None,
        pbc=(True, True, False),
        seed=None,
    ):
        self.nx = int(nx)
        self.ny = int(ny)
        self.N = self.nx * self.ny
        self.a = float(a)
        self.topology = topology.lower()
        self.species = list(species)
        self.concentrations = np.array(concentrations, dtype=float)
        self.pbc = tuple(bool(x) for x in pbc)
        self.rng = np.random.default_rng(seed)

        if len(self.species) != len(self.concentrations):
            raise ValueError("species and concentrations must have the same length")
        if not np.isclose(self.concentrations.sum(), 1.0):
            raise ValueError("concentrations must sum to 1")

        # Build lattice positions
        self.positions = self._build_lattice()

        # Random alloy configuration with given concentrations
        self.occupation = self._generate_occupation()

        # ASE Atoms object
        self.atoms = self._build_atoms()

        # Nearest-neighbor list (4 neighbors per site)
        self.neighbors = self._build_neighbor_list()

        # Interaction matrix J_{alpha,beta} as pandas DataFrame
        self.J = self._prepare_J(J)

    # ------------------------------------------------------------------
    # Lattice construction
    # ------------------------------------------------------------------
    def _build_lattice(self):
        if self.topology == "square":
            return self._build_square_lattice()
        else:
            raise NotImplementedError(
                f"Topology '{self.topology}' not implemented yet "
                "(add _build_<topology>_lattice)."
            )

    def _build_square_lattice(self):
        xs = np.arange(self.nx) * self.a
        ys = np.arange(self.ny) * self.a
        positions = []
        for j in range(self.ny):
            for i in range(self.nx):
                positions.append([xs[i], ys[j], 0.0])
        return np.array(positions, dtype=float)

    # ------------------------------------------------------------------
    # Alloy configuration
    # ------------------------------------------------------------------
    def _generate_occupation(self):
        counts = np.round(self.concentrations * self.N).astype(int)
        diff = self.N - counts.sum()
        if diff != 0:
            idx = np.argmax(self.concentrations)
            counts[idx] += diff

        labels = []
        for sp, n in zip(self.species, counts):
            labels.extend([sp] * n)
        labels = np.array(labels, dtype=object)
        self.rng.shuffle(labels)
        return labels

    # ------------------------------------------------------------------
    # ASE Atoms
    # ------------------------------------------------------------------
    def _build_atoms(self):
        cell = np.diag([self.nx * self.a, self.ny * self.a, self.a])
        atoms = Atoms(
            symbols=list(self.occupation),
            positions=self.positions,
            cell=cell,
            pbc=self.pbc,
        )
        return atoms

    # ------------------------------------------------------------------
    # Neighbor list with PBC, z = 4
    # ------------------------------------------------------------------
    def _site_index(self, i, j):
        return j * self.nx + i

    def _build_neighbor_list(self):
        """
        Build list of 4 nearest neighbors per site for a square lattice with PBC.

        For each site i:
            neighbors[i] = [right, left, up, down]
        where indices are wrapped by modulo (torus topology).
        """
        if self.topology != "square":
            raise NotImplementedError("Neighbor list only implemented for square lattice")

        neighbors = np.empty((self.N, 4), dtype=int)

        for j in range(self.ny):
            for i in range(self.nx):
                idx = self._site_index(i, j)

                ip = (i + 1) % self.nx  # right
                im = (i - 1) % self.nx  # left
                jp = (j + 1) % self.ny  # up
                jm = (j - 1) % self.ny  # down

                neighbors[idx, 0] = self._site_index(ip, j)   # right
                neighbors[idx, 1] = self._site_index(im, j)   # left
                neighbors[idx, 2] = self._site_index(i, jp)   # up
                neighbors[idx, 3] = self._site_index(i, jm)   # down

        return neighbors

    # ------------------------------------------------------------------
    # Interaction matrix J_{alpha,beta}
    # ------------------------------------------------------------------
    def _prepare_J(self, J):
        sp = self.species

        if J is None:
            data = np.ones((len(sp), len(sp)), dtype=float)
            return pd.DataFrame(data, index=sp, columns=sp)

        if isinstance(J, (int, float)):
            data = np.full((len(sp), len(sp)), float(J))
            return pd.DataFrame(data, index=sp, columns=sp)

        if isinstance(J, dict):
            data = np.zeros((len(sp), len(sp)), dtype=float)
            df = pd.DataFrame(data, index=sp, columns=sp)
            for (a, b), val in J.items():
                if a not in sp or b not in sp:
                    raise ValueError(f"Unknown species in J key: {(a, b)}")
                df.loc[a, b] = val
                df.loc[b, a] = val  # symmetric
            return df

        if isinstance(J, pd.DataFrame):
            if not set(sp).issubset(J.index) or not set(sp).issubset(J.columns):
                raise ValueError("J DataFrame must contain all species as index and columns")
            return J.reindex(index=sp, columns=sp).copy()

        raise TypeError("Unsupported type for J")

    # ------------------------------------------------------------------
    # Hamiltonian / energy
    # ------------------------------------------------------------------
    def bond_energy(self):
        """
        Compute the total bond energy:
            H = - sum_{<ij>} J_{sigma_i, sigma_j}

        Implementation details:
        - neighbors[i] has 4 neighbors (right, left, up, down).
        - For i != j, each bond (i,j) appears twice in the full sum
          (once from i, once from j). We count it only once by using i < j.
        - For i == j (self-bond, e.g., 1x1 lattice or collapsed directions),
          we sum every occurrence. For a 1x1 lattice, this gives 4 bonds
          (coordination 4 at one site).
        """
        E = 0.0
        for i in range(self.N):
            s_i = self.occupation[i]
            for j in self.neighbors[i]:
                s_j = self.occupation[j]
                J_ij = self.J.loc[s_i, s_j]

                if i == j:
                    # self-bond (e.g., 1x1 torus → 4 self-bonds)
                    E -= J_ij
                elif i < j:
                    # count each (i,j) with i != j only once
                    E -= J_ij
                # if i > j: skip to avoid double counting

        return E

    def site_energies(self):
        """
        Per-site energy: each bond contributes half to each endpoint.
        Self-bonds (i == j) are fully assigned to that site.
        """
        e_site = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            s_i = self.occupation[i]
            for j in self.neighbors[i]:
                s_j = self.occupation[j]
                J_ij = self.J.loc[s_i, s_j]

                if i == j:
                    e_site[i] -= J_ij  # self-bond
                elif i < j:
                    e_site[i] -= 0.5 * J_ij
                    e_site[j] -= 0.5 * J_ij
        return e_site

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_lattice(self, n_images=(1, 1), show_bonds=True, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))

        nx_rep, ny_rep = n_images
        nx_half = nx_rep // 2
        ny_half = ny_rep // 2

        species_unique = self.species
        cmap = plt.get_cmap("tab10")
        color_map = {sp: cmap(i % 10) for i, sp in enumerate(species_unique)}

        # Periodic images (for visualization only)
        for dx in range(-nx_half, nx_half + 1):
            for dy in range(-ny_half, ny_half + 1):
                shift = np.array([dx * self.nx * self.a, dy * self.ny * self.a, 0.0])
                pos_shifted = self.positions + shift
                for sp in species_unique:
                    mask = self.occupation == sp
                    ax.scatter(
                        pos_shifted[mask, 0],
                        pos_shifted[mask, 1],
                        marker="o",
                        s=50,
                        label=sp if (dx == 0 and dy == 0) else None,
                        color=color_map[sp],
                        edgecolors="k",
                        linewidths=0.5,
                    )

        # Bonds in the central cell
        if show_bonds:
            for i in range(self.N):
                x1, y1, _ = self.positions[i]
                for j in self.neighbors[i]:
                    # to avoid drawing each bond twice, only draw when i<j or i==j
                    if (i < j) or (i == j):
                        x2, y2, _ = self.positions[j]
                        ax.plot([x1, x2], [y1, y2], linestyle="-", linewidth=1, alpha=0.5)

        ax.set_aspect("equal")
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        ax.set_title(f"{self.topology.capitalize()} lattice ({self.nx}x{self.ny}) with PBC")
        ax.legend()
        plt.tight_layout()
        return ax

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(self, filename, format=None):
        write(filename, self.atoms, format=format)


# Example usage
if __name__ == "__main__":

    J_binary = {("Au", "Au"): 1.0, ("Pt", "Pt"): 0.5, ("Au", "Pt"): 0.8}
    model_3x3 = BondsModel(5,5, species=["Au", "Pt"], concentrations=[0.3, 0.7], J=J_binary, seed=42)

    for i in range(model_3x3.N): 
        print(f"Site {i} has {len(model_3x3.neighbors[i])} neighbors")
    E = model_3x3.bond_energy()
    print(f"Total bond energy: {E} eV")

    model_3x3.plot_lattice(n_images=(1,1), show_bonds=True)
    plt.show()
    # model_3x3.export("lattice.xyz")