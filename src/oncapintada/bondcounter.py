# -*- coding: utf-8 -*-
# file: bondcounter.py

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

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, List, Union

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

# Defining Pair type
Pair = Tuple[str, str]

@dataclass
class BondCountResult:
    counts: Dict[Pair, int]
    total: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return the pair frequency and probabilities as a pandas DataFrame.
        The resulting DataFrame contains rows for each observed pair in lexical order
        with their counts and normalized probabilities, plus a final “TOTAL” row whose
        count equals the sum of all pair occurrences and whose probability is 1.0.
        Returns:
            pd.DataFrame: A DataFrame with columns "pair", "count", and "probability".
        """

        df = pd.DataFrame(
            [{"pair": f"{a}-{b}", "count": n, "probability": n / self.total if self.total > 0 else 0} for (a, b), n in sorted(self.counts.items())]
        )
        df.loc[len(df)] = {"pair": "TOTAL", "count": self.total, "probability": 1.0}
        return df


class BondCounter:
    """Count bonded atom pairs within an ASE Atoms object or XYZ file, optionally restricting
    the analysis to a subset of atoms defined by element symbols or explicit indices. The class
    initializes the atomic structure, enforces a cutoff distance for bond detection, configures
    the subset mask, and precomputes the target atom pairs that satisfy the bonding criteria."""

    def __init__(
        self,
        xyz_path: Optional[str] = None,          # Trajectory to an .xyz file
        atoms: Optional[Atoms] = None,           # Pre-loaded ASE Atoms object. If provided, xyz_path is ignored.
        cutoff: float = 3.5,                     # Cutoff radius in angstroms (default: 3.5 Å).
        subset_symbols: Optional[str] = None,    # Element symbols defining the subset of atoms to consider.
        subset_indices: Optional[Union[Iterable[int], np.ndarray]] = None,
    ):
        if atoms is None and xyz_path is None:
            raise ValueError("Provide `atoms` or `xyz_path`.")
        self.atoms: Atoms = atoms if atoms is not None else read(xyz_path)
        self.cutoff: float = float(cutoff)              # Cutoff distance for bond detection
        self._subset_mask: Optional[np.ndarray] = None  # Mask for subset of atoms

        if subset_symbols is not None:
            self.subset_from_symbols(subset_symbols)
        elif subset_indices is not None:
            self.set_subset(subset_indices)
        else:
            self.set_subset(None)
        
        self.target_pairs: set[Pair] = self.get_target_pairs()

    def set_subset(self, subset: Optional[Union[Iterable[int], np.ndarray]]) -> None:
        """Set or reset the subset mask for the current atoms selection.
        Parameters
        ----------
        subset : Optional[Union[Iterable[int], np.ndarray]]
            Indices of atoms to include in the subset. If ``None``, all atoms
            are included; otherwise, only the atoms at the provided indices
            are marked as part of the subset.
        """
        
        n = len(self.atoms)
        if subset is None:
            self._subset_mask = np.ones(n, dtype=bool)
        else:
            idx = np.array(list(subset), dtype=int)
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            self._subset_mask = mask

    def subset_from_symbols(self, symbols: Iterable[str]) -> None:
        """
        Update the internal subset mask to include only atoms whose chemical symbols
        appear in the provided iterable of symbols, effectively filtering the structure
        to those species of interest.
        """

        wanted = list(set(symbols))
        chem = np.array(self.atoms.get_chemical_symbols())
        mask = np.isin(chem, wanted)
        self._subset_mask = mask

    def get_target_pairs(self) -> set[Pair]:
        """
        Return the set of unique element pairs present in the selected subset of atoms.
        The method inspects the cached atomic symbols, optionally filtered by a subset mask,
        and generates all unique unordered combinations (with replacement) of the symbols.
        Each pair is stored as a sorted tuple to ensure deterministic ordering.
        Returns:
            set[Pair]: A set containing all unique element symbol pairs derived from the
                current atoms, respecting the subset mask if defined.
        """

        pairs = set()
        if self._subset_mask is None:
            chem_symbols = set(self.atoms.get_chemical_symbols())
        else:
            chem_symbols = set(np.array(self.atoms.get_chemical_symbols())[self._subset_mask])
        for a in chem_symbols:
            for b in chem_symbols:
                pair = tuple(sorted((a, b)))
                pairs.add(pair)
        return pairs

    def count_bonds(self) -> BondCountResult:
        """Count unique chemical bonds among the selected subset of atoms.

        Returns
        -------
        BondCountResult
            Result containing a dictionary with counts for each target pair and the total number of bonds counted.
        """

        chem = np.array(self.atoms.get_chemical_symbols())
        n = len(self.atoms)

        i_idx, j_idx = neighbor_list("ij", self.atoms, self.cutoff, self_interaction=False)

        mask_subset = self._subset_mask[j_idx] & self._subset_mask[i_idx]
        i_idx = i_idx[mask_subset]
        j_idx = j_idx[mask_subset]

        i_min = np.minimum(i_idx, j_idx)
        j_max = np.maximum(i_idx, j_idx)
        pairs = np.vstack((i_min, j_max)).T
        if len(pairs) == 0:
            counts = {p: 0 for p in self.target_pairs}
            return BondCountResult(counts=counts, total=0)

        pairs_unique = np.unique(pairs, axis=0)

        counts: Dict[Pair, int] = {p: 0 for p in self.target_pairs}
        total = 0

        for i, j in pairs_unique:
            si, sj = chem[i], chem[j]
            key = tuple(sorted((si, sj)))
            if key in counts:
                counts[key] += 1
                total += 1

        return BondCountResult(counts=counts, total=total)

    def bonds_dataframe(self) -> pd.DataFrame:
        """Return the count of bonds as a pandas DataFrame produced by `count_bonds`."""
        return self.count_bonds().to_dataframe()
