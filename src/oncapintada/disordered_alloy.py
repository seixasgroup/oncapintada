# -*- coding: utf-8 -*-
# file: disordered_alloy.py

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
from typing import List, Sequence, Union, Optional

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write

MaskType = Union[str, Sequence[int], Sequence[bool]]

dataclass
class DisorderedAlloyConfig:
    """Container for a generated disordered alloy configuration."""
    atoms: Atoms
    site_table: pd.DataFrame  # mapping between original and substituted sites


class DisorderedAlloyGenerator:
    """
    Generator of disordered alloy geometries on a chosen sublattice.

    Parameters
    ----------
    template_file : str
        Path to the template structure file (e.g. XYZ file).
    substitution_mask : str or sequence of int or sequence of bool
        Mask selecting which sites will be substituted:
        - str: chemical symbol, e.g. "Ni" → all Ni atoms are candidates.
        - sequence of int: list/array of atom indices to be substituted.
        - sequence of bool: boolean mask of length len(atoms).
    new_elements : sequence of str
        List of chemical symbols for the new species (e.g. ["Co", "Cr"]).
    concentrations : sequence of float
        Target concentrations for each new element, same order as new_elements.
        They do not need to sum exactly to 1.0 (they will be normalized).
    seed : int, optional
        Random seed for reproducible shuffling.

    Example
    -------
    NiO template (rocksalt), substitute only Ni by Co and Cr:

    >>> gen = DisorderedAlloyGenerator(
    ...     template_file="NiO.xyz",
    ...     substitution_mask="Ni",
    ...     new_elements=["Co", "Cr"],
    ...     concentrations=[0.4, 0.6],
    ...     seed=42,
    ... )
    >>> config = gen.generate_configuration()
    >>> alloy_atoms = config.atoms
    >>> write("Co0.4Cr0.6O.xyz", alloy_atoms)
    """

    def __init__(
        self,
        template_file: str,
        substitution_mask: MaskType,
        new_elements: Sequence[str],
        concentrations: Sequence[float],
        seed: Optional[int] = None,
    ) -> None:
        # Load template structure
        self.template_file = template_file
        self.template_atoms: Atoms = read(template_file)
        self.n_atoms: int = len(self.template_atoms)

        # Store new elements and concentrations
        self.new_elements: List[str] = list(new_elements)
        self.concentrations = np.array(concentrations, dtype=float)
        if self.concentrations.ndim != 1:
            raise ValueError("concentrations must be a 1D sequence")
        if len(self.new_elements) != len(self.concentrations):
            raise ValueError("new_elements and concentrations must have same length")

        # Normalize concentrations to sum to 1
        total = self.concentrations.sum()
        if total <= 0:
            raise ValueError("concentrations must sum to a positive value")
        self.concentrations /= total

        # Random generator
        self._rng = np.random.default_rng(seed)

        # Determine which sites will be substituted
        self.substitution_indices = self._parse_mask(substitution_mask)
        if len(self.substitution_indices) == 0:
            raise ValueError("Substitution mask selected zero atoms.")

        # Precompute a table with information about substitutable sites
        self._base_site_table = self._build_base_site_table()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _parse_mask(self, mask: MaskType) -> np.ndarray:
        """Convert the user mask into an array of atom indices."""
        symbols = np.array(self.template_atoms.get_chemical_symbols())

        # Case 1: mask is a chemical symbol (e.g. "Ni")
        if isinstance(mask, str):
            idx = np.where(symbols == mask)[0]
            return idx.astype(int)

        mask = np.asarray(mask)

        # Case 2: mask is a list/array of bool
        if mask.dtype == bool:
            if mask.shape[0] != self.n_atoms:
                raise ValueError(
                    "Boolean mask length must match number of atoms in template."
                )
            return np.where(mask)[0].astype(int)

        # Case 3: mask is a list/array of indices
        if np.issubdtype(mask.dtype, np.integer):
            # Basic sanity check
            if (mask < 0).any() or (mask >= self.n_atoms).any():
                raise ValueError("Substitution indices out of range.")
            return mask.astype(int)

        raise TypeError(
            "substitution_mask must be a chemical symbol (str), "
            "a sequence of int, or a sequence of bool."
        )

    def _build_base_site_table(self) -> pd.DataFrame:
        """Create a DataFrame describing the sites that can be substituted."""
        pos = self.template_atoms.get_positions()
        symbols = np.array(self.template_atoms.get_chemical_symbols())

        df = pd.DataFrame(
            {
                "atom_index": self.substitution_indices,
                "x": pos[self.substitution_indices, 0],
                "y": pos[self.substitution_indices, 1],
                "z": pos[self.substitution_indices, 2],
                "original_symbol": symbols[self.substitution_indices],
            }
        )
        return df

    def _compute_exact_counts(self, n_sites: int) -> np.ndarray:
        """
        Compute integer counts for each element such that:
        - counts.sum() == n_sites
        - counts are as close as possible to n_sites * concentrations
        """
        raw_counts = self.concentrations * n_sites
        base_counts = np.floor(raw_counts).astype(int)
        remainder = n_sites - base_counts.sum()

        if remainder > 0:
            # Assign remaining sites to elements with largest fractional parts
            fractional = raw_counts - base_counts
            order = np.argsort(fractional)[::-1]  # descending
            for i in range(remainder):
                base_counts[order[i % len(base_counts)]] += 1

        return base_counts


    def generate_configuration(self, seed: Optional[int] = None) -> DisorderedAlloyConfig:
        """
        Generate one disordered alloy configuration.

        Parameters
        ----------
        seed : int, optional
            If provided, overrides the internal RNG seed for this call.

        Returns
        -------
        DisorderedAlloyConfig
            Container with:
            - atoms: ASE Atoms object of the disordered alloy.
            - site_table: pandas DataFrame describing substitutions
                          (original and new symbols at each substituted site).
        """
        rng = np.random.default_rng(seed) if seed is not None else self._rng

        n_sites = len(self.substitution_indices)
        counts = self._compute_exact_counts(n_sites)

        # Build the list of new symbols with the desired counts
        new_symbol_list: List[str] = []
        for elem, count in zip(self.new_elements, counts):
            new_symbol_list.extend([elem] * count)
        new_symbol_list = np.array(new_symbol_list, dtype=object)

        # Shuffle them randomly among the substitutable sites
        rng.shuffle(new_symbol_list)

        # Create a copy of the template and apply substitutions
        atoms_new = self.template_atoms.copy()
        for site_index, new_symbol in zip(self.substitution_indices, new_symbol_list):
            atoms_new[site_index].symbol = str(new_symbol)

        # Build the site table for THIS specific configuration
        site_table = self._base_site_table.copy()
        site_table["new_symbol"] = new_symbol_list

        return DisorderedAlloyConfig(atoms=atoms_new, site_table=site_table)

    def generate_multiple(
        self,
        n_configs: int,
        base_seed: Optional[int] = None,
    ) -> List[DisorderedAlloyConfig]:
        """
        Generate multiple independent disordered alloy configurations.

        Parameters
        ----------
        n_configs : int
            Number of independent configurations to generate.
        base_seed : int, optional
            If provided, guarantees reproducible sequence of configurations.

        Returns
        -------
        list of DisorderedAlloyConfig
        """
        if base_seed is not None:
            rng = np.random.default_rng(base_seed)
            seeds = rng.integers(0, 2**32 - 1, size=n_configs)
        else:
            seeds = [None] * n_configs

        configs = []
        for s in seeds:
            configs.append(self.generate_configuration(seed=s))
        return configs
