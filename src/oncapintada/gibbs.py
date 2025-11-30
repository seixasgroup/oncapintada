# -*- coding: utf-8 -*-
# file: gibbs.py

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
from ase import Atoms

class GibbsFreeEnergy:
    """
    Class to work with tabulated Gibbs free energy G(T, x) stored in a pandas DataFrame.

    - Rows (index) are temperatures T.
    - Columns are compositions x.
    - Values are G(T, x).

    Main methods:
    - dGdx()                -> returns a DataFrame with the first derivative (∂G/∂x).
    - d2Gdx2()              -> returns a DataFrame with the second derivative (∂²G/∂x²).
    - spinodal_decomposition() -> returns spinodal curve(s) x(T) from sign changes in ∂²G/∂x².
    """

    def __init__(self, gibbs_df: pd.DataFrame, x_values=None, atoms: Atoms | None = None):
        """
        Parameters
        ----------
        gibbs_df : pd.DataFrame
            DataFrame with G(T, x):
            - index: temperatures T (float or int).
            - columns: compositions x (float or convertible to float).
            - values: Gibbs free energy G(T, x).
        x_values : array-like, optional
            Values of x corresponding to the columns.
            If None, attempts to convert the column labels to float.
        atoms : ase.Atoms, optional
            Atoms object associated with the thermodynamic data (optional).
        """
        # Ensure numeric values
        self.gibbs = gibbs_df.astype(float).copy()

        # Define composition axis x
        if x_values is None:
            self.x = np.array(self.gibbs.columns, dtype=float)
            self.gibbs.columns = self.x
        else:
            self.x = np.array(x_values, dtype=float)
            if len(self.x) != self.gibbs.shape[1]:
                raise ValueError(
                    "x_values must have the same length as the number of columns in gibbs_df."
                )
            self.gibbs.columns = self.x

        # Define temperature axis T
        self.T = np.array(self.gibbs.index, dtype=float)
        self.gibbs.index = self.T

        # Optional Atoms object
        self.atoms = atoms

    # ------------------------------------------------------------------
    # First derivative ∂G/∂x
    # ------------------------------------------------------------------
    def dGdx(self) -> pd.DataFrame:
        """
        Compute the first derivative ∂G/∂x using np.gradient along the x axis.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same index (T) and columns (x),
            containing (∂G/∂x)(T, x).
        """
        G = self.gibbs.values  # shape: (nT, nX)
        dGdx_array = np.gradient(G, self.x, axis=1, edge_order=2)

        dGdx_df = pd.DataFrame(dGdx_array, index=self.gibbs.index, columns=self.gibbs.columns)
        return dGdx_df

    # ------------------------------------------------------------------
    # Second derivative ∂²G/∂x²
    # ------------------------------------------------------------------
    def d2Gdx2(self) -> pd.DataFrame:
        """
        Compute the second derivative ∂²G/∂x² from G(T, x)
        using two calls to np.gradient along x.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same index (T) and columns (x),
            containing (∂²G/∂x²)(T, x).
        """
        G = self.gibbs.values  # (nT, nX)

        # First derivative
        dGdx = np.gradient(G, self.x, axis=1, edge_order=2)
        # Second derivative
        d2Gdx2_array = np.gradient(dGdx, self.x, axis=1, edge_order=2)

        d2Gdx2_df = pd.DataFrame(d2Gdx2_array, index=self.gibbs.index, columns=self.gibbs.columns)
        return d2Gdx2_df

    # ------------------------------------------------------------------
    # Spinodal decomposition (sign change in second derivative)
    # ------------------------------------------------------------------
    def spinodal_decomposition(self, atol: float = 1e-8) -> pd.DataFrame:
        """
        Find, for each T, the composition(s) x where the second derivative ∂²G/∂x²
        changes sign (spinodal boundaries).

        Algorithm
        ---------
        - Compute ∂²G/∂x² on the (T, x) grid.
        - For each row (fixed T):
            * Scan adjacent points in x: (x_j, x_{j+1}).
            * Look for sign changes in d2Gdx2(T, x_j) and d2Gdx2(T, x_{j+1}),
              i.e., y1 * y2 < 0.
            * Approximate the zero by linear interpolation between x_j and x_{j+1}.
            * Also treat values with |y| < atol as "numerical zeros"
              and include their corresponding x.
        - For each T, keep up to two distinct x values (x1, x2), sorted.

        Parameters
        ----------
        atol : float, optional
            Absolute tolerance to treat small values of ∂²G/∂x² as zero.

        Returns
        -------
        pd.DataFrame
            Index: temperatures T.
            Columns: 'x1', 'x2', containing the first and second spinodal
            compositions (if they exist) for each T. Entries are np.nan where
            no spinodal crossing is found.
        """
        d2_df = self.d2Gdx2()
        d2 = d2_df.values  # shape: (nT, nX)
        x = self.x
        nT, nX = d2.shape

        spinodal_points = []

        for iT in range(nT):
            row = d2[iT, :]
            xs_found = []

            for j in range(nX - 1):
                y1 = row[j]
                y2 = row[j + 1]

                if np.isnan(y1) or np.isnan(y2):
                    continue

                # Treat near-zero values as zeros
                if abs(y1) < atol:
                    xs_found.append(x[j])
                if abs(y2) < atol:
                    xs_found.append(x[j + 1])

                # Check sign change (excluding exact/near zeros already handled)
                if y1 * y2 < 0.0:
                    # Linear interpolation for zero crossing between x[j] and x[j+1]
                    # y(x) = y1 + (y2 - y1)*t, y(x_zero) = 0 => t = -y1 / (y2 - y1)
                    t = -y1 / (y2 - y1)
                    x_zero = x[j] + t * (x[j + 1] - x[j])
                    xs_found.append(x_zero)

            # Remove approximate duplicates and sort
            xs_unique = []
            for xv in xs_found:
                if not any(np.isclose(xv, y, atol=1e-10) for y in xs_unique):
                    xs_unique.append(xv)
            xs_unique = sorted(xs_unique)

            if len(xs_unique) >= 1:
                x1 = xs_unique[0]
            else:
                x1 = np.nan

            if len(xs_unique) >= 2:
                x2 = xs_unique[1]
            else:
                x2 = np.nan

            spinodal_points.append((x1, x2))

        spinodal_df = pd.DataFrame(spinodal_points, index=self.gibbs.index, columns=["x1", "x2"])
        return spinodal_df
