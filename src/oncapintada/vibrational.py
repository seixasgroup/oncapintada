# -*- coding: utf-8 -*-
# file: vibrational.py

# This code is part of Onça-pintada.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>
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
from .constants import R, h, cm_to_THz, eV_to_THz, kB, kJmol, cmrec
from ase import Atoms
from ase.phonons import Phonons
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from typing import Optional


class Vibrational:
    '''Class for vibrational properties of materials.'''
    def __init__(self, vdos: Optional[pd.DataFrame] = None, frequency_unit: str = 'cm^-1'):
        self.vdos = vdos
        if vdos is not None:
            self.nu = vdos['Frequency'].values            # frequency (in cm^-1, eV, or THz depending on frequency_unit)
            self.g_nu = vdos['VDOS'].values               # vibrational density of states (in states per frequency unit)
        else:
            self.nu = None
            self.g_nu = None
        self.frequency_unit = frequency_unit

        if frequency_unit not in ['THz', 'cm^-1', 'eV']:
            raise ValueError("frequency_unit must be either 'THz', 'cm^-1', or 'eV'")
        
        if frequency_unit == 'THz':
            self.nu_THz = self.nu                     # frequency in THz
            self.g_nu_THz = self.g_nu                 # VDOS in states/THz
        elif frequency_unit == 'cm^-1':
            self.nu_THz = self.nu * cm_to_THz         # convert frequency to THz
            self.g_nu_THz = self.g_nu / cm_to_THz     # convert VDOS to states/THz
        elif frequency_unit == 'eV':
            self.nu_THz = self.nu * eV_to_THz         # convert frequency to THz
            self.g_nu_THz = self.g_nu / eV_to_THz     # convert VDOS to states/THz

        self.number_of_modes = self.get_number_of_modes()
        self.vibrational_enthalpy = None
        self.vibrational_entropy = None
        self.vibrational_free_energy = None

    def get_vdos(self,
                 atoms: Atoms,
                 calc,
                 supercell: tuple = (2,2,2),
                 kpts: tuple = (100,100,1),
                 npts: int = 3000,
                 width: float = 5e-4,
                 xmin: float = 1e-4,
                 xmax: float = 0.10,
                 delta: float = 0.01,
                 fmax: float = 0.001) -> pd.DataFrame:
        '''Calculate the vibrational density of states (VDOS) using ASE's Phonons class.
        
        Parameters
        ----------
        atoms : ASE Atoms object
            The atomic structure for which to calculate the phonons.
        calc : ASE Calculator
            The calculator to use for force calculations.
        supercell : tuple, optional
            The size of the supercell for phonon calculations (default is (2,2,2)).
        kpts : tuple, optional
            The k-point grid for phonon calculations (default is (100,100,1)).
        npts : int, optional
            The number of points in the VDOS (default is 3000).
        width : float, optional
            The width of the Gaussian smearing for the VDOS (default is 5e-4).
        xmin : float, optional
            The minimum frequency for the VDOS (default is 1e-4).
        xmax : float, optional
            The maximum frequency for the VDOS (default is 0.10).
        delta : float, optional
            Displacement for finite difference calculations (default is 0.01).
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the frequencies and corresponding VDOS values.
        '''
        atoms.calc = calc
        optim = BFGS(UnitCellFilter(atoms), logfile=None)
        optim.run(fmax=fmax)
        ph = Phonons(atoms, calc, supercell=supercell, delta=delta)
        ph.run()
        ph.read(acoustic=True)
        ph.clean()
        ph_dos = ph.get_dos(kpts=kpts).sample_grid(npts=npts, width=width, xmin=xmin, xmax=xmax)
        
        self.nu = ph_dos.get_energies() * cmrec         # frequency in cm^-1
        self.g_nu = ph_dos.get_weights() / cmrec        # VDOS in states/cm^-1
        
        self.vdos = pd.DataFrame({'Frequency': self.nu, 'VDOS': self.g_nu})
        
        return self.vdos


    def get_number_of_modes(self) -> float:
        '''Calculate the total number of vibrational modes.
        
        Returns
        -------
        float
            Total number of vibrational modes.
        '''
        return np.trapz(self.g_nu, self.nu)
    

    def get_vibrational_enthalpy(self, T: np.ndarray) -> np.ndarray:
        '''Calculate the vibrational enthalpy at a given temperature.
        
        Parameters
        ----------
        T : np.ndarray
            Temperature in Kelvin.
        
        Returns
        -------
        np.ndarray
            Vibrational enthalpy in kJ/mol.

        '''
        nu = self.nu_THz
        g_nu = self.g_nu_THz
        x = h * nu / (kB * T)

        # exclude negative frequencies (imaginary modes) from the integration
        mask = nu > 0
        nu = nu[mask]
        g_nu = g_nu[mask]
        x = x[mask]

        enthalpy = np.trapz(g_nu * ( h * nu / 2 + h * nu / (np.exp(x) - 1)), nu)   # in eV
        enthalpy *= kJmol     # convert eV to kJ/mol
        self.vibrational_enthalpy = enthalpy
        return enthalpy
    

    def get_vibrational_entropy(self, T: np.ndarray) -> np.ndarray:
        '''Calculate the vibrational entropy at a given temperature.
        
        Parameters
        ----------
        T : np.ndarray
            Temperature in Kelvin.
        
        Returns
        -------
        np.ndarray
            Vibrational entropy in kJ/(mol*K).
        '''
        nu = self.nu_THz
        g_nu = self.g_nu_THz
        x = h * nu / (kB * T)

        # exclude negative frequencies (imaginary modes) from the integration
        mask = nu > 0
        nu = nu[mask]
        g_nu = g_nu[mask]
        x = x[mask]

        entropy = R * np.trapz(g_nu * ( x / (np.exp(x) - 1) - np.log(1 - np.exp(-x)) ), nu)     # in kJ/(mol*K)
        
        self.vibrational_entropy = entropy
        return entropy


    def get_vibrational_free_energy(self, T: np.ndarray) -> np.ndarray:
        '''Calculate the vibrational free energy at a given temperature.
        
        Parameters
        ----------
        T : np.ndarray
            Temperature in Kelvin.
        
        Returns
        -------
        np.ndarray
            Vibrational free energy in kJ/mol.
        '''
        if self.vibrational_enthalpy is None:
            enthalpy = self.get_vibrational_enthalpy(T)
        else:
            enthalpy = self.vibrational_enthalpy
        
        if self.vibrational_entropy is None:
            entropy = self.get_vibrational_entropy(T)
        else:
            entropy = self.vibrational_entropy
        
        free_energy = enthalpy - T * entropy
        
        self.vibrational_free_energy = free_energy
        return free_energy



class VibrationalSubregularModel:
    '''Class for vibrational properties of subregular solutions.'''
    def __init__(self,
                 x: np.ndarray,
                 vdos_AA: pd.DataFrame,
                 vdos_BB: pd.DataFrame,
                 vdos_AB: pd.DataFrame,
                 vdos_BA: pd.DataFrame,
                 frequency_unit: str = 'cm^-1'):
        self.x = x
        self.vibrational_AA = Vibrational(vdos_AA, frequency_unit)
        self.vibrational_BB = Vibrational(vdos_BB, frequency_unit)
        self.vibrational_AB = Vibrational(vdos_AB, frequency_unit)
        self.vibrational_BA = Vibrational(vdos_BA, frequency_unit)