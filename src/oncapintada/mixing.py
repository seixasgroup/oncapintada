# -*- coding: utf-8 -*-
# file: mixing.py

# This code is part of Onça-pintada.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.seixas@proton.me> 
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
from typing import Optional, List


# To calculate the enthalpy of mixing based on DSI model.
class BinarySubregularSolution:
    def __init__(self, energy_matrix: Optional[np.ndarray] = None, dilution: float = 0.0):
        self.energy_matrix = energy_matrix
        self.dilution = dilution
        if self.dilution < 1e-8:
            raise ValueError("Set the dilution parameter.")
        
    def get_energy_matrix(self) -> np.ndarray:
        return self.energy_matrix
    
    def set_energy_matrix(self, energy_matrix: Optional[np.ndarray] = None):
        self.energy_matrix = energy_matrix

    def get_Mij(self):
        pass

# To calculate the enthalpy of mixing based on DSI model for multi-component systems.
class MultiComponentSubregularSolution:
    def __init__(self, energy_matrix: Optional[np.ndarray] = None, dilution: float = 0.0):
        self.energy_matrix = energy_matrix
        self.dilution = dilution
        if self.dilution < 1e-8:
            raise ValueError("Set the dilution parameter.")



# To generate structures with a given target correlation function using Monte Carlo sampling.
class ReverseMonteCarlo:
    def __init__(self, atoms: Optional[Atoms] = None, composition=None, target=None):
        self.atoms = atoms
        self.composition = composition
        self.target = target

# To generate SQS structures for a given composition and lattice type.
class SQS:
    def __init__(self, atoms: Optional[Atoms] = None, composition=None, method='monte_carlo'):
        self.atoms = atoms
        self.composition = composition
        self.method = method