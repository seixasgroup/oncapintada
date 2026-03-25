# -*- coding: utf-8 -*-
# file: qca.py

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

# Quasichemical Approximation (QCA) for thermodynamic modeling of alloys.
class QCA:
    def __init__(self, x_values:  None, t_values = None, coordination_number = 0, enthalpy_df = None):
        self.x_values = x_values
        self.t_values = t_values
        self.coordination_number = coordination_number
        self.enthalpy_df = enthalpy_df
        self.omega = self.calculate_omega()

        x_h, t_h = self.enthalpy_df.shape

        if self.coordination_number <= 0:
            raise ValueError("Coordination number must be a positive integer.")
        
        if not isinstance(self.enthalpy_df, pd.DataFrame):
            raise ValueError("Enthalpy data must be a pandas DataFrame.")

        if x_h != len(self.x_values):
            raise ValueError("Mismatch in x values and enthalpy data.")
        
        if t_h != len(self.t_values):
            raise ValueError("Mismatch in t values and enthalpy data.")
        
        if self.enthalpy_df.isnull().values.any():
            raise ValueError("Enthalpy data contains NaN values.")
        
        
    def calculate_omega(self) -> pd.DataFrame:
        '''
        Calculate the interaction parameter omega(x,t) from the enthalpy of mixing data.
        '''
        x = self.x_values
        t = self.t_values
        h = self.enthalpy_df.values

        # shift x=0 and x=1 to avoid division by zero in omega calculation
        x = np.clip(x, 1e-6, 1 - 1e-6)

        # enthalpy_df is a matrix of shape (len(x), len(t)) containing the enthalpy of mixing values for each composition and temperature.
        omega = h / (x * (1 - x))[:, np.newaxis]
        omega_df = pd.DataFrame(omega, index=self.x_values, columns=self.t_values)
        return omega_df


    def gamma(self) -> pd.DataFrame:
        z = self.coordination_number
        R = 8.314 / 1000    # kJ/(mol*K)
        T = self.t_values
        gamma = np.exp(-2*self.omega.values/(z*R*T))
        gamma_df = pd.DataFrame(gamma, index=self.x_values, columns=self.t_values)
        return gamma_df


    def probability_of_pair(self) -> pd.DataFrame:
        g = self.gamma()
        x = self.x_values
        t = self.t_values
        # p = np.zeros((2,2))

        p_ab = ( -g +np.sqrt(g**2 + 4*g*(1-g)*x*(1-x)) ) / ( 1-g )
        p_ab_df = pd.DataFrame(p_ab, index=x, columns=t)
        return p_ab_df
    

    def warren_cowley_parameter(self, a, b, x) -> pd.DataFrame:
        pass

    def get_enthalpy_of_mixing(self, a, b, x) -> pd.DataFrame:
        pass

    def get_entropy_of_mixing(self, a, b, x) -> pd.DataFrame:
        pass

    def get_gibbs_free_energy_of_mixing(self, a, b, x) -> pd.DataFrame:
        pass


