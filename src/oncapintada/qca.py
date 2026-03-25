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
    def __init__(self, x_values, t_values):
        self.x_values = x_values
        self.t_values = t_values

    def warren_cowley_parameter(self, a, b, x) -> pd.DataFrame:
        pass

    def get_enthalpy_of_mixing(self, a, b, x) -> pd.DataFrame:
        pass

    def get_entropy_of_mixing(self, a, b, x) -> pd.DataFrame:
        pass

    def get_gibbs_free_energy_of_mixing(self, a, b, x) -> pd.DataFrame:
        pass


