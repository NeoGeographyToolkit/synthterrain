"""This module has tests for the synthterrain crater age functions."""

# Copyright © 2024, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The “synthterrain” software is licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License
# at http://www.apache.org/licenses/LICENSE-2.0.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import unittest

import numpy as np
import pandas as pd

import synthterrain.crater.age as age
from synthterrain.crater import functions as fns


class Test_Ages(unittest.TestCase):
    def test_equilibrium_age(self):
        diameters = np.array([1, 2, 3, 4, 5, 10, 20, 50, 100])
        a = 1
        b = 1000
        pd_func = fns.GNPF(a=a, b=b)
        eq_func = fns.VIPER_Env_Spec(a=a, b=b)
        eq_ages = age.equilibrium_age(diameters, pd_func.csfd, eq_func.csfd)

        np.testing.assert_allclose(
            eq_ages,
            np.array(
                [
                    1.359312e06,
                    6.918704e06,
                    2.213654e07,
                    2.899164e07,
                    3.573974e07,
                    6.550186e07,
                    1.381746e08,
                    4.223119e08,
                    7.262570e08,
                ]
            ),
            rtol=1e-6,
        )

    def test_estimate_age(self):
        a = age.estimate_age(10, 0.09, 5e7)
        self.assertAlmostEqual(a, 25000000, places=0)

    def test_estimate_age_by_bin(self):
        pd_func = fns.GNPF(a=1, b=1000)
        eq_func = fns.VIPER_Env_Spec(a=1, b=1000)

        df = pd.DataFrame(
            data={
                "diameter": [
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    3.0,
                    4.0,
                    4.0,
                    5.0,
                    5.0,
                    10.0,
                    10.0,
                    20.0,
                    20.0,
                    50.0,
                    50.0,
                    100.0,
                    100.0,
                ],
                "d/D": [
                    0.05,
                    0.1,
                    0.05,
                    0.1,
                    0.05,
                    0.1,
                    0.05,
                    0.1,
                    0.05,
                    0.1,
                    0.05,
                    0.1,
                    0.05,
                    0.1,
                    0.05,
                    0.12,
                    0.05,
                    0.13,
                ],
            }
        )
        df_out = age.estimate_age_by_bin(
            df,
            50,  # the bin size can have a real impact here.
            pd_func.csfd,
            eq_func.csfd,
        )

        age_series = pd.Series(
            [
                2000000,
                0,
                6000000,
                0,
                12000000,
                0,
                25000000,
                0,
                36000000,
                0,
                63000000,
                0,
                138000000,
                12000000,
                424000000,
                68000000,
                702000000,
                8000000,
            ],
            name="age",
        )

        pd.testing.assert_series_equal(age_series, df_out["age"])

        df2 = pd.DataFrame(
            data={
                "diameter": [100.0, 100.0, 100.0, 100.0],
                "d/D": [0.01, 0.06, 0.10, 0.17],
            }
        )

        df2_out = age.estimate_age_by_bin(
            df2,
            50,  # With only one diameter, num is irrelevant
        )

        age2_series = pd.Series([4500000000, 4500000000, 1388000000, 0], name="age")

        pd.testing.assert_series_equal(age2_series, df2_out["age"])
