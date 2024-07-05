"""This module has tests for the synthterrain crater distribution functions."""

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
from numpy.polynomial import Polynomial

import synthterrain.crater.functions as fns  # usort: skip


class Test_Crater_rv_continuous(unittest.TestCase):
    def test_abstract(self):
        self.assertRaises(TypeError, fns.Crater_rv_continuous)

        class Test_Dist(fns.Crater_rv_continuous):
            def csfd(self, d):
                return np.power(d, -2.0)

        self.assertRaises(TypeError, Test_Dist)
        self.assertRaises(ValueError, Test_Dist, a=0)

        d = Test_Dist(a=1)
        self.assertEqual(d._cdf(10.0), 0.99)
        self.assertEqual(len(d.rvs(area=1)), 1)

    def test_VIPER_Env_Spec(self):
        rv = fns.VIPER_Env_Spec(a=1, b=100)
        self.assertAlmostEqual(rv.csfd(10), 0.0003507)
        np.testing.assert_allclose(
            rv._cdf(
                np.array(
                    [
                        10,
                    ]
                )
            ),
            np.array([0.98797736]),
        )

        np.testing.assert_allclose(
            rv._ppf(
                np.array(
                    [
                        0.5,
                        0.99,
                    ]
                )
            ),
            np.array([1.43478377, 11.00694171]),
        )

    def test_Trask(self):
        rv = fns.Trask(a=10, b=100)
        self.assertEqual(rv.csfd(20), 0.00019858205868107036)

    def test_Coef_Distribution(self):
        self.assertRaises(ValueError, fns.Coef_Distribution, a=10, b=300000)

        rv = fns.Coef_Distribution(
            a=10,
            b=300000,
            poly=Polynomial(
                [
                    -3.0768,
                    -3.557528,
                    0.781027,
                    1.021521,
                    -0.156012,
                    -0.444058,
                    0.019977,
                    0.086850,
                    -0.005874,
                    -0.006809,
                    8.25e-04,
                    5.54e-05,
                ]
            ),
        )
        self.assertEqual(rv.csfd(10), 0.003796582136635746)
        self.assertEqual(
            rv._cdf(
                np.array(
                    [
                        10,
                    ]
                )
            ),
            np.array(
                [
                    0,
                ]
            ),
        )

    def test_NPF(self):
        self.assertRaises(ValueError, fns.NPF, a=10, b=300001)
        self.assertRaises(ValueError, fns.NPF, a=9, b=300)
        rv = fns.NPF(a=10, b=300000)
        self.assertEqual(rv.csfd(10), 0.0037969668020723783)
