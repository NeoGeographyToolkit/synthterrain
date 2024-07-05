"""This module has tests for the synthterrain crater init functions."""

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

from shapely.geometry import Polygon

import synthterrain.crater as cr  # usort: skip
import synthterrain.crater.functions as fns


class Test_Init(unittest.TestCase):
    def test_generate_diameters(self):
        min_d = 10
        max_d = 11
        area = 10000
        cd = fns.Trask(a=min_d, b=max_d)

        d = cr.generate_diameters(cd, area, min_d, max_d)

        size = cd.count(area, min_d) - cd.count(area, max_d)
        self.assertEqual(size, d.size)
        self.assertEqual(0, d[d < min_d].size)
        self.assertEqual(0, d[d > max_d].size)

    def test_random_points(self):
        poly = Polygon(((0, 0), (1, 0), (0, 1), (0, 0)))

        xs, ys = cr.random_points(poly, 5)
