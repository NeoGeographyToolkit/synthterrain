#!/usr/bin/env python
"""This module has tests for the synthterrain crater init functions."""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

import unittest

from shapely.geometry import Point, Polygon

import synthterrain.crater as cr
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

