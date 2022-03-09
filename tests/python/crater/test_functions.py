#!/usr/bin/env python
"""This module has tests for the synthterrain crater distribution functions."""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

import unittest

import numpy as np

import synthterrain.crater.functions as fns


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
