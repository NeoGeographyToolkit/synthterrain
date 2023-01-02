#!/usr/bin/env python
"""This module has tests for the synthterrain crater age functions."""

# Copyright 2023, United States Government as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.

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

        np.testing.assert_allclose(eq_ages, np.array([
            1.35931206e+06, 6.91870352e+06, 2.21362552e+07, 2.89919160e+07,
            3.57407099e+07, 6.55084979e+07, 1.38188561e+08, 4.22354647e+08,
            7.26330624e+08
        ]))

    def test_estimate_age(self):
        a = age.estimate_age(10, 0.09, 5e7)
        self.assertAlmostEqual(a, 25322581, places=0)

    def test_estimate_age_by_bin(self):
        pd_func = fns.GNPF(a=1, b=1000)
        eq_func = fns.VIPER_Env_Spec(a=1, b=1000)

        df = pd.DataFrame(data={
            'diameter': [
                1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 10., 10.,
                20., 20., 50., 50., 100., 100.
            ],
            'd/D': [
                0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1,
                0.05, 0.1, 0.05, 0.12, 0.05, 0.13
            ]}
        )
        df_out = age.estimate_age_by_bin(
            df,
            pd_func.csfd,
            eq_func.csfd,
            num=50,  # the bin size can have a real impact here.
        )

        age_series = pd.Series([
            1.511054e+06, 3.538769e+03, 5.768174e+06, 1.286103e+04, 1.205410e+07,
            2.687647e+04, 2.517134e+07, 5.612338e+04, 3.585466e+07, 0.000000e+00,
            6.316055e+07, 2.944548e+05, 1.379950e+08, 1.251340e+07, 4.241947e+08,
            6.878833e+07, 7.023175e+08, 8.072615e+06], name="age")

        pd.testing.assert_series_equal(age_series, df_out["age"])
