#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module contains abstract and concrete classes for representing
rock size-frequency distributions as probability distributions.
"""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

import logging

import numpy as np

from synthterrain.crater.functions import Crater_rv_continuous

logger = logging.getLogger(__name__)


class InterCrater(Crater_rv_continuous):
    """
    This distribution is from the pre-existing synthetic_moon MATLAB code, no
    citation is given.
    """

    def csfd(self, d):
        # Low blockiness?
        return 0.0001 * np.float_power(d, -1.75457)


class VIPER_Env_Spec(Crater_rv_continuous):
    """
    This distribution is from the VIPER Environmental Specification,
    VIPER-MSE-SPEC-001 (2021-09-16).  Sadly, no citation is provided
    for it in that document.
    """

    def csfd(self, d):
        """
        CSFD( d ) = N_cum = 0.0003 / d^(2.482)
        """
        return 0.0003 * np.float_power(d, -2.482)


class Haworth(Crater_rv_continuous):
    """
    This distribution is from the pre-existing synthetic_moon MATLAB code, no
    citation is given.
    """

    def csfd(self, d):
        # High blockiness?
        return 0.002 * np.float_power(d, -2.6607)
