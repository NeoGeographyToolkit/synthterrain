# -*- coding: utf-8 -*-
"""This module contains abstract and concrete classes for representing
rock size-frequency distributions as probability distributions.
"""

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
