# -*- coding: utf-8 -*-
"""
Functions for estimating crater ages.
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

import bisect
import logging

import numpy as np
import pandas as pd

from synthterrain.crater.diffusion import diffuse_d_over_D
from synthterrain.crater.profile import stopar_fresh_dd


logger = logging.getLogger(__name__)


def equilibrium_age(diameters, pd_csfd, eq_csfd):
    """
    Returns a numpy array which contains the equilibrium ages
    which correspond to the craters provided via *diameters*
    computed based on the provided equilibrium cumulative size frequency function,
    *eq_csfd*, and the the provided production cumulative size frequency function,
    *pd_csfd*.

    Both of these functions, when given a diameter, should return an actual
    cumulative count of craters per square meter (eq_csfd), and a rate of
    cratering in craters per square meter per Gigayear at that
    diameter (pd_csfd).
    """
    upper_diameters = np.float_power(10, np.log10(diameters) + 0.1)
    eq = eq_csfd(diameters) - eq_csfd(upper_diameters)
    pf = pd_csfd(diameters) - pd_csfd(upper_diameters)

    return 1e9 * eq / pf


def estimate_age(diameter, dd, max_age):
    """
    Estimates the age of a crater in years given its diameter in meters and the
    d/D value by attempting to match the given d/D value to the diffusion shape
    of a crater of the given *diameter* and the *max_age* of that crater, which
    could be estimated via the equilibrium_ages() function.

    This function returns estimated ages in multiples of a million years.  More
    precision than that is not accurate for this approach.
    """
    fresh_dd = stopar_fresh_dd(diameter)

    if dd > fresh_dd:
        return 0

    dd_rev_list = list(
        reversed(
            diffuse_d_over_D(
                diameter, max_age, start_dd_adjust=fresh_dd, return_steps=True
            )
        )
    )
    nsteps = len(dd_rev_list)

    age_step = bisect.bisect_left(dd_rev_list, dd)

    years_per_step = max_age / nsteps

    age = (nsteps - age_step) * years_per_step

    return round(age / 1e6) * 1e6


def estimate_age_by_bin(
    df,
    num=50,
    pd_csfd=None,
    eq_csfd=None,
) -> pd.DataFrame:
    """
    Returns a pandas DataFrame identical to the input *df* but with the
    addition of two columns: "diameter_bin" and "age".  The ages are in years
    and are estimated from the "diameter" and "d/D" columns
    in the provided pandas DataFrame, *df*.

    For lage numbers of craters, running a estimate_age() for each can be
    computationally intensive.  This function generates *num* bins with log
    scale boundaries (using the numpy geomspace() function) between the maximum
    and minimum diameter values.

    Then, the center diameter of each bin has diffuse_d_over_D()
    run to evaluate the d/D ratio of that crater over the lifetime
    of the solar system (if *pd_csfd* and *eq_csfd* are both None), or
    the diffusion calculation is run for the appropriate equilibrium age
    of each crater diameter.

    Then, for each crater in the bin, its d/D is compared to the d/D ratios from the
    diffusion run, and an estimated age is assigned.
    """
    if df.empty:
        raise ValueError("The provided dataframe has no rows.")

    logger.info("estimate_age_by_bin start.")
    if df["diameter"].min() == df["diameter"].max():
        bin_edges = 1
        total_bins = 1
        logger.info("The craters are all the same size.")
    else:
        bin_edges = np.geomspace(
            df["diameter"].min(), df["diameter"].max(), num=num + 1
        )
        total_bins = num
        # logger.info(f"{df.shape[0]} craters")
        logger.info(
            f"Divided the craters into {num} diameter bins (not all bins may have "
            "craters)"
        )

    df["diameter_bin"] = pd.cut(
        df["diameter"],
        bins=bin_edges,
        include_lowest=True,
    )

    # df["equilibrium_age"] = equilibrium_ages(df["diameter"], pd_csfd, eq_csfd)
    df["age"] = 0

    for i, (interval, count) in enumerate(
        df["diameter_bin"].value_counts(sort=False).items()
    ):
        logger.info(
            f"Processing bin {i + 1}/{total_bins}, interval: {interval}, count: {count}"
        )

        if count == 0:
            continue

        fresh_dd = stopar_fresh_dd(interval.mid)

        if pd_csfd is not None and eq_csfd is not None:
            age = equilibrium_age(interval.mid, pd_csfd, eq_csfd)
        else:
            age = 4.5e9

        dd_rev_list = list(
            reversed(
                diffuse_d_over_D(
                    interval.mid, age, start_dd_adjust=fresh_dd, return_steps=True
                )
            )
        )
        nsteps = len(dd_rev_list)
        years_per_step = age / nsteps

        def guess_age(dd):
            age_step = bisect.bisect_left(dd_rev_list, dd)
            return round(int((nsteps - age_step) * years_per_step) / 1e6) * 1e6

        df.loc[df["diameter_bin"] == interval, "age"] = df.loc[
            df["diameter_bin"] == interval
        ].apply(lambda row: guess_age(row["d/D"]), axis=1)

    df["age"] = df["age"].astype("int64")

    logger.info("estimate_age_by_bin complete.")

    return df
