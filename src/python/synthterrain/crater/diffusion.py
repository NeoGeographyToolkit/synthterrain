#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performs diffusion of a crater shape.

A descriptive guide to the approach used here can be found in
Chapter 7 of Learning Scientific Programming with Python by
Christian Hill, 2nd Edition, 2020.  An online version can be found
at
https://scipython.com/book2/chapter-7-matplotlib/examples/the-two-dimensional-diffusion-equation/
"""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

import logging
import math
import statistics

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd

from synthterrain.crater.profile import FTmod_Crater, stopar_fresh_dd


logger = logging.getLogger(__name__)


def diffusion_length_scale(diameter: float, domain_size: int) -> float:
    """
    Returns the appropriate "diffusion length scale" based on the provided
    *diameter* and the *domain_size*.  Where *domain_size* is the width of a
    square grid upon which the diffusion calculation is performed.

    In the equations and code detailed in Hill (2020), the diffusion
    coefficient is denoted as D, and the time step as 𝚫t or dt.

    The equation for dt is:

        dt = (dx dy)^2 / (2 * D * (dx^2 + dy^2))

    In the case where dx = dy, this simplifies to:

        dt = dx^2 / (4 * D)

    For our approach, it is useful to define a "diffusion length scale" which
    we'll denote as dls, such that

        dt = dls / D

    This value of dls can then be easily used in other functions in this
    module.
    """
    return math.pow(diameter * 2 / domain_size, 2) / 4


def kappa_diffusivity(diameter: float) -> float:
    """
    Returns the diffusivity for a crater with a *diameter* in meters.  The
    returned diffusivity is in units of m^2 per year.

    This calculation is based on Fassett and Thomson (2014,
    https://doi.org/10.1002/2014JE004698) and Fassett and Thomson (2015,
    https://ui.adsabs.harvard.edu/abs/2015LPI....46.1120F/abstract),
    but modeling done by Caleb Fassett since then has influenced this
    function.
    """
    # Fassett and Thomson (2014) don't actually define this functional
    # form, this was provided to me by Caleb.
    # The *kappa0* value is the diffusivity at 1 km, and *kappa_corr* is the
    # power law exponent for correcting this to small sizes.
    # Fassett and Thomson (2015) indicates *kappa0* is 5.5e-6 m^2 / year
    # (the default).
    # kappa0=5.5e-6, kappa_corr=0.9
    # return kappa0 * math.pow(diameter / 1000, kappa_corr)

    # The logic below replaced the above simple logic on
    # 2022-05-22 by Caleb.
    if diameter <= 11.2:
        k = 0.0155  # m2/myr
    elif diameter < 45:
        k = 1.55e-3 * math.pow(diameter, 0.974)
    elif diameter < 125:
        k = 1.23e-3 * math.pow(diameter, 0.8386)
    else:  # UNCONSTRAINED BY EQUILIBRIUM ABOVE 125m!!!!!!!
        k = 5.2e-3 * math.pow(diameter, 1.3)

    return k / 1.0e6  # m2/yr


def diffuse_d_over_D(
    diameter,
    age,
    domain_size=200,
    start_dd_adjust=False,
    start_dd_mean=0.15,
    start_dd_std=0.02,
    return_steps=False,
    return_surface=False,
    crater_cls=FTmod_Crater
):
    """
    Returns a depth to diameter ratio of a crater of *diameter* in meters
    and *age* in years after the model in Fassett and Thomson (2014,
    https://doi.org/10.1002/2014JE004698).

    The *domain_size* is the width of a square grid upon which the
    diffusion calculation is performed.

    If *start_dd_adjust* is True, the initial d/D ratio of the
    crater will be determined by randomly selecting a value from
    a normal distribution with a mean of *start_dd_mean* and
    and standard deviation of *start_dd_std*.

    If *start_dd_adjust* is numeric, then the starting depth to Diameter
    ratio will be set to this specific value (and *start_dd_mean* and
    *start_dd_std* will be ignored).

    In order to make no changes to the initial crater shape, set
    *start_dd_adjust* to False (the default).

    If *return_steps* is True, instead of a single depth to diameter ratio
    returned, a list of depth to diameter ratios is returned, one for each
    time step in the diffusion process, with the last list item being identical
    to what would be returned when *return_steps* is False (the default).

    If *return_surface* is True, instead of a single object returned, a tuple
    will be returned, where the nature of the zeroth element is based on
    *return_steps* and the final element is the numpy array of elevations
    which represents the final diffused surface relative to a starting flat
    surface of zero.

    If a different crater profile is desired, pass a subclass (not an instance)
    of crater.profile.Crater to *crater_cls* that takes a depth parameter,
    otherwise defaults to crater.profile.FTmod_Crater.
    """
    # Set up grid and initialize crater shape.

    # sscale = diameter / 50
    dx = diameter * 2 / domain_size  # Could define dy, but would be identical
    dx2 = dx * dx

    # The array rr contains radius fraction values from the center of the
    # domain.
    x = np.linspace(-2, 2, domain_size)  # spans a space 2x the diameter.
    xx, yy = np.meshgrid(x, x, sparse=True)  # square domain
    rr = np.sqrt(xx**2 + yy**2)

    # Create crater with the right depth
    if isinstance(start_dd_adjust, bool):
        if start_dd_adjust:
            # u *= np.random.normal(start_dd_mean, start_dd_std) / crater_dd
            s_dd = np.random.normal(start_dd_mean, start_dd_std)
            crater = crater_cls(diameter, depth=(s_dd * diameter))
        else:
            crater = crater_cls(diameter)
    else:
        # u *= start_dd_adjust / crater_dd
        crater = crater_cls(diameter, depth=(start_dd_adjust * diameter))

    # Now create starting height field:
    u = crater.profile(rr)

    # This commented block is a structure from Caleb's code, which was mostly
    # meant for larger craters, but for small craters (<~ 10 m), the d/D
    # basically gets set to 0.04 to start and then diffuses to nothing.
    #
    # This text was in the docstring:
    # If the *diameter* is <= *d_mag_thresh* the topographic amplitude
    # of the starting crater shape will be randomly magnified by a factor
    # equal to 0.214 * variability * (diameter^(0.22)), where the variability
    # is randomly selected from a normal distribution with a mean of 1 and
    # a standard deviation of 0.1.
    #
    # # Magnify the topographic amplitude for smaller craters.
    # # Caleb indicates this is from Mahanti et al. (2018), but I can't
    # # quite find these numbers
    # if diameter <= d_mag_thresh:
    #     variability = np.random.normal(1.0, 0.1)
    #     # print(f"variability {0.214 * variability}")
    #     u *= 0.214 * variability * (diameter**(0.22))  # modified from Mahanti

    # Set up diffusion calculation parameters.

    kappaT = kappa_diffusivity(diameter) * age
    # print(f"kappaT: {kappaT}")

    dls = diffusion_length_scale(diameter, domain_size)

    nsteps = math.ceil(kappaT / dls)

    # D * dt appears in the Hill (2020) diffusion calculation, but
    # we have formulated dls, such that D * dt = dls

    dd_for_each_step = list()
    un = np.copy(u)
    for step in range(nsteps):
        un[1:-1, 1:-1] = u[1:-1, 1:-1] + dls * (
            (
                u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]
            ) / dx2 + (
                u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]
            ) / dx2
        )
        dd_for_each_step.append((np.max(u) - np.min(u)) / diameter)
        u = np.copy(un)

    # Final depth to diameter:
    if return_steps:
        dd = dd_for_each_step
    else:
        dd = dd_for_each_step[-1]

    if return_surface:
        return dd, u
    else:
        return dd


def diffuse_d_over_D_by_bin(
    df,
    num=50,
    domain_size=200,
    start_dd_mean=0.15,
    start_dd_std=0.02,
) -> pd.DataFrame:
    """
    Returns a pandas DataFrame identical to the input *df* but with the
    addition of two columns: "diameter_bin" and "d/D".  The depth to Diameter
    ratios in column "d/D" are estimated from the "diameter" and "age" columns
    in the provided pandas DataFrame, *df*.

    For lage numbers of craters, running a diffusion model via
    diffuse_d_over_D() for each can be computationally intensive.  This
    function generates *num* bins with log scale boundaries (using the numpy
    geomspace() function) between the maximum and minimum diameter values.

    If there are three or fewer craters in a size bin, then diffuse_d_over_D()
    is run with start_dd_adjust=True, and the *start_dd_mean*, and
    *start_dd_std* set as specified for each individual crater.

    If there are more than three craters in a size bin, then diffuse_d_over_D()
    is run three times, once with start_dd_adjust=*start_dd_mean*, once with
    *start_dd_mean* - *start_dd_std*, and once with *start_dd_mean* +
    *start_dd_std*.  These three models then essentially provide three values
    for d/D for each time step that represent a "high", "mean", and "middle"
    d/D ratio.

    Then, for each crater in the bin, at the age specified, a d/D value is
    determined by selecting from a normal distribution with a mean d/D provided
    by the "mean" d/D ratio curve, and a standard deviation specified by the
    mean of the difference of the "high" and "low" diffiusion model values with
    the "mean" d/D model value at that time step.
    """
    logger.info("diffuse_d_over_D_by_bin start.")

    bin_edges = np.geomspace(
        df["diameter"].min(), df["diameter"].max(), num=num + 1
    )
    # logger.info(f"{df.shape[0]} craters")
    logger.info(
        f"Divided the craters into {num} diameter bins (not all bins may have "
        "craters)"
    )

    if start_dd_mean == "Stopar fit":
        # This is a 3-degree fit to the data from Stopar et al. (2017)
        # The resulting function, stopar_dD() will return d/D ratios when
        # given a diameter in meters.
        stopar_poly = Polynomial([
            1.23447427e-01, 1.49135061e-04, -6.16681361e-08, 7.08449143e-12
        ])

        def start_dd(diameter):
            if diameter < 850:
                return stopar_poly(diameter)
            else:
                return 0.2

        def start_std(diameter):
            if diameter < 10:
                return start_dd_std + 0.01
            else:
                return start_dd_std

    elif start_dd_mean == "Stopar step":
        # Stopar et al. (2017) define a set of graduate d/D categories
        # defined down to 40 m.  This creates two extrapolated categories:
        # def start_dd(diameter):
        #     # The last two elements are extrapolated
        #     d_lower_bounds = (400, 200, 100, 40, 10, 0)
        #     dds = (0.21, 0.17, 0.15, 0.13, 0.11, 0.10)
        #     for d, dd in zip(d_lower_bounds, dds):
        #         if diameter > d:
        #             return dd
        #     else:
        #         raise ValueError("Diameter was less than zero.")
        start_dd = stopar_fresh_dd

        def start_std(diameter):
            # if diameter < 10:
            #     return start_dd_std + 0.01
            # else:
            return start_dd_std

    else:
        def start_dd(diameter):
            return start_dd_mean

        def start_std(diameter):
            return start_dd_std

    df["diameter_bin"] = pd.cut(df["diameter"], bins=bin_edges)
    df["d/D"] = 0.0

    # Need to convert this loop to multiprocessing.
    for i, (interval, count) in enumerate(
        df["diameter_bin"].value_counts(sort=False).items()
    ):
        logger.info(
            f"Processing bin {i}/{num}, interval: {interval}, count: {count}"
        )

        if count == 0:
            continue
        elif 0 < count <= 3:
            # Run individual models for each crater.
            df.loc[df["diameter_bin"] == interval, "d/D"] = df.loc[
                df["diameter_bin"] == interval
            ].apply(
                lambda row: diffuse_d_over_D(
                    row["diameter"],
                    row["age"],
                    domain_size=domain_size,
                    start_dd_adjust=True,
                    start_dd_mean=start_dd(row["diameter"]),
                    start_dd_std=start_dd(row["diameter"]),
                ),
                axis=1
            )
        else:
            # Run three representative models for this "bin"
            oldest_age = df.loc[df["diameter_bin"] == interval, "age"].max()

            start = start_dd(interval.mid)
            std = start_std(interval.mid)

            middle_dds = diffuse_d_over_D(
                interval.mid,
                oldest_age,
                domain_size=domain_size,
                start_dd_adjust=start,
                return_steps=True
            )
            high_dds = diffuse_d_over_D(
                interval.mid,
                oldest_age,
                domain_size=domain_size,
                start_dd_adjust=start + std,
                return_steps=True
            )
            low_dds = diffuse_d_over_D(
                interval.mid,
                oldest_age,
                domain_size=domain_size,
                start_dd_adjust=start - std,
                return_steps=True
            )

            kappa = kappa_diffusivity(interval.mid)
            dls = diffusion_length_scale(interval.mid, domain_size)

            # Defining this in-place since it really isn't needed outside
            # this function.
            def dd_from_rep(age):
                age_step = math.floor(age * kappa / dls)
                return np.random.normal(
                    middle_dds[age_step],
                    statistics.mean([
                        middle_dds[age_step] - low_dds[age_step],
                        high_dds[age_step] - middle_dds[age_step]
                    ])
                )

            df.loc[df["diameter_bin"] == interval, "d/D"] = df.loc[
                df["diameter_bin"] == interval
            ].apply(
                lambda row: dd_from_rep(row["age"]),
                axis=1
            )

    logger.info("diffuse_d_over_D_by_bin complete.")

    return df
