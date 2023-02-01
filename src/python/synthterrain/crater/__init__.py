#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generates synthetic crater populations.
"""

# Copyright 2022-2023, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

import logging
import math
from pathlib import Path
import random

from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from synthterrain.crater import functions
from synthterrain.crater.age import equilibrium_age
from synthterrain.crater.diffusion import diffuse_d_over_D, diffuse_d_over_D_by_bin


logger = logging.getLogger(__name__)


def synthesize(
    crater_dist: functions.Crater_rv_continuous,
    polygon: Polygon,
    production_fn=None,
    by_bin=True,
    min_d=None,
    max_d=None,
):
    """Return a pandas DataFrame which contains craters and their properties
       synthesized from the input parameters.
    """
    if production_fn is None:
        production_fn = determine_production_function(crater_dist.a, crater_dist.b)
    logger.info(f"Production function is {production_fn.__class__}")

    # Get craters
    if min_d is None and max_d is None:
        diameters = crater_dist.rvs(area=polygon.area)
    elif min_d is not None and max_d is not None:
        diameters = generate_diameters(crater_dist, polygon.area, min_d, max_d)
    else:
        raise ValueError(
            f"One of min_d, max_d ({min_d}, {max_d}) was None, they must "
            "either both be None or both have a value."
        )
    logger.info(f"In {polygon.area} m^2, generated {len(diameters)} craters.")

    # Generate ages and start working with a dataframe.
    df = generate_ages(diameters, production_fn.csfd, crater_dist.csfd)
    logger.info(
        f"Generated ages from {math.floor(df['age'].min()):,} to "
        f"{math.ceil(df['age'].max()):,} years."
    )

    # Generate depth to diameter ratio
    if by_bin:
        df = diffuse_d_over_D_by_bin(df, start_dd_mean="Stopar step")
    else:
        df["d/D"] = df.apply(
            lambda crater: diffuse_d_over_D(crater["diameter"], crater["age"]),
            axis=1,
        )

    # Randomly generate positions within the polygon for the locations of
    # the craters.
    logger.info("Generating center positions.")
    # positions = random_points(polygon, len(diameters))
    xlist, ylist = random_points(polygon, len(diameters))
    # Add x and y positional information to the dataframe
    df["x"] = xlist
    df["y"] = ylist

    return df


def determine_production_function(a: float, b: float):
    if a >= 10:
        return functions.NPF(a, b)
    elif b <= 2.76:
        return functions.Grun(a, b)
    else:
        return functions.GNPF(a, b)


def random_points(poly: Polygon, num_points: int):
    """Returns two lists, the first being the x coordinates, and the second
       being the y coordinates, each *num_points* long that represent
       random locations within the provided *poly*.
    """
    # We could have returned a list of shapely Point objects, but that's
    # not how we need the data later.
    min_x, min_y, max_x, max_y = poly.bounds
    # points = []
    x_list = list()
    y_list = list()
    # while len(points) < num_points:
    while len(x_list) < num_points:
        random_point = Point(
            [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
        )
        if random_point.within(poly):
            # points.append(random_point)
            x_list.append(random_point.x)
            y_list.append(random_point.y)

    # return points
    return x_list, y_list


def generate_diameters(crater_dist, area, min, max):
    """
    Returns a numpy array with diameters selected from the *crater_dist*
    function based on the *area*, and no craters smaller than *min* or
    larger than *max* will be returned.
    """
    size = crater_dist.count(area, min) - crater_dist.count(area, max)
    diameters = list()

    while len(diameters) != size:
        d = crater_dist.rvs(size=(size - len(diameters)))
        diameters += d[
            np.logical_and(min <= d, d <= max)
        ].tolist()

    return np.array(diameters)


def generate_ages(diameters, pd_csfd, eq_csfd):
    """
    Returns a pandas DataFrame which contains "diameters" and "ages"
    columns with "diameters" being those provided via *diameters* and
    "ages" determined randomly from the range computed based on the
    provided equilibrium cumulative size frequency function, *eq_csfd*,
    and the the provided production cumulative size frequency function,
    *pd_csfd*.

    Both of these functions, when given a diameter, should return an actual
    cumulative count of craters per square meter (eq_csfd), and a rate of
    cratering in craters per square meter per Gigayear at that
    diameter (pd_csfd).
    """
    yrs_to_equilibrium = equilibrium_age(diameters, pd_csfd, eq_csfd)
    # print(yrs_to_equilibrium)

    ages = np.random.default_rng().uniform(0, yrs_to_equilibrium)

    return pd.DataFrame(data={"diameter": diameters, "age": ages})


def plot(df):
    """
    Generates a plot display with a variety of subplots for the provided
    pandas DataFrame, consistent with the columns in the DataFrame output
    by synthesize().
    """
    # Plots are:
    # CSFD, Age
    # d/D, location

    plt.ioff()
    fig, ((ax_csfd, ax_age), (ax_dd, ax_location)) = plt.subplots(2, 2)

    ax_csfd.hist(
        df["diameter"],
        cumulative=-1,
        log=True,
        bins=50,
        histtype='stepfilled',
        label="Craters"
    )
    ax_csfd.set_ylabel("Count")
    ax_csfd.yaxis.set_major_formatter(ScalarFormatter())
    ax_csfd.set_xlabel("Diameter (m)")
    ax_csfd.legend(loc='best', frameon=False)

    ax_age.scatter(df["diameter"], df["age"], alpha=0.2, edgecolors="none", s=10)
    ax_age.set_xscale("log")
    ax_age.xaxis.set_major_formatter(ScalarFormatter())
    ax_age.set_yscale("log")
    ax_age.set_ylabel("Age (yr)")
    ax_age.set_xlabel("Diameter (m)")

    ax_dd.scatter(df["diameter"], df["d/D"], alpha=0.2, edgecolors="none", s=10)
    ax_dd.set_xscale("log")
    ax_dd.xaxis.set_major_formatter(ScalarFormatter())
    ax_dd.set_ylabel("depth / Diameter")
    ax_dd.set_xlabel("Diameter (m)")

    patches = [
        Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(
            df["x"], df["y"], df["diameter"] / 2
        )
    ]
    collection = PatchCollection(patches)
    collection.set_array(df["d/D"])  # Sets circle color to this data property.
    ax_location.add_collection(collection)
    ax_location.autoscale_view()
    ax_location.set_aspect("equal")
    fig.colorbar(collection, ax=ax_location)

    plt.show()
    return


def to_file(df: pd.DataFrame, outfile: Path, xml=False):

    if xml:
        # Write out the dataframe in the XML style of the old MATLAB
        # program.
        df["rimRadius"] = df["diameter"] / 2

        # freshness float: 0 is an undetectable crater and 1 is "fresh"
        # The mapping from d/D to "freshness" is just the fraction of
        # d/D versus 0.2731, although I'm not sure why that value was selected.
        df["freshness"] = df["d/D"] / 0.2731

        # This indicates whether a crater in the listing was synthetically
        # generated or not, at this time, all are.
        df["isGenerated"] = 1

        df.to_xml(
            outfile,
            index=False,
            root_name="CraterList",
            row_name="CraterData",
            parser="etree",
            attr_cols=["x", "y", "rimRadius", "freshness", "isGenerated"],
        )
    else:
        df.to_csv(
            outfile,
            index=False,
            columns=["x", "y", "diameter", "age", "d/D"],
        )

    return


def from_file(infile: Path):
    """Load previously written crater information from disk"""
    if infile.suffix == ".xml":
        df = pd.read_xml(infile)
    else:
        df = pd.read_csv(infile)
    return df
