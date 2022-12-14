#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generates synthetic rock populations.
"""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

import logging
import math
from pathlib import Path
import time

import opensimplex
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from rasterio.features import geometry_mask
from rasterio.transform import from_origin, rowcol, xy
from rasterio.windows import Window, from_bounds, intersect, shape
import pandas as pd
from shapely.geometry import Polygon


from synthterrain.crater import generate_diameters
from synthterrain.crater.functions import Crater_rv_continuous

logger = logging.getLogger(__name__)


def crater_probability_map(diameter_px, domain_size=None, decay_multiplier=(-1 / 0.7)):
    """
    Returns two numpy arrays of size (*domain_size*, *domain_size*) representing the
    probability of rocks exterior to a crater of *diameter_px* and the second is the
    probability of rocks interior to the crater of that size.

    If *domain_size* is not given, it will be twice *diameter_px*.

    The *decay_multiplier* is multiplied by the distance from the crater center in the
    exponential that defines the radial probability field outside of the rim.  Larger
    negative numbers result in steeper fall-offs.  Positive values will result in a
    ValueError.
    """
    if decay_multiplier > 0:
        raise ValueError(
            "The decay_multiplier must be negative, otherwise the probability increases"
            "as you move outwards."
        )

    if domain_size is None:
        # Span a space 2x the diameter.
        domain_size = math.ceil(diameter_px * 2)
        domain_edge = 2
    else:
        domain_edge = domain_size / diameter_px
    x = np.linspace(-1 * domain_edge, domain_edge, domain_size)

    # The array rr contains radius fraction values from the center of the
    # domain.
    xx, yy = np.meshgrid(x, x, sparse=True)  # square domain
    rr = np.sqrt(xx**2 + yy**2)

    outer_pmap = np.zeros_like(rr)

    inner_idx = rr <= 1
    outer_idx = np.logical_and(1 < rr, rr <= domain_edge)

    outer_pmap[outer_idx] = np.exp(decay_multiplier * rr[outer_idx])

    inner_pmap = np.zeros_like(rr)
    inner_pmap[inner_idx] = 0.05 * np.amax(outer_pmap)

    return outer_pmap, inner_pmap


def craters_probability_map(
    crater_frame: pd.DataFrame,
    transform,
    window: Window,
    rock_age_decay=3,
):
    """
    Returns a 2D numpy array whose elements sum to 1 which describes the probability
    of a rock existing in the ejecta pattern of the craters in *crater_frame*.  The
    affine *transform* describes the relation of the *window* to the x,y coordinates
    of the craters in *crater_frame*.  The *rock_age_decay* parameter is the exponential
    in the equation that determines how the probability field changes with crater
    age.
    """
    if abs(transform.a) != abs(transform.e):
        raise ValueError("The transform does not have even spacing in X and Y.")
    pmap = np.zeros(shape(window))
    df = crater_frame.sort_values(by="age", ascending=False)
    for row in df.itertuples(index=False):
        outer, inner = crater_probability_map(row.diameter / abs(transform.a))

        # print(pmap.shape)
        # print(outer.shape)
        row_center, col_center = rowcol(transform, row.x, row.y)
        # print(row_center)
        # print(col_center)
        crater_window = Window(
            col_center - int(outer.shape[0] / 2),
            row_center - int(outer.shape[1] / 2),
            *outer.shape,
        )
        # print(f"crater_window: {crater_window}")

        if not intersect((window, crater_window)):
            # print("does not intersect")
            continue

        window_inter = window.intersection(crater_window)
        # print(f"window_inter {window_inter}")
        # print(window_inter.toslices())
        crater_inter = intersection_relative_to(crater_window, window)
        # print(f"crater_inter {crater_inter}")

        # Need to determine scheme for reducing the outer and inner maps relative
        # to their age, but original code was actually based on d/D and otherwise
        # backwards (more degraded craters had larger probability multipliers.
        age_multiplier = np.power(1 - (row.age / 4e9), rock_age_decay)

        if age_multiplier < 0:
            age_multiplier = 0

        outer *= age_multiplier
        inner *= age_multiplier

        # Ejecta field adds to probability:
        pmap[window_inter.toslices()] = (
            pmap[window_inter.toslices()] + outer[crater_inter.toslices()]
        )

        # Interior of crater replaces:
        pmap[window_inter.toslices()] = np.where(
            inner[crater_inter.toslices()] > 0,
            inner[crater_inter.toslices()],
            pmap[window_inter.toslices()],
        )

    return pmap / np.sum(pmap)


def intersection_relative_to(w1, w2):
    """
    Returns the intersection of the two windows relative to the row_off and col_off of
    *w1*.
    """

    row_shift = w1.row_off
    col_shift = w1.col_off

    w1_shift = Window(0, 0, w1.width, w1.height)
    w2_shift = Window(
        w2.col_off - col_shift, w2.row_off - row_shift, w2.width, w2.height
    )

    return w1_shift.intersection(w2_shift)


def place_rocks(
    diameters,
    polygon: Polygon,
    pmap: np.array,
    transform,
    seed=None,
    epsilon=0.0001,
):
    """
    Return a pandas DataFrame containing the diameter and x,y location of the rocks
    provided in *diameters*.  These locations will be interior to *polygon* and
    will conform to the affine *transform* which describes the relation of the *polygon*
    to the *pmap* array which should be a probability map.

    A ValueError will be raised if the elements of *pmap* sum to less than *epsilon*.
    """
    # Mask the probability map
    mask = geometry_mask((polygon,), pmap.shape, transform)
    pmap[mask] = 0

    prob_map_sum = np.sum(pmap)

    if prob_map_sum < epsilon:
        raise ValueError(f"The sum of the pmap is less than {epsilon}!")

    logger.debug(f"The probability map sums to {prob_map_sum} before normalization.")

    # Flatten and normalize the probability map for choosing.
    # flat_prob_map = pmap.ravel() / prob_map_sum
    flat_prob_map = pmap.ravel()

    rng = np.random.default_rng(seed)
    position_idxs = rng.choice(
        len(flat_prob_map),
        size=len(diameters),
        replace=False,
        p=flat_prob_map,
    )

    # Convert indexes back to row/column coordinates and then x/y coordinates.
    rows, cols = np.unravel_index(position_idxs, pmap.shape)
    xs, ys = xy(transform, rows, cols)

    # Select sub-pixel location:
    delta_pos = rng.random((2, len(diameters)))

    xs += delta_pos[0] * abs(transform.a)
    ys += delta_pos[1] * abs(transform.a)

    return pd.DataFrame(data={"diameter": diameters, "x": xs, "y": ys})


def random_probability_map(rows, cols, seed=None):
    """
    Returns a 2D numpy array with shape (*rows*, *cols*) which is a random probability
    map.

    This function uses opensimplex to produce 2D noise as a basis for
    the map, and then normalizes the returned array, such that all of
    its probabilities sum to 1.

    If *rows* or *cols* is less than 10, then a uniform probability map is returned.
    """

    if rows < 10 or cols < 10:
        p_map = np.ones((rows, cols))
    else:
        if seed is None:
            opensimplex.seed(time.time_ns())
        else:
            opensimplex.seed(seed)
        noise = opensimplex.noise2array(np.arange(cols), np.arange(rows))
        p_map = (noise + 1) / 2  # noise runs from -1 to +1

    return p_map / np.sum(p_map)


def synthesize(
    rock_dist: Crater_rv_continuous,
    polygon: Polygon,
    pmap_gsd: int,
    crater_frame=None,
    min_d=None,
    max_d=None,
    seed=None,
):
    """
    Return a two-tuple with a pandas DataFrame and a 2D numpy array.

    The DataFrame contains information about rocks and their properties synthesized
    from the input parameters.  The 2D numpy array is the probability map used to
    generate the x and y values in the DataFrame.
    """

    logger.info(f"Rock distribution function is {rock_dist.__class__}")
    # Get Rocks
    if min_d is None and max_d is None:
        diameters = rock_dist.rvs(area=polygon.area)
    elif min_d is not None and max_d is not None:
        diameters = generate_diameters(rock_dist, polygon.area, min_d, max_d)
    else:
        raise ValueError(
            f"One of min_d, max_d ({min_d}, {max_d}) was None, they must "
            "either both be None or both have a value."
        )
    logger.info(f"In {polygon.area} m^2, generated {len(diameters)} rocks.")

    # Build probability map
    (minx, miny, maxx, maxy) = polygon.bounds
    transform = from_origin(minx, maxy, pmap_gsd, pmap_gsd)
    window = (
        from_bounds(minx, miny, maxx, maxy, transform).round_lengths().round_offsets()
    )
    random_map = random_probability_map(*shape(window), seed)
    logger.info(f"Random probability map of size {random_map.shape} generated.")
    logger.debug(f"Random probability map sums to {np.sum(random_map)}.")

    if crater_frame is not None:
        crater_map = craters_probability_map(crater_frame, transform, window)
        pmap = (random_map + crater_map) / 2
    else:
        pmap = random_map

    return place_rocks(diameters, polygon, pmap, transform, seed), pmap


def plot(df, pmap=None, extent=None):
    """
    Generates a plot display with a variety of subplots for the provided
    pandas DataFrame, consistent with the columns in the DataFrame output
    by synthesize().
    """
    # Plots are:
    # CSFD
    # probability map, location map

    plt.ioff()
    # fig, ((ax_csfd, ax_), (ax_pmap, ax_location)) = plt.subplots(2, 2)
    fig, (ax_csfd, ax_pmap, ax_location) = plt.subplots(1, 3)

    ax_csfd.hist(
        df["diameter"],
        cumulative=-1,
        log=True,
        bins=50,
        histtype="stepfilled",
        label="Rocks",
    )
    ax_csfd.set_ylabel("Count")
    ax_csfd.yaxis.set_major_formatter(ScalarFormatter())
    ax_csfd.set_xlabel("Diameter (m)")
    ax_csfd.legend(loc="best", frameon=False)

    # ax_age.scatter(df["diameter"], df["age"], alpha=0.2, edgecolors="none", s=10)
    # ax_age.set_xscale("log")
    # ax_age.xaxis.set_major_formatter(ScalarFormatter())
    # ax_age.set_yscale("log")
    # ax_age.set_ylabel("Age (yr)")
    # ax_age.set_xlabel("Diameter (m)")

    ax_pmap.imshow(pmap)

    ax_location.imshow(pmap, extent=extent)
    patches = [
        Circle((x_, y_), s_)
        for x_, y_, s_ in np.broadcast(df["x"], df["y"], df["diameter"] / 2)
    ]
    collection = PatchCollection(patches)
    collection.set_color("white")
    ax_location.add_collection(collection)
    ax_location.autoscale_view()
    ax_location.set_aspect("equal")

    plt.show()
    return


def to_file(df: pd.DataFrame, outfile: Path, xml=False):

    if xml:
        # Write out the dataframe in the XML style of the old MATLAB
        # program.
        df.to_xml(
            outfile,
            index=False,
            root_name="RockList",
            row_name="RockData",
            parser="etree",
            attr_cols=["diameter", "x", "y"],
        )
    else:
        df.to_csv(
            outfile,
            index=False,
            columns=["diameter", "x", "y"],
        )

    return
