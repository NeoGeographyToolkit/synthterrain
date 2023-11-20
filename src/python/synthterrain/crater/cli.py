#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generates synthetic crater populations.
"""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

import logging
import math
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from shapely.geometry import box

import synthterrain.crater as crater
from synthterrain.crater import functions
from synthterrain.crater.diffusion import make_crater_field
import synthterrain.util as util


logger = logging.getLogger(__name__)

# Assemble this global variable value.
csfd_dict = dict()
for fn in functions.equilibrium_functions:
    csfd_dict[fn.__name__] = fn.__doc__


def arg_parser():
    parser = util.FileArgumentParser(
        description=__doc__,
        parents=[util.parent_parser()],
        # formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # parser.add_argument(
    #     "--area",
    #     type=float,
    #     help="Area in square kilometers to evaluate."
    # )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=[0, 1000, 1000, 0],
        metavar=("MINX", "MAXY", "MAXX", "MINY"),
        help="The coordinates of the bounding box, expressed in meters, to "
        "evaluate in min-x, max-y, max-x, min-y order (which is ulx, "
        "uly, lrx, lry, the GDAL pattern). "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--csfd",
        default="VIPER_Env_Spec",
        choices=csfd_dict.keys(),
        help="The name of the crater size-frequency distribution to use. "
        f"Options are: {', '.join(csfd_dict.keys())}. "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--maxd",
        default=1000,
        type=float,
        help="Maximum crater diameter in meters. Default: %(default)s",
    )
    parser.add_argument(
        "--mind",
        default=1,
        type=float,
        help="Minimum crater diameter in meters. Default: %(default)s",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="This will cause a matplotlib window to open with some summary "
        "plots after the program has generated the data.",
    )
    parser.add_argument(
        "-proj",
        help="If -t is given this is needed to determine the proj string for the "
        "output GeoTIFF.  e.g. '+proj=eqc +R=1737400 +units=m'",
    )
    parser.add_argument(
        "--run_individual",
        # Inverted the action to typically be set to true as it will be
        # given to the by_bin parameter of synthesize.
        action="store_false",
        help="If given, this will run a diffusion model for each synthetic "
        "crater individually and depending on the area provided and the "
        "crater range could cause this program to run for many hours as "
        "it tried to calculate tens of thousands of diffusion models. "
        "The default behavior is to gather the craters into diameter bins "
        "and only run a few representative diffusion models to span the "
        "parameter space.",
    )
    parser.add_argument(
        "-t",
        "--terrain_gsd",
        type=float,
        help="If provided, will trigger creation of a terrain model based on bbox, and "
        "the value given here will set the ground sample distance (GSD) of that model. "
        "The terrain model output file will be the same as --outfile, but with a .tif "
        "ending.",
    )
    parser.add_argument(
        "-x",
        "--xml",
        action="store_true",
        help="Default output is in CSV format, but if given this will result "
        "in XML output that conforms to the old MATLAB code.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--csfd_info",
        action=util.PrintDictAction,
        dict=csfd_dict,
        help="If given, will list detailed information about each of the "
        "available CSFDs and exit.",
    )
    group.add_argument("-o", "--outfile", type=Path, help="Path to output file.")
    return parser


def main():
    args = arg_parser().parse_args()

    util.set_logger(args.verbose)

    # if args.csfd_info:
    #     print(csfd_dict)
    #     return

    # This could more generically take an arbitrary polygon
    # bbox argument takes: 'MINX', 'MAXY', 'MAXX', 'MINY'
    # the box() function takes: (minx, miny, maxx, maxy)
    poly = box(args.bbox[0], args.bbox[3], args.bbox[2], args.bbox[1])

    crater_dist = getattr(functions, args.csfd)(a=args.mind, b=args.maxd)

    df = crater.synthesize(
        crater_dist,
        polygon=poly,
        by_bin=args.run_individual,
        min_d=args.mind,
        max_d=args.maxd,
        return_surfaces=bool(args.terrain_gsd),
    )

    if args.plot:
        crater.plot(df)

    # Write out results.
    crater.to_file(df, args.outfile, args.xml)

    if args.terrain_gsd is not None:
        transform = from_origin(
            poly.bounds[0], poly.bounds[3], args.terrain_gsd, args.terrain_gsd
        )
        window = from_bounds(*poly.bounds, transform=transform)

        tm = make_crater_field(
            df, np.zeros((math.ceil(window.height), math.ceil(window.width))), transform
        )

        with rasterio.open(
            args.outfile.with_suffix(".tif"),
            "w",
            driver="GTiff",
            height=tm.shape[0],
            width=tm.shape[1],
            count=1,
            dtype=tm.dtype,
            crs=args.proj,
            transform=transform,
        ) as dst:
            dst.write(tm, 1)

    return


if __name__ == "__main__":
    sys.exit(main())
