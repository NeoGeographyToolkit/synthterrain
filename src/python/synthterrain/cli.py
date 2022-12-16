#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generates synthetic crater and rock populations.
"""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

import logging
from pathlib import Path
import sys

import pandas as pd
from shapely.geometry import box

import synthterrain.crater as cr
from synthterrain.crater import functions as cr_dist
from synthterrain.crater.cli import csfd_dict
import synthterrain.rock as rk
from synthterrain.rock import functions as rk_func
import synthterrain.util as util


logger = logging.getLogger(__name__)


def arg_parser():
    parser = util.FileArgumentParser(
        description=__doc__,
        parents=[util.parent_parser()],
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=[0, 1000, 1000, 0],
        metavar=('MINX', 'MAXY', 'MAXX', 'MINY'),
        help="The coordinates of the bounding box, expressed in meters, to "
             "evaluate in min-x, max-y, max-x, min-y order (which is ulx, "
             "uly, lrx, lry, the GDAL pattern). "
             "Default: %(default)s"
    )
    parser.add_argument(
        "-c", "--craters",
        type=Path,
        help="Crater CSV or XML file of pre-existing craters.  This option is usually "
             "used as follows: A set of 'real' craters are identified from a target "
             "area above a certain diameter (say 5 m/pixel) and given to this option. "
             "Then --cr-mind and --cr_maxd are set to some range less than 5 m/pixel. "
             "This generates synthetic craters in the specified range, and then uses "
             "those synthetic craters in addition to the craters from --craters when "
             "building the rock probability map."
    )
    parser.add_argument(
        "--csfd",
        default="VIPER_Env_Spec",
        choices=csfd_dict.keys(),
        help="The name of the crater size-frequency distribution to use. "
             f"Options are: {', '.join(csfd_dict.keys())}. "
             "Default: %(default)s"
    )
    parser.add_argument(
        "--cr_maxd",
        default=1000,
        type=float,
        help="Maximum crater diameter in meters. Default: %(default)s"
    )
    parser.add_argument(
        "--cr_mind",
        default=1,
        type=float,
        help="Minimum crater diameter in meters. Default: %(default)s"
    )
    parser.add_argument(
        "--rk_maxd",
        default=2,
        type=float,
        help="Maximum crater diameter in meters. Default: %(default)s"
    )
    parser.add_argument(
        "--rk_mind",
        default=0.1,
        type=float,
        help="Minimum crater diameter in meters. Default: %(default)s"
    )
    parser.add_argument(
        "-p", "--plot",
        action="store_true",
        help="This will cause a matplotlib windows to open with some summary "
             "plots after the program has generated craters and then rocks."
    )
    parser.add_argument(
        "-x", "--xml",
        action="store_true",
        help="Default output is in CSV format, but if given this will result "
             "in XML output that conforms to the old MATLAB code."
    )
    parser.add_argument(
        "--probability_map_gsd",
        type=float,
        default=1,
        help="This program builds a probability map to generate locations, and this "
             "sets the ground sample distance in the units of --bbox for that map.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--csfd_info",
        action=util.PrintDictAction,
        dict=csfd_dict,
        help="If given, will list detailed information about each of the "
             "available CSFDs and exit."
    )
    group.add_argument(
        "--cr_outfile",
        type=Path,
        help="Path to crater output file."
    )
    parser.add_argument(
        "--rk_outfile",
        type=Path,
        required=True,
        help="Path to crater output file."
    )
    return parser


def main():
    args = arg_parser().parse_args()

    util.set_logger(args.verbose)

    poly = box(args.bbox[0], args.bbox[3], args.bbox[2], args.bbox[1])

    logger.info("Synthesizing Craters.")
    crater_dist = getattr(cr_dist, args.csfd)(a=args.cr_mind, b=args.cr_maxd)

    crater_df = cr.synthesize(
        crater_dist, polygon=poly, min_d=args.cr_mind, max_d=args.cr_maxd
    )
    cr.to_file(crater_df, args.cr_outfile, args.xml)
    if args.plot:
        cr.plot(crater_df)

    if args.craters is not None:
        precraters = cr.from_file(args.craters)
        logger.info(f"Adding {precraters.shape[0]} craters from {args.craters}")
        crater_df = pd.concat([crater_df, precraters], ignore_index=True)

    logger.info("Synthesizing Rocks.")
    df, pmap = rk.synthesize(
        rk_func.VIPER_Env_Spec(a=args.rk_mind, b=args.rk_maxd),
        polygon=poly,
        pmap_gsd=args.probability_map_gsd,
        crater_frame=crater_df,
        min_d=args.rk_mind,
        max_d=args.rk_maxd,
    )

    rk.to_file(df, args.rk_outfile, args.xml)
    if args.plot:
        rk.plot(
            df, pmap, [poly.bounds[0], poly.bounds[2], poly.bounds[1], poly.bounds[3]]
        )

    return


if __name__ == "__main__":
    sys.exit(main())
