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
import sys

from shapely.geometry import box

import synthterrain.crater as cr
from synthterrain.crater import distributions as cr_dist
from synthterrain.crater.cli import arg_parser as cr_arg_parser
import synthterrain.rock as rk
from synthterrain.rock.cli import arg_parser as rk_arg_parser
import synthterrain.util as util


logger = logging.getLogger(__name__)


def arg_parser():
    parser = util.FileArgumentParser(
        description=__doc__,
        parents=[util.parent_parser(), cr_arg_parser(), rk_arg_parser()],
        conflict_handler="resolve",
    )
    return parser


def main():
    args = arg_parser().parse_args()

    util.set_logger(args.verbose)

    # This could more generically take an arbitrary polygon
    canvas = box(*args.bbox)

    crater_dist = getattr(cr_dist, args.csfd)(a=args.mind, b=args.maxd)

    crater_df = cr.synthesize(crater_dist, polygon=canvas)
    cr.to_file(crater_df, args.outfile, args.xml)

    # Maybe something like this, to be revised once the rock modules are
    # fleshed out.
    rock_df = rk.synthesize()
    rk.to_file(rock_df, craters=crater_df)

    return


if __name__ == "__main__":
    sys.exit(main())
