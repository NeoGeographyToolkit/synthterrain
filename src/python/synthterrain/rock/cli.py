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
from pathlib import Path
import sys

from shapely.geometry import box

import synthterrain.util as util

logger = logging.getLogger(__name__)


def arg_parser():
    parser = util.FileArgumentParser(
        description=__doc__,
        parents=[util.parent_parser()],
    )
    parser.add_argument(
        "-c", "--crater",
        type=Path,
        help="Crater file."
    )
    parser.add_argument(
        "-x", "--xml",
        action="store_true",
        help="Default output is in CSV format, but if given this will result "
             "in XML output that conforms to the old MATLAB code."
    )
    parser.add_argument(
        "-o", "--outfile",
        required=True,
        type=Path,
        help="Path to output file."
    )
    return parser


def main():
    args = arg_parser().parse_args()

    util.set_logger(args.verbose)

    # This could more generically take an arbitrary polygon
    # What if the polygon is different than the one in the crater file?
    # Probably need to figure out how to standardize and robustify.
    canvas = box(*args.bbox)

    # Do generic rock distro across bbox.

    # Do intracrater rock distro if crater details are provided.

    # Write out results.
    # rock.to_file(df, args.outfile, args.xml)

    return


if __name__ == "__main__":
    sys.exit(main())
