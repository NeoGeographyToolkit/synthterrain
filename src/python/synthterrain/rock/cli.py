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
#from re import T
import matplotlib.pyplot as plt
import sys

# TODO: Clean up includes
#from matplotlib.pyplot import figure

#from shapely.geometry import box

import synthterrain.util as util
import synthterrain.crater as crater

from synthterrain.rock import rocks
from synthterrain.rock import inter_crater_rocks
from synthterrain.rock import intra_crater_rocks

logger = logging.getLogger(__name__)


def arg_parser():
    parser = util.FileArgumentParser(
        description=__doc__,
        parents=[util.parent_parser()],
    )
    parser.add_argument(
        "-c", "--crater",
        type=Path,
        help="Crater file.  If provided, generate intra-crater rocks"
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

    return parser


def main():
    args = arg_parser().parse_args()

    util.set_logger(args.verbose)

    # This could more generically take an arbitrary polygon
    # What if the polygon is different than the one in the crater file?
    # Probably need to figure out how to standardize and robustify.
    #canvas = box(*args.bbox)

    # Do generic rock distro across bbox.
    print('Constructing terrain...')
    raster = rocks.Raster(origin = (args.bbox[0], args.bbox[3]),
                          height = args.bbox[1] - args.bbox[3],
                          width = args.bbox[2] - args.bbox[0],
                          resolution_meters = 1.0)

    #print('CRATERS')
    # Deprecated craters class
    #crater_file = '/usr/local/home/smcmich1/repo/synthterrain/craters_short.xml'
    #c = craters.Craters(t)
    ##c.readExistingCraterFile(crater_file)
    ##c.INPUT_CRATER_FILE = crater_file
    #c.OUTPUT_FILE = '/usr/local/home/smcmich1/repo/synthterrain/craters_short_copy.xml'
    #c.generate()

    #TODO: configure both types

    # New craters class
    if args.crater:
        loaded_craters = crater.from_file(str(args.crater))
        print('Input crater info:')
        print(loaded_craters)
        print('Constructing IntraCraterRocks')
        intra = intra_crater_rocks.IntraCraterRocks(raster)
        print('Generating rocks...')
        intra.generate(loaded_craters)
    else:
        print('Constructing InterCraterRocks')
        inter = inter_crater_rocks.InterCraterRocks(raster)
        print('Generating rocks...')
        inter.generate()

    plt.show()
    return


if __name__ == "__main__":
    sys.exit(main())
