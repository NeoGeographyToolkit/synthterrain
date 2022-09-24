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
import matplotlib.pyplot as plt
import sys

import synthterrain.util as util
import synthterrain.crater as crater

from synthterrain.rock import rocks

logger = logging.getLogger(__name__)


def arg_parser():
    parser = util.FileArgumentParser(
        description=__doc__,
        parents=[util.parent_parser()],
    )
    parser.add_argument(
        "-c", "--crater",
        type=Path,
        required=True,
        help="Crater file"
    )
    parser.add_argument(
        "-x", "--xml",
        action="store_true",
        help="Default output is in CSV format, but if given this will result "
             "in XML output that conforms to the old MATLAB code."
    )
    parser.add_argument(
        "-o", "--outfile",
        default=None,
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
    RESOLUTION = 0.8
    raster = rocks.Raster(origin = (args.bbox[0], args.bbox[3]),
                          nrows = (args.bbox[1] - args.bbox[3]) / RESOLUTION,
                          ncols = (args.bbox[2] - args.bbox[0]) / RESOLUTION,
                          resolution_meters = RESOLUTION)

    rock_parameters = rocks.RockParams()

    # New craters class
    
    loaded_craters = crater.from_file(str(args.crater))
    print('Input crater info:')
    print(loaded_craters)
    print('Constructing')
    generator = rocks.RockGenerator(raster, loaded_craters, rock_parameters)

    print('Generating rocks...')
    generator.generate()

    generator.plotDensityDistribution(1)
    generator.plotDiameterDistributions(2)
    generator.plotLocationProbabilityMap(3)
    generator.plotLocations(4)

    if args.outfile:
        generator.writeXml(args.outfile)

    plt.show()
    return


if __name__ == "__main__":
    sys.exit(main())
