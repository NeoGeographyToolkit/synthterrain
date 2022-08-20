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
from re import T
import matplotlib.pyplot as plt
import sys

from matplotlib.pyplot import figure

from shapely.geometry import box

import synthterrain.util as util
import synthterrain.crater as crater

#from synthterrain.rock import craters
from synthterrain.rock import terrain
from synthterrain.rock import rocks
from synthterrain.rock import inter_crater_rocks
from synthterrain.rock import intra_crater_rocks
from synthterrain.rock import utilities

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
    #canvas = box(*args.bbox)

    # Do generic rock distro across bbox.
    print('TERRAIN')
    t = terrain.Terrain()
    t.setOrigin(-660, -435) # TODO: This is actually the corner?
    t.setXsize(200) #TODO
    t.setYsize(200) #TODO
    print('TERRAIN generate')
    t.generate()

    #print('CRATERS')
    # Deprecated craters class
    #crater_file = '/usr/local/home/smcmich1/repo/synthterrain/craters_short.xml'
    #c = craters.Craters(t)
    ##c.readExistingCraterFile(crater_file)
    ##c.INPUT_CRATER_FILE = crater_file
    #c.OUTPUT_FILE = '/usr/local/home/smcmich1/repo/synthterrain/craters_short_copy.xml'
    #c.generate()

    # New craters class
    crater_file = '/usr/local/home/smcmich1/repo/synthterrain/crater_output.csv'
    c = crater.from_file(crater_file)
    print('Input crater info:')
    print(c)
    #raise Exception('DEBUG')

    print('InterCraterRocks')
    inter = inter_crater_rocks.InterCraterRocks(t)
    #print('IntraCraterRocks')
    #intra = intra_crater_rocks.IntraCraterRocks(t)

    #TODO: configure both types

    # This also writes the output file
    print('GENERATE')
    inter.generate()
    #intra.generate(c)
    #raise Exception('DEBUG')

    # TODO: Pick one or more of the existing functions
    if True:#args.plot: TODO
        print('PLOT')
        figureNumber = 1
        inter.plotLocations(figureNumber)
        #intra.plotLocations(figureNumber)
    plt.show()
    return


if __name__ == "__main__":
    sys.exit(main())
