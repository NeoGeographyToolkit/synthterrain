# -*- coding: utf-8 -*-
"""Generates synthetic rock populations.
"""

# Copyright © 2024, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The “synthterrain” software is licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License
# at http://www.apache.org/licenses/LICENSE-2.0.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import logging
import sys
from pathlib import Path

from shapely.geometry import box

from synthterrain import crater, rock, util
from synthterrain.rock import functions

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
        metavar=("MINX", "MAXY", "MAXX", "MINY"),
        help="The coordinates of the bounding box, expressed in meters, to "
        "evaluate in min-x, max-y, max-x, min-y order (which is ulx, "
        "uly, lrx, lry, the GDAL pattern). "
        "Default: %(default)s",
    )
    parser.add_argument(
        "-c", "--craters", type=Path, help="Crater csv file from synthcraters."
    )
    parser.add_argument(
        "--maxd",
        default=2,
        type=float,
        help="Maximum rock diameter in meters. Default: %(default)s",
    )
    parser.add_argument(
        "--mind",
        default=0.1,
        type=float,
        help="Minimum rock diameter in meters. Default: %(default)s",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="This will cause a matplotlib window to open with some summary "
        "plots after the program has generated the data.",
    )
    parser.add_argument(
        "--probability_map_gsd",
        type=float,
        default=1,
        help="This program builds a probability map to generate locations, and this "
        "sets the ground sample distance in the units of --bbox for that map.",
    )
    parser.add_argument(
        "-x",
        "--xml",
        action="store_true",
        help="Default output is in CSV format, but if given this will result "
        "in XML output that conforms to the old MATLAB code.",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        required=True,
        default=None,
        type=Path,
        help="Path to output file.",
    )

    return parser


def main():
    args = arg_parser().parse_args()

    util.set_logger(args.verbose)

    # This could more generically take an arbitrary polygon
    # bbox argument takes: 'MINX', 'MAXY', 'MAXX', 'MINY'
    # the box() function takes: (minx, miny, maxx, maxy)
    poly = box(args.bbox[0], args.bbox[3], args.bbox[2], args.bbox[1])

    rock_dist = functions.VIPER_Env_Spec(a=args.mind, b=args.maxd)

    if args.craters is None:
        cf = None
    else:
        cf = crater.from_file(args.craters)

    df, pmap = rock.synthesize(
        rock_dist,
        polygon=poly,
        pmap_gsd=args.probability_map_gsd,
        crater_frame=cf,
        min_d=args.mind,
        max_d=args.maxd,
    )

    if args.plot:
        rock.plot(
            df, pmap, [poly.bounds[0], poly.bounds[2], poly.bounds[1], poly.bounds[3]]
        )

    # Write out results.
    rock.to_file(df, args.outfile, args.xml)


if __name__ == "__main__":
    sys.exit(main())
