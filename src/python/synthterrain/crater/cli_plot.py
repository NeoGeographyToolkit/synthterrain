# -*- coding: utf-8 -*-
"""Generates plots from .csv files.
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

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from synthterrain import crater, util


logger = logging.getLogger(__name__)


def arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[util.parent_parser()],
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="A CSV file with a header row, and the following columns: "
        "x, y, diameter, age, d/D.",
    )
    return parser


def main():
    args = arg_parser().parse_args()

    util.set_logger(args.verbose)

    df = pd.read_csv(args.csv)

    crater.plot(df)


if __name__ == "__main__":
    sys.exit(main())
