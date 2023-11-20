#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generates plots from .csv files.
"""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

import synthterrain.crater as crater
import synthterrain.util as util


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

    return


if __name__ == "__main__":
    sys.exit(main())
