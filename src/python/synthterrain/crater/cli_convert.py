#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Converts between the crater CSV and XML formats.
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
        "infile",
        type=Path,
        help="A CSV or XML file produced by synthcraters."
    )
    parser.add_argument(
        "outfile",
        type=Path,
        help="The output file an XML or CSV file (whatever the opposite of the "
             "first argument is)."
    )
    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()

    util.set_logger(args.verbose)

    if args.infile.suffix.casefold() == ".csv":
        df = pd.read_csv(args.infile)
        crater.to_file(df, args.outfile, xml=True)

    elif args.infile.suffix.casefold() == ".xml":
        df = pd.read_xml(args.infile, parser="etree", xpath=".//CraterData")

        # Create the columns that the CSV output is expecting.
        df["diameter"] = df["rimRadius"] * 2
        df["d/D"] = df["freshness"] * 0.2731
        df["age"] = 0  # No age information from XML file format.

        crater.to_file(df, args.outfile, xml=False)
    else:
        parser.error(
            f"The input file {args.infile} did not end in .csv or .xml."
        )

    return


if __name__ == "__main__":
    sys.exit(main())
