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
from synthterrain.crater.age import estimate_age_by_bin
import synthterrain.crater.functions as crater_func
import synthterrain.util as util


logger = logging.getLogger(__name__)


def arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[util.parent_parser()],
    )
    parser.add_argument(
        "--estimate_ages",
        action="store_true",
        help="When given, craters in the input file with no age specified, or an age "
             "of zero, will have an age estimated based on their diameter and d/D "
             "ratio using the Grun/Neukum production function and the VIPER "
             "Environmental Specification equilibrium crater function.  Some resulting "
             "craters may still yield a zero age if the d/D ratio was large relative "
             "to the diameter, indicating a very fresh crater."
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

    elif args.infile.suffix.casefold() == ".xml":
        df = pd.read_xml(args.infile, parser="etree", xpath=".//CraterData")

        # Create the columns that the CSV output is expecting.
        df["diameter"] = df["rimRadius"] * 2
        df["d/D"] = df["freshness"] * 0.2731
        df["age"] = 0  # No age information from XML file format.

    else:
        parser.error(
            f"The input file {args.infile} did not end in .csv or .xml."
        )

    if args.estimate_ages:
        a = df["diameter"].min()
        b = df["diameter"].max()
        pd_func = crater_func.GNPF(a=a, b=b)
        eq_func = crater_func.VIPER_Env_Spec(a=a, b=b)
        if "age" in df.columns:
            df[df["age"] == 0] = estimate_age_by_bin(
                df[df["age"] == 0], pd_func.csfd, eq_func.csfd, num=50
            )
        else:
            df = estimate_age_by_bin(df, pd_func.csfd, eq_func.csfd, num=50)

    crater.to_file(df, args.outfile, xml=(args.outfile.suffix.casefold() == ".xml"))

    return


if __name__ == "__main__":
    sys.exit(main())
