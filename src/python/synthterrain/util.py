"""This module contains viss utility functions."""

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
import textwrap

import synthterrain


class FileArgumentParser(argparse.ArgumentParser):
    """This argument parser sets the fromfile_prefix_chars to the
    at-symbol (@), treats lines that begin with the octothorpe (#)
    as comments, and allows multiple argument elements per line.
    """

    def __init__(self, *args, **kwargs):
        kwargs["fromfile_prefix_chars"] = "@"

        fileinfo = textwrap.dedent(
            """\
            In addition to regular command-line arguments, this program also
            accepts filenames prepended with the at-symbol (@) which can
            contain command line arguments (lines that begin with # are
            ignored, multiple lines are allowed) as you'd type them so you
            can keep often-used arguments in a handy file.
            """
        )

        if "description" in kwargs:
            kwargs["description"] += " " + fileinfo
        else:
            kwargs["description"] = fileinfo

        super().__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, arg_line):
        if arg_line.startswith("#"):
            return []

        return arg_line.split()


class PrintDictAction(argparse.Action):
    """A custom action that interrupts argument processing, prints
    the contents of the *dict* argument, and then exits the
    program.

    It may need to be placed in a mutually exclusive argument
    group (see argparse documentation) with any required arguments
    that your program should have.
    """

    def __init__(self, *args, dict=None, **kwargs):
        kwargs["nargs"] = 0
        super().__init__(*args, **kwargs)
        self.dict = dict

    def __call__(self, parser, namespace, values, option_string=None):
        for k, v in self.dict.items():
            print(k)
            if v.startswith("\n"):
                docstring = v[1:]
            else:
                docstring = v
            print(textwrap.indent(textwrap.dedent(docstring), "   "))
        sys.exit()


def parent_parser() -> argparse.ArgumentParser:
    """Returns a parent parser with common arguments."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Displays additional information.",
    )
    parent.add_argument(
        "--version",
        action="version",
        version=f"synthterrain Software version {synthterrain.__version__}",
        help="Show library version number.",
    )
    return parent


def set_logger(verblvl=None) -> None:
    """Sets the log level and configuration for applications."""
    logger = logging.getLogger(__name__.split(".", maxsplit=1)[0])
    lvl_dict = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    if verblvl in lvl_dict:
        lvl = lvl_dict[verblvl]
    else:
        lvl = lvl_dict[max(lvl_dict.keys())]

    logger.setLevel(lvl)

    ch = logging.StreamHandler()
    ch.setLevel(lvl)

    if lvl < 20:  # less than INFO
        formatter = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
    else:
        formatter = logging.Formatter("%(levelname)s: %(message)s")

    ch.setFormatter(formatter)
    logger.addHandler(ch)
