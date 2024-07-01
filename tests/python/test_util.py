# -*- coding: utf-8 -*-
"""This module has tests for the util module."""

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
import unittest
from unittest.mock import call, patch

import synthterrain.util as util


class TestUtil(unittest.TestCase):
    def test_FileArgumentParser(self):
        p = util.FileArgumentParser()

        self.assertEqual(
            p.convert_arg_line_to_args("# Comment, should be ignored"),
            list(),
        )

        self.assertEqual(
            p.convert_arg_line_to_args("should be split"),
            ["should", "be", "split"],
        )

    @patch("synthterrain.util.sys.exit")
    @patch("builtins.print")
    def test_PrintDictAction(self, m_print, m_exit):
        a = util.PrintDictAction(
            "--dummy", "dummy", dict={"a": "a value", "b": "b value"}
        )

        a("dummy", "dummy", "dummy")
        self.assertEqual(
            m_print.call_args_list,
            [call("a"), call("   a value"), call("b"), call("   b value")],
        )
        m_exit.assert_called_once()

    def test_parent_parser(self):
        self.assertIsInstance(util.parent_parser(), argparse.ArgumentParser)

    def test_logging(self):
        util.set_logger(verblvl=2)
        logger = logging.getLogger()
        self.assertEqual(30, logger.getEffectiveLevel())
