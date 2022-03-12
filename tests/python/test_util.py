#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module has tests for the util module."""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

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
            [
                call("a"),
                call("   a value"),
                call("b"),
                call("   b value")
            ]
        )
        m_exit.assert_called_once()

    def test_parent_parser(self):
        self.assertIsInstance(util.parent_parser(), argparse.ArgumentParser)

    def test_logging(self):
        util.set_logger(verblvl=2)
        logger = logging.getLogger()
        self.assertEqual(30, logger.getEffectiveLevel())
