#!/usr/bin/env python3
# Intentionally tiny sentinel so the compression-lab branch has an explicit
# test file proving this lane is research-only and separate from main.
import unittest


class CompressionLaneSentinel(unittest.TestCase):
    def test_research_lane(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
