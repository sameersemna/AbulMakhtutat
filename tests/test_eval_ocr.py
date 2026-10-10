"""Unit tests for the probe evaluation metrics in eval_ocr.py.

Run:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_ocr import _cer, _edit_distance, _wer, score_pair  # noqa: E402


class TestEditDistance(unittest.TestCase):
    def test_known_distance(self) -> None:
        self.assertEqual(_edit_distance("kitten", "sitting"), 3)

    def test_identical(self) -> None:
        self.assertEqual(_edit_distance("abc", "abc"), 0)

    def test_empty(self) -> None:
        self.assertEqual(_edit_distance("", "abc"), 3)
        self.assertEqual(_edit_distance("abc", ""), 3)


class TestRates(unittest.TestCase):
    def test_perfect(self) -> None:
        self.assertEqual(_cer("abc", "abc"), 0.0)
        self.assertEqual(_wer("a b c", "a b c"), 0.0)

    def test_all_wrong(self) -> None:
        self.assertEqual(_cer("abc", "xyz"), 1.0)
        self.assertEqual(_wer("a b", "x y"), 1.0)

    def test_empty_reference(self) -> None:
        self.assertEqual(_cer("", ""), 0.0)
        self.assertEqual(_cer("", "abc"), 1.0)


class TestScorePair(unittest.TestCase):
    def test_diacritics_only_difference(self) -> None:
        m = score_pair("كتاب", "كِتَاب")
        self.assertEqual(m["cer_normalized"], 0.0)
        self.assertGreater(m["cer_verbatim"], 0.0)

    def test_digit_forms_equal_when_normalized(self) -> None:
        m = score_pair("١٢٣", "123")
        self.assertEqual(m["cer_normalized"], 0.0)
        self.assertEqual(m["cer_verbatim"], 1.0)

    def test_identical(self) -> None:
        m = score_pair("نص عربي", "نص عربي")
        self.assertEqual(m["cer_normalized"], 0.0)
        self.assertEqual(m["wer_normalized"], 0.0)


if __name__ == "__main__":
    unittest.main()
