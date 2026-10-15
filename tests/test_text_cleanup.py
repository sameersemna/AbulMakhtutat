"""Unit tests for text_cleanup (page-number stripping + Arabic normalization).

Run:
    .venv/bin/python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure the repo root is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_cleanup import (  # noqa: E402
    HONOURIFIC_LIGATURES,
    cleanup_text,
    expand_honourifics,
    normalize_arabic,
    strip_page_numbers,
)


class TestPageCleanup(unittest.TestCase):
    def test_strip_latin_page_number(self) -> None:
        self.assertEqual(strip_page_numbers("مرحبا\n42\nالعالم"), "مرحبا\nالعالم")

    def test_strip_arabic_indic_page_number(self) -> None:
        self.assertEqual(strip_page_numbers("نص\n٦٨\nمزيد"), "نص\nمزيد")

    def test_strip_heading_wrapped_page_number(self) -> None:
        self.assertEqual(strip_page_numbers("## ٦٨"), "")

    def test_keeps_short_number_inside_text(self) -> None:
        # A line with real words is not a page number.
        self.assertEqual(strip_page_numbers("سنة 42 هـ"), "سنة 42 هـ")

    def test_cleanup_with_header(self) -> None:
        text = "# عنوان الكتاب\nنص الصفحة"
        self.assertEqual(cleanup_text(text, "عنوان الكتاب"), "نص الصفحة")


class TestNormalizeArabic(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(normalize_arabic(""), "")

    def test_lam_alef_presentation_form(self) -> None:
        # U+FEFB ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM -> لا (U+0644 U+0627)
        self.assertEqual(normalize_arabic("\uFEFB"), "\u0644\u0627")

    def test_arabic_indic_digits(self) -> None:
        self.assertEqual(normalize_arabic("\u0661\u0662\u0663"), "123")

    def test_extended_arabic_indic_digits(self) -> None:
        self.assertEqual(normalize_arabic("\u06F1\u06F2"), "12")

    def test_alef_variants(self) -> None:
        self.assertEqual(normalize_arabic("\u0623\u0625\u0622\u0671"), "\u0627\u0627\u0627\u0627")

    def test_ta_marbuta_and_ya(self) -> None:
        self.assertEqual(normalize_arabic("\u0629"), "\u0647")  # ة -> ه
        self.assertEqual(normalize_arabic("\u0649"), "\u064A")  # ى -> ي

    def test_tatweel_removed(self) -> None:
        self.assertEqual(normalize_arabic("كت\u0640اب"), "كتاب")

    def test_diacritics_removed(self) -> None:
        self.assertEqual(normalize_arabic("كِتَاب"), "كتاب")

    def test_whitespace_collapsed(self) -> None:
        self.assertEqual(normalize_arabic("a  b\n\nc"), "a b c")

    def test_mixed_bidi(self) -> None:
        self.assertEqual(normalize_arabic("مرحبا 123 hello"), "مرحبا 123 hello")

    def test_flags_disable_transforms(self) -> None:
        self.assertEqual(normalize_arabic("\u0623", normalize_letters=False), "\u0623")
        self.assertEqual(normalize_arabic("كِتَاب", strip_diacritics=False), "كِتَاب")
        self.assertEqual(normalize_arabic("\u0661", normalize_digits=False), "\u0661")

    def test_idempotent(self) -> None:
        once = normalize_arabic("أَلَا إِنَّ ١٢٣")
        self.assertEqual(normalize_arabic(once), once)


class TestHonourificLigatures(unittest.TestCase):
    def test_ligature_expanded_to_phrase(self) -> None:
        # A non-NFKC honourific ligature (corpus mapping; see HONOURIFIC_LIGATURES).
        self.assertEqual(normalize_arabic("\uFD42"), "رضي الله عنها")

    def test_nfkc_only_ligature_expanded(self) -> None:
        # U+FDFA is decomposed by NFKC too; the result must be the phrase,
        # with the letter map applied (ى -> ي).
        self.assertEqual(normalize_arabic("\uFDFA"), "صلي الله عليه وسلم")

    def test_ligature_inside_sentence(self) -> None:
        self.assertEqual(normalize_arabic("قال \uFDFA ثم"), "قال صلي الله عليه وسلم ثم")

    def test_ornate_parentheses_kept(self) -> None:
        self.assertEqual(normalize_arabic("﴿نص﴾"), "﴿نص﴾")

    def test_expand_honourifics_idempotent(self) -> None:
        once = expand_honourifics("\uFD43")
        self.assertEqual(expand_honourifics(once), once)
        self.assertEqual(once, "رضي الله عنهم")

    def test_expand_honourifics_no_ligatures(self) -> None:
        self.assertEqual(expand_honourifics("نص عادي"), "نص عادي")

    def test_flag_disables_expansion(self) -> None:
        # Without expansion, a non-NFKC ligature survives as its presentation form.
        self.assertEqual(normalize_arabic("\uFD42", expand_ligatures=False), "\uFD42")

    def test_all_mapped_ligatures_expanded(self) -> None:
        for ch, phrase in HONOURIFIC_LIGATURES.items():
            self.assertEqual(expand_honourifics(ch), phrase)
            self.assertNotIn(ch, normalize_arabic(ch))


if __name__ == "__main__":
    unittest.main()
