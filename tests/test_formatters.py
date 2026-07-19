"""Characterization tests for the pure formatting helpers.

Covers presentation.formatters plus the archive-row helpers (dur_fmt,
date_fmt, st, stt) that live with the SQLite row mapping in storage.archive.
"""

from __future__ import annotations

import unittest

from presentation import formatters as fmt
from storage.archive import date_fmt, dur_fmt, st, stt


class TestDurFmt(unittest.TestCase):
    def test_missing(self):
        self.assertEqual(dur_fmt(None), "0s")
        self.assertEqual(dur_fmt(0), "0s")

    def test_seconds_only(self):
        self.assertEqual(dur_fmt(59), "59s")

    def test_minutes(self):
        self.assertEqual(dur_fmt(60), "1m 0s")
        self.assertEqual(dur_fmt(125), "2m 5s")
        self.assertEqual(dur_fmt(3599), "59m 59s")

    def test_hours_drop_seconds(self):
        self.assertEqual(dur_fmt(3600), "1h 0m")
        self.assertEqual(dur_fmt(3725), "1h 2m")
        self.assertEqual(dur_fmt(86400), "24h 0m")


class TestDateFmt(unittest.TestCase):
    def test_missing(self):
        self.assertEqual(date_fmt(None), "N/A")
        self.assertEqual(date_fmt(""), "N/A")

    def test_yyyymmdd(self):
        self.assertEqual(date_fmt("20240115"), "2024-01-15")

    def test_passthrough(self):
        self.assertEqual(date_fmt("2024-01-15"), "2024-01-15")
        self.assertEqual(date_fmt("abc"), "abc")
        self.assertEqual(date_fmt("1234567"), "1234567")


class TestStatusLabels(unittest.TestCase):
    def test_known_labels(self):
        self.assertEqual(st("complete"), "COMPLETE")
        self.assertEqual(st("queued"), "QUEUED")
        self.assertEqual(st("processing"), "PROCESSING")
        self.assertEqual(st("failed"), "FAILED")
        self.assertEqual(st("not_started"), "NOT STARTED")
        self.assertEqual(st("downloaded"), "DOWNLOAD▰")

    def test_unknown_label_uppercased(self):
        self.assertEqual(st("weird_status"), "WEIRD_STATUS")

    def test_known_classes(self):
        self.assertEqual(stt("complete"), "ok")
        self.assertEqual(stt("queued"), "pending")
        self.assertEqual(stt("processing"), "progress")
        self.assertEqual(stt("failed"), "alert")
        self.assertEqual(stt("not_started"), "neutral")

    def test_unknown_class_neutral(self):
        self.assertEqual(stt("weird_status"), "neutral")


class TestEsc(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertEqual(fmt.esc(None), "")

    def test_escapes_markup_and_quotes(self):
        self.assertEqual(fmt.esc('<a "x">'), "&lt;a &quot;x&quot;&gt;")
        self.assertEqual(fmt.esc("a & b"), "a &amp; b")

    def test_non_string_values(self):
        self.assertEqual(fmt.esc(5), "5")


class TestFmtN(unittest.TestCase):
    def test_none_and_zero(self):
        self.assertEqual(fmt.fmt_n(None), "0")
        self.assertEqual(fmt.fmt_n(0), "0")

    def test_thousands_separator(self):
        self.assertEqual(fmt.fmt_n(1234567), "1,234,567")

    def test_numeric_string(self):
        self.assertEqual(fmt.fmt_n("42"), "42")

    def test_invalid_is_zero(self):
        self.assertEqual(fmt.fmt_n("abc"), "0")

    def test_float_truncates(self):
        self.assertEqual(fmt.fmt_n(3.9), "3")


class TestHmsToSeconds(unittest.TestCase):
    def test_hms(self):
        self.assertEqual(fmt.hms_to_seconds("1:53:14"), 6794)

    def test_ms(self):
        self.assertEqual(fmt.hms_to_seconds("12:34"), 754)
        self.assertEqual(fmt.hms_to_seconds("0:05"), 5)

    def test_seconds_only(self):
        self.assertEqual(fmt.hms_to_seconds("45"), 45)

    def test_missing(self):
        self.assertEqual(fmt.hms_to_seconds(None), 0)
        self.assertEqual(fmt.hms_to_seconds(""), 0)
        self.assertEqual(fmt.hms_to_seconds("N/A"), 0)

    def test_invalid(self):
        self.assertEqual(fmt.hms_to_seconds("ab:cd"), 0)

    def test_empty_parts_are_zero(self):
        self.assertEqual(fmt.hms_to_seconds("::"), 0)


class TestRing(unittest.TestCase):
    def test_empty_ring(self):
        self.assertEqual(fmt._ring(0, 10), "138.23")
        self.assertEqual(fmt._ring(None, 5), "138.23")
        self.assertEqual(fmt._ring(0, 0), "138.23")

    def test_full_ring(self):
        self.assertEqual(fmt._ring(10, 10), "0.00")

    def test_overfull_clamps(self):
        self.assertEqual(fmt._ring(20, 10), "0.00")

    def test_half_ring(self):
        self.assertEqual(fmt._ring(5, 10), f"{fmt.RING_C * 0.5:.2f}")


class TestDashOffset(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(fmt._dash_offset(0), "138.23")
        self.assertEqual(fmt._dash_offset(257), "0.00")

    def test_zero_total_returns_circumference(self):
        self.assertEqual(fmt._dash_offset(5, 0), "138.23")


class TestCleanSummary(unittest.TestCase):
    def test_strips_think_tags(self):
        self.assertEqual(fmt._clean_summary("<think>hidden</think>visible"), "visible")
        self.assertEqual(fmt._clean_summary("<THINK>x\ny</THINK>ok"), "ok")

    def test_think_only_is_empty(self):
        self.assertEqual(fmt._clean_summary("<think>only</think>"), "")

    def test_escapes_and_preserves_breaks(self):
        self.assertEqual(fmt._clean_summary("a\nb & <b>c</b>"), "a<br>b &amp; &lt;b&gt;c&lt;/b&gt;")

    def test_missing(self):
        self.assertEqual(fmt._clean_summary(None), "")
        self.assertEqual(fmt._clean_summary("  padded  "), "padded")


class TestCleanVid(unittest.TestCase):
    def test_strips_prefix(self):
        self.assertEqual(fmt._clean_vid("youtube_id:abc123"), "abc123")

    def test_plain_id(self):
        self.assertEqual(fmt._clean_vid("abc123"), "abc123")

    def test_missing(self):
        self.assertEqual(fmt._clean_vid(None), "N/A")
        self.assertEqual(fmt._clean_vid(""), "N/A")
        self.assertEqual(fmt._clean_vid("youtube_id:"), "N/A")

    def test_strips_whitespace(self):
        self.assertEqual(fmt._clean_vid("  x  "), "x")


class TestSigClass(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(fmt._sig_class(70), "sig-hi")
        self.assertEqual(fmt._sig_class(70.5), "sig-hi")
        self.assertEqual(fmt._sig_class(69.9), "sig-md")
        self.assertEqual(fmt._sig_class(60), "sig-md")
        self.assertEqual(fmt._sig_class(59.9), "sig-lo")
        self.assertEqual(fmt._sig_class(0), "sig-lo")


if __name__ == "__main__":
    unittest.main()
