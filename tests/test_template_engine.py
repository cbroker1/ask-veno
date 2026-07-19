"""Characterization tests for the custom placeholder template engine."""

from __future__ import annotations

import unittest

from presentation.template_engine import PAGE, SEARCH_PAGE, load_template, tmpl


class TestTmpl(unittest.TestCase):
    def test_simple_replacement(self):
        self.assertEqual(tmpl("a {{x}} b", x="1"), "a 1 b")

    def test_missing_placeholder_left_as_is(self):
        self.assertEqual(tmpl("hi {{name}}", other="x"), "hi {{name}}")

    def test_multiple_placeholders(self):
        self.assertEqual(tmpl("{{a}}{{b}}{{a}}", a="1", b=2), "121")

    def test_values_stringified(self):
        self.assertEqual(tmpl("{{n}}", n=0), "0")

    def test_conditional_true_keeps_block(self):
        self.assertEqual(tmpl("A{{c_if}}X{{/c}}B", c_if=True), "AXB")

    def test_conditional_false_drops_block(self):
        self.assertEqual(tmpl("A{{c_if}}X{{/c}}B", c_if=False), "AB")

    def test_conditional_without_close_tag_left_as_is(self):
        self.assertEqual(tmpl("A{{c_if}}X no close", c_if=True), "A{{c_if}}X no close")


class TestPageTemplate(unittest.TestCase):
    def test_page_loads(self):
        self.assertIn("<!DOCTYPE html>", PAGE)
        self.assertIn("ASK VENO", PAGE)
        self.assertGreater(len(PAGE), 30000)

    def test_page_has_render_placeholders(self):
        for key in ("led_db_cls", "led_synth_txt", "total", "chunks_fmt", "q_attr",
                    "echo_html", "report_html", "results_html", "video_rows", "archive_shown"):
            self.assertIn("{{" + key + "}}", PAGE)

    def test_search_page_alias(self):
        self.assertIs(SEARCH_PAGE, PAGE)

    def test_load_template_matches_module_constant(self):
        self.assertEqual(load_template("page.html"), PAGE)


if __name__ == "__main__":
    unittest.main()
