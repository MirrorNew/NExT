import unittest

from rebuttal_runner.scoring import AnswerParseError, parse_answer, score_answer


class ParseAnswerTests(unittest.TestCase):
    def test_accepts_supported_payloads_and_wrappers(self):
        self.assertEqual(parse_answer("FinalAnswer=【12.5】"), 12.5)
        self.assertEqual(parse_answer("FinalAnswer=<[1, 2, {\"x\": 3}]>"), [1, 2, {"x": 3}])
        self.assertEqual(parse_answer("log\nFinalAnswer={\"x\": [2]}\n"), {"x": [2]})

    def test_rejects_missing_malformed_and_multiple_markers(self):
        cases = [
            "no answer",
            "FinalAnswer = 2",  # JSON must follow the equals sign.
            "prefix FinalAnswer=2",
            "FinalAnswer=2\nFinalAnswer=3",
            "FinalAnswer=2\ntext FinalAnswer=3",
            "FinalAnswer=2 trailing",
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(AnswerParseError):
                parse_answer(text)

    def test_rejects_non_numeric_or_non_finite_json(self):
        for text in (
            "FinalAnswer=true",
            "FinalAnswer=null",
            "FinalAnswer=\"3\"",
            "FinalAnswer=[]",
            "FinalAnswer={}",
            'FinalAnswer={"x": 1, "x": 2}',
            "FinalAnswer=[1, false]",
            "FinalAnswer={\"x\": \"3\"}",
            "FinalAnswer=NaN",
            "FinalAnswer=Infinity",
            "FinalAnswer=1e999",
        ):
            with self.subTest(text=text), self.assertRaises(AnswerParseError):
                parse_answer(text)


class ScoreAnswerTests(unittest.TestCase):
    def test_nextor_relative_sign_zero_and_inclusive_boundary(self):
        protocol = {"name": "nextor_relative", "tolerance": 0.01}
        self.assertTrue(score_answer(101, 100, protocol)["correct"])
        self.assertTrue(score_answer(99, 100, protocol)["correct"])
        self.assertTrue(score_answer(0, 0, protocol)["correct"])
        zero_error = score_answer(1, 0, protocol)["errors"][0]
        self.assertIsNone(zero_error["relative_error"])
        self.assertEqual(zero_error["absolute_error"], 1)

    def test_absolute_and_normalized_protocols_use_strict_boundary(self):
        self.assertFalse(score_answer(1.0001, 1, {"name": "optibench_absolute", "tolerance": 0.0001})["correct"])
        self.assertTrue(score_answer(1.00009, 1, {"name": "optibench_absolute", "tolerance": 0.0001})["correct"])
        self.assertFalse(score_answer(1.000002, 1, {"name": "optmath_normalized", "tolerance": 0.000001})["correct"])

    def test_zero_reference_and_default_tolerances(self):
        self.assertTrue(score_answer(0, 0, {"name": "nextor_relative"})["correct"])
        self.assertFalse(score_answer(1e-12, 0, {"name": "nextor_relative"})["correct"])
        self.assertTrue(score_answer(0, 0, {"name": "optibench_absolute"})["correct"])
        self.assertTrue(score_answer(1e-7, 0, {"name": "optmath_normalized"})["correct"])

    def test_lists_require_same_length_order_and_complete_itemwise_match(self):
        protocol = {"name": "exact"}
        self.assertFalse(score_answer([1], [1, 2], protocol)["correct"])
        self.assertFalse(score_answer([1, 2], [2, 1], protocol)["correct"])
        self.assertFalse(score_answer([1, 2.1], [1, 2], {"name": "optibench_absolute"})["correct"])
        self.assertFalse(score_answer(1, [1], protocol)["correct"])

    def test_dict_keys_must_match_and_numeric_paths_are_reported(self):
        result = score_answer({"a": 2, "extra": 4}, {"a": 1, "required": 3}, {"name": "exact"})
        self.assertFalse(result["correct"])
        self.assertEqual({item["kind"] for item in result["errors"]}, {"unexpected_key", "missing_key", "numeric_mismatch"})
        nested = score_answer({"a": [1, 3]}, {"a": [1, 2]}, {"name": "exact"})
        self.assertEqual(nested["errors"][0]["path"], '$["a"][1]')
        self.assertEqual(nested["errors"][0]["absolute_error"], 1)
        self.assertEqual(nested["errors"][0]["relative_error"], 0.5)

    def test_bool_is_not_a_number_and_exact_is_numeric_structural_equality(self):
        self.assertFalse(score_answer(True, 1, {"name": "exact"})["correct"])
        self.assertFalse(score_answer(1, True, {"name": "exact"})["correct"])
        self.assertFalse(score_answer([], [], {"name": "exact"})["correct"])
        self.assertTrue(score_answer(1, 1.0, {"name": "exact"})["correct"])
        self.assertFalse(score_answer([1], (1,), {"name": "exact"})["correct"])

    def test_aliases_are_explicit_and_unknown_names_fail(self):
        self.assertTrue(score_answer(101, 100, {"name": "nextor"})["correct"])
        self.assertTrue(score_answer(1, 1, {"name": "optibench"})["correct"])
        self.assertTrue(score_answer(1, 1, {"name": "optmath"})["correct"])
        with self.assertRaises(ValueError):
            score_answer(1, 1, {"name": "relative"})

    def test_protocol_validation(self):
        for protocol in (
            {},
            {"name": "exact", "tolerance": 0},
            {"name": "nextor_relative", "tolerance": -1},
            {"name": "nextor_relative", "tolerance": True},
            {"name": "optmath_normalized", "tolerance": float("inf")},
        ):
            with self.subTest(protocol=protocol), self.assertRaises(ValueError):
                score_answer(1, 1, protocol)


if __name__ == "__main__":
    unittest.main()
