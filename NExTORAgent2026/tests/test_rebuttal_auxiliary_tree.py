import json
import unittest

from rebuttal_runner.auxiliary_tree import (
    AUXILIARY_SCHEMA_VERSION,
    parse_auxiliary_response,
)


MATH_INPUT = {
    "Parameters_List": [],
    "Variables_List": [{"symbol": "x"}, {"symbol": "z"}],
    "Constraint_Table": [{"name": "limit", "Mathematical_expressions": "x + z <= 10"}],
    "Objective": {"Mathematical_expressions": "Minimize f = x**2 + z"},
    "Problem_Type": "NLP",
}


def response(candidates):
    return json.dumps({
        "math_model_advice": [{"variables": [], "constraints": [], "objective": "use t"}],
        "decomposition": {"schema_version": AUXILIARY_SCHEMA_VERSION, "candidates": candidates},
    })


class AuxiliaryTreeTests(unittest.TestCase):
    def test_structured_auxiliary_output_is_ordered_and_bound_to_sources(self):
        content = response([
            {
                "source_id": "objective",
                "source_expression": "Minimize f = x**2 + z",
                "relation": "objective",
                "linearized_form": "t + z",
                "definitions": [{"symbol": "t", "rhs": "x**2", "domain_conditions": []}],
                "domain_conditions": [],
            },
            {
                "source_id": "constraint:0",
                "source_expression": "x + z <= 10",
                "relation": "<=",
                "linearized_form": "x + z - 10",
                "definitions": [],
                "domain_conditions": [],
            },
        ])
        tree = parse_auxiliary_response(content, MATH_INPUT)
        self.assertEqual(tree["status"], "validated")
        self.assertEqual(tree["candidates"][0]["definitions"][0]["dependencies"], ["x"])
        self.assertEqual(tree["candidates"][0]["equivalence_status"], "not_proven")

    def test_missing_structured_decomposition_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "refusing raw-expression redecomposition"):
            parse_auxiliary_response(json.dumps({"math_model_advice": []}), MATH_INPUT)

    def test_malformed_legacy_advice_is_rejected_with_valid_tree_schema(self):
        payload = json.loads(response([]))
        payload["math_model_advice"] = ["unstructured advice"]
        with self.assertRaisesRegex(ValueError, r"math_model_advice\[0\] has an invalid schema"):
            parse_auxiliary_response(json.dumps(payload), MATH_INPUT)

    def test_source_mismatch_is_rejected(self):
        candidate = {
            "source_id": "objective", "source_expression": "Minimize f = x + 1",
            "relation": "objective", "linearized_form": "x", "definitions": [],
            "domain_conditions": [],
        }
        with self.assertRaisesRegex(ValueError, "differs from the frozen input"):
            parse_auxiliary_response(response([candidate]), MATH_INPUT)

    def test_forward_or_undeclared_definition_dependency_is_rejected(self):
        candidate = {
            "source_id": "objective", "source_expression": "Minimize f = x**2 + z",
            "relation": "objective", "linearized_form": "t + z",
            "definitions": [{"symbol": "t", "rhs": "u + x", "domain_conditions": []}],
            "domain_conditions": [],
        }
        with self.assertRaisesRegex(ValueError, "forward or undeclared dependencies"):
            parse_auxiliary_response(response([candidate]), MATH_INPUT)

    def test_declared_array_base_allows_constant_index_without_evaluation(self):
        math_input = {
            **MATH_INPUT,
            "Variables_List": [{"symbol": "x"}],
            "Objective": {"Mathematical_expressions": "Maximize x[0]"},
        }
        candidate = {
            "source_id": "objective", "source_expression": "Maximize x[0]",
            "relation": "objective", "linearized_form": "t",
            "definitions": [{"symbol": "t", "rhs": "x[0]", "domain_conditions": []}],
            "domain_conditions": [],
        }
        tree = parse_auxiliary_response(response([candidate]), math_input)
        self.assertEqual(tree["candidates"][0]["definitions"][0]["dependencies"], ["x[0]"])

    def test_exact_index_declaration_does_not_authorize_other_indices(self):
        math_input = {
            **MATH_INPUT,
            "Variables_List": [{"symbol": "x[0]"}],
            "Objective": {"Mathematical_expressions": "Maximize x[9]"},
        }
        candidate = {
            "source_id": "objective", "source_expression": "Maximize x[9]",
            "relation": "objective", "linearized_form": "x[9]", "definitions": [],
            "domain_conditions": [],
        }
        with self.assertRaisesRegex(ValueError, "undeclared dependencies"):
            parse_auxiliary_response(response([candidate]), math_input)

    def test_parameter_index_is_checked_against_its_declaration(self):
        math_input = {
            **MATH_INPUT,
            "Parameters_List": [{"Name": "a[0]"}],
            "Objective": {"Mathematical_expressions": "Minimize f = a[0] + x"},
        }
        candidate = {
            "source_id": "objective", "source_expression": "Minimize f = a[0] + x",
            "relation": "objective", "linearized_form": "a[1] + x", "definitions": [],
            "domain_conditions": [],
        }
        with self.assertRaisesRegex(ValueError, "undeclared dependencies"):
            parse_auxiliary_response(response([candidate]), math_input)

    def test_function_names_are_only_exempt_when_called(self):
        candidate = {
            "source_id": "objective", "source_expression": "Minimize f = x**2 + z",
            "relation": "objective", "linearized_form": "log + z", "definitions": [],
            "domain_conditions": [],
        }
        with self.assertRaisesRegex(ValueError, "undeclared dependencies"):
            parse_auxiliary_response(response([candidate]), MATH_INPUT)
        candidate["linearized_form"] = "log(x) + z"
        tree = parse_auxiliary_response(response([candidate]), MATH_INPUT)
        self.assertEqual(tree["status"], "validated")

    def test_ln_is_accepted_as_natural_log_and_source_text_is_preserved(self):
        math_input = {
            **MATH_INPUT,
            "Constraint_Table": [{"name": "log_limit", "Mathematical_expressions": "ln(x) >= ln(2)"}],
        }
        candidate = {
            "source_id": "constraint:0", "source_expression": "ln(x) >= ln(2)",
            "relation": ">=", "linearized_form": "y_ln - ln(2)",
            "definitions": [{"symbol": "y_ln", "rhs": "ln(x)", "domain_conditions": ["x > 0"]}],
            "domain_conditions": ["x > 0"],
        }
        tree = parse_auxiliary_response(response([candidate]), math_input)
        self.assertEqual(tree["status"], "validated")
        self.assertEqual(tree["candidates"][0]["source_expression"], "ln(x) >= ln(2)")
        self.assertEqual(tree["candidates"][0]["definitions"][0]["rhs"], "ln(x)")

    def test_log2_log10_are_supported_with_single_argument_and_ascii_symbols(self):
        math_input = {
            **MATH_INPUT,
            "Constraint_Table": [{"name": "log_limit", "Mathematical_expressions": "log2(x) + log10(z) >= 1"}],
        }
        candidate = {
            "source_id": "constraint:0",
            "source_expression": "log2(x) + log10(z) >= 1",
            "relation": ">=",
            "linearized_form": "u + v - 1",
            "definitions": [
                {"symbol": "u", "rhs": "log2(x)", "domain_conditions": ["x > 0"]},
                {"symbol": "v", "rhs": "log10(z)", "domain_conditions": ["z > 0"]},
            ],
            "domain_conditions": ["x > 0", "z > 0"],
        }
        tree = parse_auxiliary_response(response([candidate]), math_input)
        self.assertEqual(tree["status"], "validated")
        self.assertEqual(tree["candidates"][0]["definitions"][0]["dependencies"], ["x"])

        candidate["definitions"][0]["rhs"] = "log2(x, z)"
        with self.assertRaisesRegex(ValueError, "exactly one argument"):
            parse_auxiliary_response(response([candidate]), math_input)

    def test_constraint_rows_and_domain_conditions_must_be_atomic(self):
        chained_domain = {
            "source_id": "objective", "source_expression": "Minimize f = x**2 + z",
            "relation": "objective", "linearized_form": "x + z", "definitions": [],
            "domain_conditions": ["x >= 0 <= z"],
        }
        with self.assertRaisesRegex(ValueError, "simple arithmetic relation"):
            parse_auxiliary_response(response([chained_domain]), MATH_INPUT)

        compound_input = {
            **MATH_INPUT,
            "Constraint_Table": [{"name": "combined", "Mathematical_expressions": "x + z <= 10, z >= 0"}],
        }
        compound = {
            "source_id": "constraint:0", "source_expression": "x + z <= 10, z >= 0",
            "relation": "<=", "linearized_form": "x + z - 10", "definitions": [],
            "domain_conditions": [],
        }
        with self.assertRaisesRegex(ValueError, "unsupported chained relation"):
            parse_auxiliary_response(response([compound]), compound_input)

    def test_unicode_identifier_and_implicit_multiplication_are_rejected(self):
        unicode_input = {
            **MATH_INPUT,
            "Variables_List": [{"symbol": "x₁"}],
            "Objective": {"Mathematical_expressions": "Maximize x₁"},
        }
        unicode_candidate = {
            "source_id": "objective", "source_expression": "Maximize x₁",
            "relation": "objective", "linearized_form": "x₁", "definitions": [],
            "domain_conditions": [],
        }
        with self.assertRaises(ValueError):
            parse_auxiliary_response(response([unicode_candidate]), unicode_input)

        implicit = {
            "source_id": "objective", "source_expression": "Minimize f = x**2 + z",
            "relation": "objective", "linearized_form": "0.5 x + z", "definitions": [],
            "domain_conditions": [],
        }
        with self.assertRaises(ValueError):
            parse_auxiliary_response(response([implicit]), MATH_INPUT)

    def test_auxiliary_definition_not_used_in_relation_is_rejected(self):
        math_input = {
            **MATH_INPUT,
            "Constraint_Table": [{"name": "log_limit", "Mathematical_expressions": "ln(x) >= ln(2)"}],
        }
        candidate = {
            "source_id": "constraint:0", "source_expression": "ln(x) >= ln(2)",
            "relation": ">=", "linearized_form": "ln(x) - ln(2)",
            "definitions": [{"symbol": "y_ln", "rhs": "ln(x)", "domain_conditions": ["x > 0"]}],
            "domain_conditions": [],
        }
        with self.assertRaisesRegex(ValueError, "does not use any declared auxiliary definition"):
            parse_auxiliary_response(response([candidate]), math_input)

    def test_definition_domain_may_bound_its_own_auxiliary_symbol(self):
        candidate = {
            "source_id": "objective", "source_expression": "Minimize f = x**2 + z",
            "relation": "objective", "linearized_form": "t + z",
            "definitions": [{"symbol": "t", "rhs": "x**2", "domain_conditions": ["t >= 0"]}],
            "domain_conditions": [],
        }
        tree = parse_auxiliary_response(response([candidate]), MATH_INPUT)
        self.assertEqual(tree["candidates"][0]["definitions"][0]["domain_conditions"], ["t >= 0"])

    def test_unicode_relation_is_normalized_and_chain_is_rejected(self):
        math_input = {
            **MATH_INPUT,
            "Constraint_Table": [{"name": "limit", "Mathematical_expressions": "x + z ≤ 10"}],
        }
        candidate = {
            "source_id": "constraint:0", "source_expression": "x + z ≤ 10",
            "relation": "<=", "linearized_form": "x + z - 10", "definitions": [],
            "domain_conditions": [],
        }
        tree = parse_auxiliary_response(response([candidate]), math_input)
        self.assertEqual(tree["candidates"][0]["relation"], "<=")

        chained = {
            **math_input,
            "Constraint_Table": [{"name": "chain", "Mathematical_expressions": "x <= z <= 10"}],
        }
        candidate["source_expression"] = "x <= z <= 10"
        with self.assertRaisesRegex(ValueError, "unsupported chained relation"):
            parse_auxiliary_response(response([candidate]), chained)

    def test_cross_candidate_auxiliary_name_collision_is_rejected(self):
        candidates = [
            {
                "source_id": "objective", "source_expression": "Minimize f = x**2 + z",
                "relation": "objective", "linearized_form": "t + z",
                "definitions": [{"symbol": "t", "rhs": "x**2", "domain_conditions": []}],
                "domain_conditions": [],
            },
            {
                "source_id": "constraint:0", "source_expression": "x + z <= 10",
                "relation": "<=", "linearized_form": "t",
                "definitions": [{"symbol": "t", "rhs": "x + z", "domain_conditions": []}],
                "domain_conditions": [],
            },
        ]
        with self.assertRaisesRegex(ValueError, "conflicts with an auxiliary definition"):
            parse_auxiliary_response(response(candidates), MATH_INPUT)

    def test_cycles_unresolved_indices_and_non_numeric_literals_are_rejected(self):
        cycle = {
            "source_id": "objective", "source_expression": "Minimize f = x**2 + z",
            "relation": "objective", "linearized_form": "t",
            "definitions": [
                {"symbol": "t", "rhs": "u + x", "domain_conditions": []},
                {"symbol": "u", "rhs": "t + z", "domain_conditions": []},
            ],
            "domain_conditions": [],
        }
        with self.assertRaisesRegex(ValueError, "forward or undeclared dependencies"):
            parse_auxiliary_response(response([cycle]), MATH_INPUT)

        indexed = {
            "source_id": "objective", "source_expression": "Minimize f = x**2 + z",
            "relation": "objective", "linearized_form": "x[9]",
            "definitions": [], "domain_conditions": [],
        }
        with self.assertRaisesRegex(ValueError, "undeclared dependencies"):
            parse_auxiliary_response(
                response([indexed]), {**MATH_INPUT, "Variables_List": [{"symbol": "x[0]"}, {"symbol": "z"}]}
            )

        for invalid in ["'text'", "True", "1e999", "nan", "inf"]:
            with self.subTest(invalid=invalid):
                candidate = {
                    "source_id": "objective", "source_expression": "Minimize f = x**2 + z",
                    "relation": "objective", "linearized_form": invalid,
                    "definitions": [], "domain_conditions": [],
                }
                with self.assertRaises(ValueError):
                    parse_auxiliary_response(response([candidate]), MATH_INPUT)


if __name__ == "__main__":
    unittest.main()
