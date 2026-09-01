from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import sympy


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("nedtree4r", ROOT / "NEDTree-4R.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build_tree():
    return MODULE.TopDownNEDTree(
        params=["alpha", "beta", "gamma"],
        vars_list=["x_1", "x_2", "x_3"],
    )


def expand_result(result):
    expanded_definitions = {}
    for symbol, expression in result["definitions"].items():
        expanded_definitions[symbol] = expression.xreplace(expanded_definitions)
    return result["linear_expr"].xreplace(expanded_definitions)


def test_case_study_expression():
    ned = build_tree()
    result = ned.process(r"alpha + beta * 3^x_1 * exp(2*x_2) + gamma * cos(log(x_3)) \ge 10")

    assert str(result["relation"]) == ">="
    assert "alpha" in str(result["linear_expr"])
    assert "y_temp" in str(result["linear_expr"])
    assert any("cos" in str(value) for value in result["definitions"].values())
    assert any("log" in str(value) for value in result["definitions"].values())
    assert any("exp" in str(value) for value in result["definitions"].values())


def test_state_is_reset_between_calls():
    ned = build_tree()
    first = ned.process(r"alpha + beta * x_1 * x_2 >= 1")
    first_defs = len(first["definitions"])

    second = ned.process(r"alpha + x_1 <= 2")

    assert first_defs > 0
    assert len(second["definitions"]) == 0
    assert "y_temp" not in str(second["linear_expr"])
    assert str(second["relation"]) == "<="


def test_latex_fraction_and_sqrt():
    ned = build_tree()
    result = ned.process(r"\frac{x_1}{x_2} + \sqrt{\frac{x_3}{gamma}} \le gamma")

    assert str(result["relation"]) == "<="
    assert "y_temp" in str(result["linear_expr"])
    assert any("Pow" in value.func.__name__ or "**" in str(value) or "pow" in str(value).lower()
               for value in result["definitions"].values())


def test_plain_sqrt_log2_and_implicit_multiplication():
    result = build_tree().process(
        "2x_1 + 0.5(x_2 + x_3) + x_1(x_2) + sqrt(x_3) + log2(x_1)"
    )

    assert result["definitions"]
    assert "x_1 > 0" in result["domain_constraints"]
    assert "x_3 >= 0" in result["domain_constraints"]


def test_latex_subscript_and_spaced_numeric_coefficient():
    ned = MODULE.TopDownNEDTree(params=[], vars_list=["x_11"])
    result = ned.process(r"0.01 x_{11}^2 + 2 x_{11}")

    assert result["definitions"]
    assert "x_11" in str(result["linear_expr"])


def test_negative_power_records_domain_constraint():
    ned = build_tree()
    result = ned.process(r"x_1^-1 + alpha <= gamma")

    assert result["domain_constraints"]
    assert "x_1" in result["domain_constraints"][0]


def test_definitions_are_closed_acyclic_and_topological():
    result = build_tree().process(
        r"alpha + beta * 3^x_1 * exp(2*x_2) + gamma * cos(log(x_3)) \ge 10"
    )
    definitions = result["definitions"]
    auxiliary_symbols = set(definitions)
    defined = set()

    for symbol, expression in definitions.items():
        dependencies = expression.free_symbols.intersection(auxiliary_symbols)
        assert dependencies <= defined
        defined.add(symbol)

    assert result["linear_expr"].free_symbols.intersection(auxiliary_symbols) <= defined
    assert result["validation"] == {
        "closed": True,
        "acyclic": True,
        "topological": True,
    }


def test_output_is_deterministic():
    expression = r"alpha + beta*x_1*x_2 + cos(log(x_3)) <= gamma"
    first = build_tree().process(expression)
    second = build_tree().process(expression)

    def serialize(result):
        return (
            sympy.srepr(result["linear_expr"]),
            tuple(
                (str(symbol), sympy.srepr(value))
                for symbol, value in result["definitions"].items()
            ),
            tuple(result["domain_constraints"]),
        )

    assert serialize(first) == serialize(second)


def test_domain_obligations_are_explicit():
    result = build_tree().process(
        r"log(x_1) + x_2^-1 + \sqrt{x_3} <= gamma"
    )

    assert "x_1 > 0" in result["domain_constraints"]
    assert "x_2 != 0" in result["domain_constraints"]
    assert "x_3 >= 0" in result["domain_constraints"]


def test_domains_are_collected_before_sympy_simplification():
    quotient = build_tree().process("x_1/x_1")
    cancelled_log = build_tree().process("0*log(x_1)")

    assert quotient["linear_expr"] == 1
    assert "x_1 != 0" in quotient["domain_constraints"]
    assert cancelled_log["linear_expr"] == 0
    assert "x_1 > 0" in cancelled_log["domain_constraints"]


@pytest.mark.parametrize(
    "expression",
    [
        "x_1/0",
        "1/(x_1-x_1)",
    ],
)
def test_zero_denominators_are_rejected(expression):
    with pytest.raises(MODULE.UnsupportedExpressionError, match="ZERO_DENOMINATOR"):
        build_tree().process(expression)


@pytest.mark.parametrize(
    ("expression", "error_code"),
    [
        ("log(-1)", "INVALID_LOG_DOMAIN"),
        ("(-1)**(1/2)", "NONREAL_OR_NONFINITE_EXPRESSION"),
        ("zoo", "NONREAL_OR_NONFINITE_EXPRESSION"),
        ("oo", "NONREAL_OR_NONFINITE_EXPRESSION"),
        ("nan", "NONREAL_OR_NONFINITE_EXPRESSION"),
        ("I", "NONREAL_OR_NONFINITE_EXPRESSION"),
        ("1e309", "NONREAL_OR_NONFINITE_EXPRESSION"),
    ],
)
def test_nonreal_and_nonfinite_inputs_are_rejected(expression, error_code):
    with pytest.raises(MODULE.UnsupportedExpressionError, match=error_code):
        build_tree().process(expression)


@pytest.mark.parametrize("function_name", ["unknown", "abs", "Abs", "indicator"])
def test_unapproved_functions_are_explicitly_unsupported(function_name):
    with pytest.raises(MODULE.UnsupportedExpressionError, match="UNSUPPORTED_OPERATOR"):
        build_tree().process(f"{function_name}(x_1)")


def test_ast_whitelist_rejects_python_evaluation_constructs():
    with pytest.raises(MODULE.UnsupportedExpressionError, match="UNSUPPORTED_OPERATOR"):
        build_tree().process("__import__('os').system('echo unsafe')")


def test_only_declared_symbols_are_accepted_even_for_legacy_constants():
    with pytest.raises(MODULE.UnsupportedExpressionError, match="UNDECLARED_SYMBOL"):
        build_tree().process("pi*x_1")


@pytest.mark.parametrize(
    ("expression", "error_code"),
    [
        ("x_1 != x_2", "UNSUPPORTED_RELATION"),
        ("Abs(x_1) <= gamma", "UNSUPPORTED_OPERATOR"),
        ("x_1^x_2 <= gamma", "VARIABLE_EXPONENT_REQUIRES_POSITIVE_NUMERIC_BASE"),
        ("x_1 + undeclared <= gamma", "UNDECLARED_SYMBOL"),
    ],
)
def test_unsupported_inputs_fail_explicitly(expression, error_code):
    with pytest.raises(MODULE.UnsupportedExpressionError, match=error_code):
        build_tree().process(expression)


def test_generated_representation_is_numerically_equivalent():
    result = build_tree().process(
        "alpha + beta*3^x_1*exp(2*x_2) + gamma*cos(log(x_3))"
    )
    expanded = expand_result(result)
    alpha, beta, gamma, x_1, x_2, x_3 = sympy.symbols(
        "alpha beta gamma x_1 x_2 x_3"
    )
    original = alpha + beta * 3**x_1 * sympy.exp(2 * x_2) + gamma * sympy.cos(
        sympy.log(x_3)
    )
    point = {
        alpha: 1.5,
        beta: 0.7,
        gamma: 2.0,
        x_1: 0.8,
        x_2: -0.2,
        x_3: 1.7,
    }

    assert float(expanded.subs(point)) == pytest.approx(
        float(original.subs(point)), rel=1e-12, abs=1e-12
    )


def test_definition_validator_rejects_cycles_and_missing_definitions():
    y_1, y_2 = sympy.symbols("y_temp_1 y_temp_2")
    ned = build_tree()
    ned.L_f = y_1
    ned.y_vars = {y_1, y_2}
    ned.D_new = {y_1: y_2, y_2: y_1}
    with pytest.raises(MODULE.DefinitionValidationError, match="CYCLIC_DEFINITIONS"):
        ned._validate_and_order_definitions()

    ned.L_f = y_1
    ned.y_vars = {y_1, y_2}
    ned.D_new = {y_1: y_2}
    with pytest.raises(MODULE.DefinitionValidationError, match="UNCLOSED_DEFINITIONS"):
        ned._validate_and_order_definitions()


if __name__ == "__main__":
    tests = [
        test_case_study_expression,
        test_state_is_reset_between_calls,
        test_latex_fraction_and_sqrt,
        test_negative_power_records_domain_constraint,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
