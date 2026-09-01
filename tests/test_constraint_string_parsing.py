from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import sympy


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("nedtree4r_constraint_strings", ROOT / "NEDTree-4R.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PARAMETERS = ["alpha", "beta", "gamma"]
VARIABLES = ["x_1", "x_2", "x_3"]
SYMBOLS = {
    name: sympy.Symbol(name)
    for name in PARAMETERS + VARIABLES
}
SYMBOLS.update(
    exp=sympy.exp,
    log=sympy.log,
    sin=sympy.sin,
    cos=sympy.cos,
    sqrt=sympy.sqrt,
)


def expand_result(result):
    expanded_definitions = {}
    for symbol, expression in result["definitions"].items():
        expanded_definitions[symbol] = expression.xreplace(expanded_definitions)
    return result["linear_expr"].xreplace(expanded_definitions)


CONSTRAINT_CASES = [
    pytest.param(
        r"x_1 * x_2 * x_3 <= alpha",
        "<=",
        "x_1*x_2*x_3-alpha",
        set(),
        id="previous-product",
    ),
    pytest.param(
        r"x_1^(3/2) >= beta",
        ">=",
        "x_1**(3/2)-beta",
        {"x_1 >= 0"},
        id="previous-positive-fractional-power",
    ),
    pytest.param(
        r"x_1/x_2 + x_2^-1 <= gamma",
        "<=",
        "x_1/x_2+x_2**-1-gamma",
        {"x_2 != 0"},
        id="previous-division-negative-power",
    ),
    pytest.param(
        r"log(x_1) >= alpha",
        ">=",
        "log(x_1)-alpha",
        {"x_1 > 0"},
        id="previous-logarithm",
    ),
    pytest.param(
        r"sqrt(x_1) <= beta",
        "<=",
        "sqrt(x_1)-beta",
        {"x_1 >= 0"},
        id="previous-square-root",
    ),
    pytest.param(
        r"cos(log(x_1*x_2)) + exp(x_3^-1) <= gamma",
        "<=",
        "cos(log(x_1*x_2))+exp(x_3**-1)-gamma",
        {"x_1*x_2 > 0", "x_3 != 0"},
        id="previous-nested-expression",
    ),
    pytest.param(
        r"alpha + beta * x_1**3 * exp(2*x_2 + x_1**2) + gamma * cos(log(x_3)) > 0",
        ">",
        "alpha+beta*x_1**3*exp(2*x_2+x_1**2)+gamma*cos(log(x_3))",
        {"x_3 > 0"},
        id="user-strict-nested-constraint",
    ),
    pytest.param(
        r"exp(x_1 + x_2) + sin(x_3) = alpha",
        "=",
        "exp(x_1+x_2)+sin(x_3)-alpha",
        set(),
        id="equality-nested-functions",
    ),
    pytest.param(
        r"\frac{x_1}{x_2} + \sqrt{x_3} \le beta",
        "<=",
        "x_1/x_2+sqrt(x_3)-beta",
        {"x_2 != 0", "x_3 >= 0"},
        id="latex-fraction-square-root",
    ),
    pytest.param(
        r"alpha + x_1**2 >= beta * exp(x_2)",
        ">=",
        "alpha+x_1**2-beta*exp(x_2)",
        set(),
        id="nonlinear-both-sides",
    ),
]


@pytest.mark.parametrize(
    ("expr_str", "expected_relation", "expected_root", "expected_domains"),
    CONSTRAINT_CASES,
)
def test_constraint_string_parsing_ac(
    expr_str,
    expected_relation,
    expected_root,
    expected_domains,
):
    result = MODULE.TopDownNEDTree(PARAMETERS, VARIABLES).process(expr_str)
    reconstructed = expand_result(result)
    expected = sympy.sympify(expected_root, locals=SYMBOLS)

    assert result["relation"] == expected_relation
    assert set(result["domain_constraints"]) == expected_domains
    assert sympy.simplify(reconstructed - expected) == 0
    assert result["validation"] == {
        "closed": True,
        "acyclic": True,
        "topological": True,
    }
