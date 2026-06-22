from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("nedtree4r", ROOT / "NEDTree-4R.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def build_tree():
    return MODULE.TopDownNEDTree(
        params=["alpha", "beta", "gamma"],
        vars_list=["x_1", "x_2", "x_3"],
    )


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


def test_negative_power_records_domain_constraint():
    ned = build_tree()
    result = ned.process(r"x_1^-1 + alpha <= gamma")

    assert result["domain_constraints"]
    assert "x_1" in result["domain_constraints"][0]


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
