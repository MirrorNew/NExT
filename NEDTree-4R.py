import ast
import math
import re

import sympy
from sympy import Symbol, Add, Mul, Pow, exp, log, sin, cos


class NEDTreeError(ValueError):
    """Base exception for declared NED-Tree input-contract failures."""


class UnsupportedExpressionError(NEDTreeError):
    """Raised when an expression falls outside the supported scalar grammar."""


class DefinitionValidationError(NEDTreeError):
    """Raised when generated auxiliary definitions are not closed or acyclic."""


class TopDownNEDTree:
    def __init__(self, params, vars_list):
        self.params = {Symbol(p) for p in params}
        self.vars = {Symbol(v) for v in vars_list}
        self._reset_state()

    def _reset_state(self):
        self.L_f = None
        self.relation = None
        self.D_new = {}
        self.y_counter = 1

        self.queue = []
        self.y_vars = set()
        self.domain_constraints = []

    def _replace_frac(self, expr_str):
        while r'\frac' in expr_str:
            idx = expr_str.find(r'\frac')
            cursor = idx + 5

            def find_block(start_idx):
                while start_idx < len(expr_str) and expr_str[start_idx].isspace():
                    start_idx += 1
                if start_idx >= len(expr_str) or expr_str[start_idx] != '{':
                    return None, start_idx

                depth = 1
                i = start_idx + 1
                while i < len(expr_str) and depth > 0:
                    if expr_str[i] == '{':
                        depth += 1
                    elif expr_str[i] == '}':
                        depth -= 1
                    i += 1

                if depth == 0:
                    return expr_str[start_idx + 1:i - 1], i
                return None, i

            numerator, next_idx = find_block(cursor)
            if numerator is None:
                break
            denominator, end_idx = find_block(next_idx)
            if denominator is None:
                break

            expr_str = expr_str[:idx] + f"(({numerator})/({denominator}))" + expr_str[end_idx:]
        return expr_str

    def _replace_sqrt(self, expr_str):
        while r'\sqrt' in expr_str:
            idx = expr_str.find(r'\sqrt')
            cursor = idx + 5
            while cursor < len(expr_str) and expr_str[cursor].isspace():
                cursor += 1
            if cursor >= len(expr_str) or expr_str[cursor] != '{':
                break

            depth = 1
            i = cursor + 1
            while i < len(expr_str) and depth > 0:
                if expr_str[i] == '{':
                    depth += 1
                elif expr_str[i] == '}':
                    depth -= 1
                i += 1

            if depth != 0:
                break

            inner = expr_str[cursor + 1:i - 1]
            expr_str = expr_str[:idx] + f"(({inner})**(1/2))" + expr_str[i:]
        return expr_str

    @staticmethod
    def _replace_latex_subscripts(expr_str):
        return re.sub(
            r"([^\W\d]\w*)_\{([A-Za-z0-9]+)\}",
            r"\1_\2",
            expr_str,
        )

    def _insert_implicit_multiplication(self, expr_str):
        """Expand only unambiguous products involving declared symbols."""

        names = sorted(
            (str(symbol) for symbol in self.params.union(self.vars)),
            key=len,
            reverse=True,
        )
        tokens = [rf"(?<!\w){re.escape(name)}(?!\w)" for name in names]
        right_tokens = [rf"{re.escape(name)}(?!\w)" for name in names]

        # Numeric coefficients and adjacent parenthesized factors.
        for right_token in right_tokens:
            expr_str = re.sub(
                rf"(?<![\w.])(\d+(?:\.\d+)?)\s*(?={right_token})",
                r"\1*",
                expr_str,
            )
            expr_str = re.sub(rf"\)\s*(?={right_token})", ")*", expr_str)
        expr_str = re.sub(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?=\()",
            r"\1*",
            expr_str,
        )
        expr_str = re.sub(r"\)\s*(?=\()", ")*", expr_str)

        # A declared scalar followed by a parenthesized factor or another
        # declared scalar is multiplication, never a free-form function call.
        for left_token in tokens:
            expr_str = re.sub(
                rf"({left_token})\s*(?=\()",
                rf"\1*",
                expr_str,
            )
            for right_token in right_tokens:
                expr_str = re.sub(
                    rf"({left_token})\s+(?={right_token})",
                    rf"\1*",
                    expr_str,
                )
        return expr_str

    def _preprocess_expr(self, expr_str):
        expr_str = str(expr_str)
        expr_str = self._replace_latex_subscripts(expr_str)
        expr_str = self._replace_frac(expr_str)
        expr_str = self._replace_sqrt(expr_str)
        replacements = {
            r'\cdot': '*',
            r'\times': '*',
            r'\leq': '<=',
            r'\geq': '>=',
            r'\le': '<=',
            r'\ge': '>=',
        }
        for old, new in replacements.items():
            expr_str = expr_str.replace(old, new)

        expr_str = expr_str.replace('^', '**')
        expr_str = expr_str.replace('{', '(').replace('}', ')')
        expr_str = expr_str.replace('\\', '')
        expr_str = self._insert_implicit_multiplication(expr_str)
        return expr_str

    def _get_new_var(self):
        y_sym = Symbol(f'y_temp_{self.y_counter}')
        self.y_counter += 1
        self.y_vars.add(y_sym)
        return y_sym

    def is_constant(self, expr):
        if expr.is_number:
            return True
        return all(sym in self.params for sym in expr.free_symbols)

    @staticmethod
    def _symbol_sort_key(symbol):
        name = str(symbol)
        try:
            return int(name.rsplit('_', 1)[-1])
        except ValueError:
            return name

    def _add_domain_constraint(self, constraint):
        if constraint not in self.domain_constraints:
            self.domain_constraints.append(constraint)

    @staticmethod
    def _reject_nonreal(expr):
        invalid_atoms = (sympy.zoo, sympy.oo, -sympy.oo, sympy.nan, sympy.I)
        if expr.has(*invalid_atoms) or expr.is_real is False:
            raise UnsupportedExpressionError("NONREAL_OR_NONFINITE_EXPRESSION")

    def _record_division_domain(self, denominator):
        if denominator.is_zero is True:
            raise UnsupportedExpressionError("ZERO_DENOMINATOR")
        if denominator.is_zero is not False:
            self._add_domain_constraint(f"{denominator} != 0")

    def _record_power_domain(self, base, exponent):
        variable_exponent = bool(exponent.free_symbols.intersection(self.vars))
        if variable_exponent:
            if not (base.is_number and base.is_positive is True):
                raise UnsupportedExpressionError(
                    "VARIABLE_EXPONENT_REQUIRES_POSITIVE_NUMERIC_BASE"
                )
            return

        if exponent.is_number:
            if exponent.is_integer is True:
                if exponent.is_negative is True:
                    self._record_division_domain(base)
                return

            if exponent.is_Rational:
                if exponent.q % 2 == 0:
                    if base.is_negative is True:
                        raise UnsupportedExpressionError(
                            "NONREAL_OR_NONFINITE_EXPRESSION"
                        )
                    if exponent.is_negative is True:
                        if base.is_zero is True:
                            raise UnsupportedExpressionError("ZERO_DENOMINATOR")
                        if base.is_positive is not True:
                            self._add_domain_constraint(f"{base} > 0")
                    elif base.is_nonnegative is not True:
                        self._add_domain_constraint(f"{base} >= 0")
                elif exponent.is_negative is True:
                    self._record_division_domain(base)
                return

            if base.is_negative is True:
                raise UnsupportedExpressionError(
                    "NONREAL_OR_NONFINITE_EXPRESSION"
                )
            if exponent.is_negative is True:
                if base.is_zero is True:
                    raise UnsupportedExpressionError("ZERO_DENOMINATOR")
                if base.is_positive is not True:
                    self._add_domain_constraint(f"{base} > 0")
            elif base.is_nonnegative is not True:
                self._add_domain_constraint(f"{base} >= 0")
            return

        if self.is_constant(exponent):
            self._add_domain_constraint(f"real_power_domain({base}, {exponent})")
            return

        raise UnsupportedExpressionError("UNSUPPORTED_POWER_EXPONENT")

    def _record_function_domain(self, function, argument):
        if function != log:
            return
        if argument.is_positive is True:
            return
        if argument.is_nonpositive is True:
            raise UnsupportedExpressionError("INVALID_LOG_DOMAIN")
        self._add_domain_constraint(f"{argument} > 0")

    def _build_sympy_from_ast(self, node, declared_symbols):
        if isinstance(node, ast.Expression):
            return self._build_sympy_from_ast(node.body, declared_symbols)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise UnsupportedExpressionError("UNSUPPORTED_LITERAL")
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise UnsupportedExpressionError("NONREAL_OR_NONFINITE_EXPRESSION")
            return sympy.Integer(node.value) if isinstance(node.value, int) else sympy.Float(node.value)

        if isinstance(node, ast.Name):
            if node.id in {"zoo", "oo", "nan", "I"}:
                raise UnsupportedExpressionError(
                    f"NONREAL_OR_NONFINITE_EXPRESSION: {node.id}"
                )
            if node.id not in declared_symbols:
                raise UnsupportedExpressionError(f"UNDECLARED_SYMBOL: {node.id}")
            return declared_symbols[node.id]

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._build_sympy_from_ast(node.operand, declared_symbols)
            result = operand if isinstance(node.op, ast.UAdd) else Mul(-1, operand)
            self._reject_nonreal(result)
            return result

        if isinstance(node, ast.BinOp):
            left = self._build_sympy_from_ast(node.left, declared_symbols)
            right = self._build_sympy_from_ast(node.right, declared_symbols)
            if isinstance(node.op, ast.Add):
                result = Add(left, right)
            elif isinstance(node.op, ast.Sub):
                result = Add(left, Mul(-1, right))
            elif isinstance(node.op, ast.Mult):
                result = Mul(left, right)
            elif isinstance(node.op, ast.Div):
                self._record_division_domain(right)
                result = Mul(left, Pow(right, -1))
            elif isinstance(node.op, ast.Pow):
                self._record_power_domain(left, right)
                result = Pow(left, right)
            else:
                raise UnsupportedExpressionError(
                    f"UNSUPPORTED_OPERATOR: {node.op.__class__.__name__}"
                )
            self._reject_nonreal(result)
            return result

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsupportedExpressionError("UNSUPPORTED_OPERATOR: function_call")
            functions = {"exp": exp, "log": log, "sin": sin, "cos": cos}
            special_functions = {"sqrt", "log2"}
            if node.func.id not in functions and node.func.id not in special_functions:
                raise UnsupportedExpressionError(
                    f"UNSUPPORTED_OPERATOR: {node.func.id}"
                )
            if len(node.args) != 1 or node.keywords:
                raise UnsupportedExpressionError(
                    f"UNSUPPORTED_FUNCTION_ARITY: {node.func.id}"
                )
            argument = self._build_sympy_from_ast(node.args[0], declared_symbols)
            if node.func.id == "sqrt":
                exponent = sympy.Rational(1, 2)
                self._record_power_domain(argument, exponent)
                result = Pow(argument, exponent)
                self._reject_nonreal(result)
                return result
            if node.func.id == "log2":
                self._record_function_domain(log, argument)
                result = Mul(log(argument), Pow(log(2), -1))
                self._reject_nonreal(result)
                return result
            function = functions[node.func.id]
            self._record_function_domain(function, argument)
            result = function(argument)
            self._reject_nonreal(result)
            return result

        raise UnsupportedExpressionError(
            f"UNSUPPORTED_OPERATOR: {node.__class__.__name__}"
        )

    def _parse_expression(self, expr_str):
        expr_str = expr_str.strip()
        if not expr_str:
            raise UnsupportedExpressionError("INVALID_EXPRESSION_SYNTAX")
        try:
            parsed = ast.parse(expr_str, mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise UnsupportedExpressionError("INVALID_EXPRESSION_SYNTAX") from exc

        nodes = list(ast.walk(parsed))
        if len(nodes) > 512:
            raise UnsupportedExpressionError("EXPRESSION_TOO_COMPLEX")

        declared_symbols = {
            str(symbol): symbol for symbol in self.params.union(self.vars)
        }
        return self._build_sympy_from_ast(parsed, declared_symbols)

    def _validate_expression(self, expr):
        declared_symbols = self.params.union(self.vars)
        unknown_symbols = expr.free_symbols - declared_symbols
        if unknown_symbols:
            names = ', '.join(sorted(str(symbol) for symbol in unknown_symbols))
            raise UnsupportedExpressionError(f"UNDECLARED_SYMBOL: {names}")

        for node in sympy.preorder_traversal(expr):
            if node.is_Atom:
                continue

            if node.is_Add or node.is_Mul:
                continue

            if node.is_Pow:
                base, exponent = node.args
                continue

            if node.func in [exp, log, sin, cos]:
                continue

            raise UnsupportedExpressionError(
                f"UNSUPPORTED_OPERATOR: {getattr(node.func, '__name__', str(node.func))}"
            )

    def _validate_and_order_definitions(self):
        definition_symbols = set(self.D_new)
        referenced_by_root = self.L_f.free_symbols.intersection(self.y_vars)
        referenced_by_definitions = set().union(
            *(expr.free_symbols.intersection(self.y_vars) for expr in self.D_new.values())
        ) if self.D_new else set()
        undefined = (referenced_by_root | referenced_by_definitions) - definition_symbols
        if undefined:
            names = ', '.join(sorted(str(symbol) for symbol in undefined))
            raise DefinitionValidationError(f"UNCLOSED_DEFINITIONS: {names}")

        dependencies = {
            symbol: expr.free_symbols.intersection(self.y_vars)
            for symbol, expr in self.D_new.items()
        }
        state = {}
        ordered = []

        def visit(symbol):
            marker = state.get(symbol, 0)
            if marker == 1:
                raise DefinitionValidationError(f"CYCLIC_DEFINITIONS: {symbol}")
            if marker == 2:
                return

            state[symbol] = 1
            for dependency in sorted(
                dependencies[symbol], key=self._symbol_sort_key
            ):
                visit(dependency)
            state[symbol] = 2
            ordered.append(symbol)

        for symbol in sorted(definition_symbols, key=self._symbol_sort_key):
            visit(symbol)

        self.D_new = {symbol: self.D_new[symbol] for symbol in ordered}

    def extract_linear(self, expr):
        if self.is_constant(expr) or expr in self.vars or expr in self.y_vars:
            return expr

        if expr.is_Add:
            return Add(*[self.extract_linear(arg) for arg in expr.args])

        if expr.is_Mul:
            coeffs = [arg for arg in expr.args if self.is_constant(arg)]
            non_coeffs = [arg for arg in expr.args if not self.is_constant(arg)]

            if len(non_coeffs) == 0:
                return expr
            elif len(non_coeffs) == 1:
                return Mul(*(coeffs + [self.extract_linear(non_coeffs[0])]))
            else:
                mul_expr = Mul(*non_coeffs)
                y_new = self._get_new_var()
                self.queue.append((y_new, mul_expr))
                return Mul(*(coeffs + [y_new]))

        y_new = self._get_new_var()
        self.queue.append((y_new, expr))
        return y_new

    def ensure_single_var(self, expr):
        if self.is_constant(expr) or expr in self.vars or expr in self.y_vars:
            return expr
        y_new = self._get_new_var()
        self.queue.append((y_new, expr))
        return y_new

    def process(self, expr_str):
        self._reset_state()
        expr_str = self._preprocess_expr(expr_str)
        operators = ['>=', '<=', '>', '<', '==', '=']

        if '!=' in expr_str:
            raise UnsupportedExpressionError(
                "UNSUPPORTED_RELATION: != is not a solver constraint in this grammar"
            )

        lhs_str, rhs_str = expr_str, "0"
        for op in operators:
            if op in expr_str:
                self.relation = op if op != '==' else '='
                parts = expr_str.split(op, 1)
                lhs_str, rhs_str = parts[0], parts[1]
                break

        lhs_expr = self._parse_expression(lhs_str)
        rhs_expr = self._parse_expression(rhs_str)
        self._validate_expression(lhs_expr)
        self._validate_expression(rhs_expr)
        root_expr = lhs_expr - rhs_expr

        self.L_f = self.extract_linear(root_expr)

        while self.queue:
            y_sym, current_expr = self.queue.pop(0)

            if current_expr.is_Add:
                lin_expr = Add(*[self.extract_linear(arg) for arg in current_expr.args])
                self.D_new[y_sym] = lin_expr

            elif current_expr.is_Mul:
                coeffs = [arg for arg in current_expr.args if self.is_constant(arg)]
                non_coeffs = [arg for arg in current_expr.args if not self.is_constant(arg)]

                v_args = [self.ensure_single_var(arg) for arg in non_coeffs]

                if len(v_args) == 1:
                    self.D_new[y_sym] = Mul(*(coeffs + v_args))
                else:
                    cur = v_args[0]
                    for nxt in v_args[1:]:
                        if nxt == v_args[-1] and not coeffs:
                            self.D_new[y_sym] = Mul(cur, nxt, evaluate=False)
                        else:
                            y_tmp = self._get_new_var()
                            self.D_new[y_tmp] = Mul(cur, nxt, evaluate=False)
                            cur = y_tmp

                    if coeffs:
                        self.D_new[y_sym] = Mul(*(coeffs + [cur]), evaluate=False)

            elif current_expr.is_Pow:
                base, exponent = current_expr.args
                if self.is_constant(base) and not self.is_constant(exponent):
                    c = log(base)
                    inner = Mul(exponent, c)
                    v_inner = self.ensure_single_var(inner)
                    self.D_new[y_sym] = exp(v_inner)
                else:
                    v_base = self.ensure_single_var(base)
                    v_exp = self.ensure_single_var(exponent)
                    self.D_new[y_sym] = Pow(v_base, v_exp, evaluate=False)

            elif current_expr.func in [exp, log, sin, cos]:
                arg = current_expr.args[0]
                v_arg = self.ensure_single_var(arg)
                self.D_new[y_sym] = current_expr.func(v_arg)

            else:
                raise UnsupportedExpressionError(
                    f"UNSUPPORTED_OPERATOR: {current_expr.func.__name__}"
                )

        self._validate_and_order_definitions()

        return {
            "linear_expr": self.L_f,
            "relation": self.relation or "expression",
            "definitions": self.D_new,
            "domain_constraints": self.domain_constraints,
            "validation": {
                "closed": True,
                "acyclic": True,
                "topological": True,
            },
        }

    # ==========================================
    # 打印与可视化模块
    # ==========================================
    def print_tree(self):
        print("=" * 50)
        print(" [最终线性主干约束 L_f] ".center(50, "="))
        if self.relation:
            print(f"  {self.L_f} {self.relation} 0")
        else:
            print(f"  {self.L_f} (纯表达式)")

        print("\n [辅助变量定义 D_new] ".center(50, "="))
        sorted_defs = sorted(self.D_new.items(), key=lambda x: int(str(x[0]).split('_')[-1]))
        for y, eq in sorted_defs:
            print(f"  {y} = {eq}")
        if self.domain_constraints:
            print("\n [定义域约束] ".center(50, "="))
            for item in self.domain_constraints:
                print(f"  {item}")

        print("\n [NED-Tree 拓扑可视化] ".center(50, "="))
        # 调用基于您提供逻辑改进的打印函数
        self.print_ned_tree(self.L_f)
        print("=" * 50)

    def print_ned_tree(self, expr, prefix="", is_last=True, is_root=True):
        if is_root:
            connector = ""
            new_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")

        node_text = ""
        children = []

        if isinstance(expr, Symbol) and expr in self.y_vars:
            definition = self.D_new[expr]
            def_str = str(definition)
            node_text = f"\033[94m{expr}\033[0m \033[90m[= {def_str}]\033[0m"

            # 【修复在这里】：把 definition 当作唯一的子节点装进去，而不是拿 definition.args
            children = [definition]

        elif expr.is_Add or expr.is_Mul or expr.is_Pow or expr.func in [exp, log, sin, cos]:
            node_text = f"\033[92m{expr.func.__name__}\033[0m"
            children = list(expr.args)
        else:
            node_text = str(expr)
            children = []

        print(f"{prefix}{connector}{node_text}")
        count = len(children)
        for i, child in enumerate(children):
            self.print_ned_tree(child, new_prefix, i == count - 1, is_root=False)


# ==================== 测试用例 ====================
if __name__ == '__main__':
    print("\n>>> Test Case 1: 带有 \\ge 的综合约束与树状打印")
    expr1 = "alpha + beta * 3**x_1 * exp(2*x_2) + gamma * cos(log(x_3)) \\ge 10"
    params1 = ['alpha', 'beta', 'gamma']
    vars1 = ['x_1', 'x_2', 'x_3']

    ned1 = TopDownNEDTree(params1, vars1)
    ned1.process(expr1)
    ned1.print_tree()
