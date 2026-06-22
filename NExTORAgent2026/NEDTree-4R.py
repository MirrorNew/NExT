import sympy
from sympy import Symbol, Add, Mul, Pow, exp, log, sin, cos
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


class TopDownNEDTree:
    def __init__(self, params, vars_list):
        self.params = {Symbol(p) for p in params}
        self.vars = {Symbol(v) for v in vars_list}
        self.transformations = standard_transformations + (implicit_multiplication_application,)
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
            expr_str = expr_str[:idx] + f"(({inner})**(0.5))" + expr_str[i:]
        return expr_str

    def _preprocess_expr(self, expr_str):
        expr_str = str(expr_str)
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

        lhs_str, rhs_str = expr_str, "0"
        for op in operators:
            if op in expr_str:
                self.relation = op if op != '==' else '='
                parts = expr_str.split(op, 1)
                lhs_str, rhs_str = parts[0], parts[1]
                break

        env = {str(s): s for s in self.params.union(self.vars)}
        env.update({'exp': exp, 'log': log, 'sin': sin, 'cos': cos, 'pow': Pow, 'pi': sympy.pi})

        lhs_expr = parse_expr(lhs_str, local_dict=env, transformations=self.transformations)
        rhs_expr = parse_expr(rhs_str, local_dict=env, transformations=self.transformations)
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
                if self.is_constant(exponent) and exponent.is_number and exponent < 0:
                    self.domain_constraints.append(f"{base} != 0")
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
                self.D_new[y_sym] = current_expr

        return {
            "linear_expr": self.L_f,
            "relation": self.relation or "expression",
            "definitions": self.D_new,
            "domain_constraints": self.domain_constraints,
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
