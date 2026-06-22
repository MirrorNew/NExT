import sympy
import re
from sympy import Symbol, Add, Mul, Pow, exp, log, sin, cos, pi
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


class NEDTreeBuilder:
    def __init__(self):
        self.y_counter = 1
        self.definitions = {}
        self.constraints = []
        self.allowed_funcs = {'exp', 'log', 'sin', 'cos', 'pow'}
        self.func_map = {exp, log, sin, cos, Pow}

        self.name_to_alias = {}
        self.alias_to_name = {}
        self.param_aliases = set()
        self.var_aliases = set()
        self.local_env = {}

    def register_symbols(self, params_list, vars_list):
        self.name_to_alias = {}
        self.alias_to_name = {}
        self.param_aliases = set()
        self.var_aliases = set()
        self.local_env = {}

        # 注册参数
        for i, name in enumerate(params_list):
            alias_str = f"__PAR_{i}"
            alias_sym = Symbol(alias_str)
            self.name_to_alias[name] = alias_str
            self.alias_to_name[alias_str] = name
            self.param_aliases.add(alias_sym)
            self.local_env[alias_str] = alias_sym

        # 注册变量
        for i, name in enumerate(vars_list):
            alias_str = f"__VAR_{i}"
            alias_sym = Symbol(alias_str)
            self.name_to_alias[name] = alias_str
            self.alias_to_name[alias_str] = name
            self.var_aliases.add(alias_sym)
            self.local_env[alias_str] = alias_sym

    def _replace_frac(self, expr_str):
        while r'\frac' in expr_str:
            idx = expr_str.find(r'\frac')
            cursor = idx + 5

            def find_block(start_idx):
                while start_idx < len(expr_str) and expr_str[start_idx].isspace():
                    start_idx += 1
                if start_idx >= len(expr_str) or expr_str[start_idx] != '{':
                    return None, start_idx
                count = 1
                i = start_idx + 1
                while i < len(expr_str) and count > 0:
                    if expr_str[i] == '{':
                        count += 1
                    elif expr_str[i] == '}':
                        count -= 1
                    i += 1
                if count == 0: return expr_str[start_idx + 1:i - 1], i
                return None, i

            num_str, next_idx = find_block(cursor)
            if num_str is None: break
            denom_str, end_idx = find_block(next_idx)
            if denom_str is None: break
            new_segment = f"({num_str})/({denom_str})"
            expr_str = expr_str[:idx] + new_segment + expr_str[end_idx:]
        return expr_str

    def _mangle_string(self, expr_str):
        expr_str = self._replace_frac(expr_str)
        expr_str = expr_str.replace(r'\cdot', '*')
        expr_str = expr_str.replace(r'\times', '*')
        expr_str = re.sub(r'\\sqrt\{([^}]+)\}', r'pow(\1, 0.5)', expr_str)
        expr_str = expr_str.replace(r'\leq', '<=')
        expr_str = expr_str.replace(r'\geq', '>=')
        expr_str = expr_str.replace(r'\le', '<=')
        expr_str = expr_str.replace(r'\ge', '>=')
        expr_str = expr_str.replace('^', '**')
        expr_str = expr_str.replace('\\', '')

        if self.name_to_alias:
            keys = sorted(self.name_to_alias.keys(), key=len, reverse=True)
            pattern = re.compile("|".join(map(re.escape, keys)))

            def replace_callback(match):
                return f" {self.name_to_alias[match.group(0)]} "

            expr_str = pattern.sub(replace_callback, expr_str)

        expr_str = expr_str.replace('{', '(').replace('}', ')')
        return expr_str

    def _custom_format(self, expr):
        if isinstance(expr, Symbol):
            name = str(expr)
            return self.alias_to_name.get(name, name)

        if expr.is_Pow:
            base, exp_val = expr.args
            base_str = self._custom_format(base)
            exp_str = self._custom_format(exp_val)
            return f"pow({base_str}, {exp_str})"

        if expr.args:
            func_name = str(expr.func)
            if expr.is_Add:
                return " + ".join([self._custom_format(arg) for arg in expr.args])
            elif expr.is_Mul:
                args_str = []
                for arg in expr.args:
                    s = self._custom_format(arg)
                    if arg.is_Add: s = f"({s})"
                    args_str.append(s)
                return "*".join(args_str)
            elif expr.func in self.func_map:
                args_str = ", ".join([self._custom_format(arg) for arg in expr.args])
                return f"{func_name}({args_str})"

        return str(expr)

    def _normalize_negative_args(self, func_type, args):
        arg = args[0]
        if func_type == exp:
            coeff, term = arg.as_coeff_Mul()
            if coeff < 0:
                positive_arg = term * abs(coeff)
                inner = exp(positive_arg)
                return Pow(inner, -1, evaluate=False)
        return None

    def process(self, expr_input):
        self.y_counter = 1
        self.definitions = {}
        self.constraints = []

        mangled_str = self._mangle_string(str(expr_input))

        relation_op = None
        parts = []
        for op in ['<=', '>=', '<', '>', '=']:
            if op in mangled_str:
                relation_op = op
                parts = mangled_str.split(op)
                break
        if not parts: parts = [mangled_str]

        extra_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', mangled_str)
        combined_env = self.local_env.copy()
        reserved = self.allowed_funcs.union({'pi'})
        for token in set(extra_tokens):
            if token not in combined_env and token not in reserved:
                combined_env[token] = Symbol(token)

        transformations = (standard_transformations + (implicit_multiplication_application,))
        lhs = parse_expr(parts[0], local_dict=combined_env, transformations=transformations)
        rhs = parse_expr(parts[1], local_dict=combined_env, transformations=transformations) if len(parts) > 1 else 0

        full_expr = lhs - rhs
        linearized_expr = self._recursive_linearize(full_expr)

        return {
            "linear_expr": linearized_expr,
            "relation": relation_op if relation_op else "expression",
            "definitions": self.definitions,
            "constraints": self.constraints
        }

    def _recursive_linearize(self, expr):
        if expr.is_number: return expr
        if expr in self.param_aliases: return expr
        if expr in self.var_aliases: return expr

        if expr.is_Add:
            new_args = [self._recursive_linearize(arg) for arg in expr.args]
            return Add(*new_args)

        if expr.is_Mul:
            coeff, terms = expr.as_coeff_Mul()
            if terms == 1: return expr

            mul_args = list(terms.args) if terms.is_Mul else [terms]
            processed_args = [self._recursive_linearize(a) for a in mul_args]

            const_args = []
            var_args = []

            for arg in processed_args:
                is_const = False
                if arg.is_number:
                    is_const = True
                elif arg in self.param_aliases:
                    is_const = True
                elif arg.is_Symbol and arg not in self.var_aliases and not str(arg).startswith('y_temp'):
                    is_const = True

                if is_const:
                    const_args.append(arg)
                else:
                    var_args.append(arg)

            const_part = Mul(coeff, *const_args)

            if not var_args: return const_part
            if len(var_args) == 1: return Mul(const_part, var_args[0])

            current_term = var_args[0]
            for next_term in var_args[1:]:
                binary_expr = Mul(current_term, next_term, evaluate=False)
                current_term = self._create_atom(binary_expr)

            return Mul(const_part, current_term)

        if expr.is_Pow:
            base, exp_val = expr.args
            if exp_val.is_Number and exp_val < 0:
                self.constraints.append(f"{self._custom_format(base)} != 0")

            # 递归线性化底数和指数
            linear_base = self._recursive_linearize(base)
            linear_exp = self._recursive_linearize(exp_val)

            # 判断是否为 Param^Var 形式
            # Param 定义: 是数字 或者 在参数列表中
            is_base_const = (linear_base.is_number or linear_base in self.param_aliases)
            # Var 定义: 不是数字 且 不是参数
            is_exp_var = (not linear_exp.is_number and linear_exp not in self.param_aliases)

            # ---------------------------------------------------------
            # 变换逻辑 1: pow(Param, Var) -> exp(Var * log(Param))
            # ---------------------------------------------------------
            if is_base_const and is_exp_var:
                # 1. 构造 log(base)
                log_coeff = log(linear_base)

                # 2. 构造内部项: Var * log(Param)
                # 注意：这里 linear_exp 可能是原子变量，也可能是表达式（如果还没被原子化）
                inner_term = Mul(linear_exp, log_coeff)

                # 3. 将内部项原子化 -> y_temp_k = Var * log(Param)
                inner_var = self._create_atom(inner_term)

                # 4. 构造外部 exp -> exp(y_temp_k)
                outer_expr = exp(inner_var)

                # 5. 将外部 exp 原子化 -> y_temp_m = exp(y_temp_k)
                return self._create_atom(outer_expr)

            # ---------------------------------------------------------
            # 变换逻辑 2: 强制指数原子化
            # 如果指数是变量，必须是纯原子变量 (Symbol)，不能是表达式 (如 0.2*x)
            # ---------------------------------------------------------
            if not linear_exp.is_number and not isinstance(linear_exp, Symbol):
                # 如果 linear_exp 是 Mul, Add 等复杂项，强制创建一个中间变量
                linear_exp = self._create_atom(linear_exp)

            new_expr = Pow(linear_base, linear_exp, evaluate=False)
            return self._create_atom(new_expr)

        if expr.func in self.func_map:
            normalized = self._normalize_negative_args(expr.func, expr.args)
            if normalized is not None:
                return self._recursive_linearize(normalized)
            new_args = [self._recursive_linearize(arg) for arg in expr.args]
            new_expr = expr.func(*new_args)
            return self._create_atom(new_expr)

        raise ValueError(f"Unsupported function or structure: {expr}")

    def _get_new_var(self):
        name = f'y_temp_{self.y_counter}'
        self.y_counter += 1
        return Symbol(name)

    def _create_atom(self, expr):
        for var, definition in self.definitions.items():
            if definition == expr: return var
        new_var = self._get_new_var()
        self.definitions[new_var] = expr
        return new_var

    def print_ned_tree(self, expr, prefix="", is_last=True, is_root=True):
        if is_root:
            connector = ""
            new_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            new_prefix = prefix + ("    " if is_last else "│   ")

        node_text = ""
        children = []

        if isinstance(expr, Symbol) and expr in self.definitions:
            definition = self.definitions[expr]
            def_str = self._custom_format(definition)
            node_text = f"\033[94m{expr}\033[0m \033[90m[= {def_str}]\033[0m"
            if definition.args:
                children = list(definition.args)
            else:
                children = [definition]
        elif expr.args:
            node_text = f"\033[92m{expr.func.__name__}\033[0m"
            children = list(expr.args)
        else:
            node_text = self._custom_format(expr)
            children = []

        print(f"{prefix}{connector}{node_text}")
        count = len(children)
        for i, child in enumerate(children):
            self.print_ned_tree(child, new_prefix, i == count - 1, is_root=False)


def run_demo_advanced(expr_str, params, vars_list):
    builder = NEDTreeBuilder()
    builder.register_symbols(params, vars_list)

    print(f"Input: {expr_str}")
    print(f"Params: {params}")
    print(f"Variables: {vars_list}")

    try:
        result = builder.process(expr_str)

        print(f"Relation: {result['relation']}")
        if result['constraints']: print("Constraints:", result['constraints'])

        print("\n--- Definitions ---")
        # 按 y_temp_id 排序输出
        sorted_defs = sorted(result['definitions'].items(), key=lambda x: int(x[0].name.split('_')[-1]))
        for var, expr in sorted_defs:
            print(f"  {var} = {builder._custom_format(expr)}")

        final_str = builder._custom_format(result['linear_expr'])
        print(
            f"\n--- Final Linear Form ---\n{final_str} {result['relation'] if result['relation'] != 'expression' else ''} 0")

        print("\n" + "=" * 20 + " NED Tree " + "=" * 20)
        builder.print_ned_tree(result['linear_expr'])
        print("=" * 50 + "\n")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")





# ==========================================
# 示例场景
# ==========================================
if __name__ == '__main__':
    # 场景 1: 验证 pow(Param, Var) 的转换逻辑
    # 预期:
    # 1. cos(x_1) -> y_temp_1
    # 2. segments^y_temp_1 -> 转换为 exp(y_temp_1 * log(segments))
    # 3. y_temp_new = y_temp_1 * log(segments)
    # 4. y_temp_final = exp(y_temp_new)

    # print("\n" + "-" * 30 + " Test Case 1 " + "-" * 30)
    # expr1 = r'f >= 1 / (2 * pi) * (k / m)**0.5'
    # params1 = ['pi']
    # vars1 = ['m','f', 'k']
    #
    # run_demo_advanced(expr1, params1, vars1)
    #
    # # 场景 2: 验证 pow(Var, Expr) 的原子化逻辑
    # # 预期: pow(t, 0.2*t) -> 0.2*t 必须变成 y_temp_k, 然后 pow(t, y_temp_k)
    # expr2 = r'pow(t, 0.2*t) \le 10'
    # params2 = []
    # vars2 = ['t']
    #
    # print("\n" + "-" * 30 + " Test Case 2 " + "-" * 30)
    # run_demo_advanced(expr2, params2, vars2)
    #
    #
    # print("\n" + "-" * 30 + " Test Case 3 " + "-" * 30)
    # 测试用例
    # expr4 = r"At x_1^alpha x_2^beta"
    # params4 = [r'alpha', r'beta',r'At']
    # vars4 = ['x_1', 'x_2']
    # run_demo_advanced(expr4, params4, vars4)

    # print("\n" + "-" * 30 + " Test Case 4 " + "-" * 30)
    # # 测试 2: 简单的三变量连乘
    # expr3 = r'x*y*z'
    # params3 = []
    # vars3 = ['x', 'y', 'z']
    # run_demo_advanced(expr3, params3, vars3)



    params = [r'alpha', r'beta', r'gamma']
    vars_list = ['x_1', 'x_2', 'x_3']
    expr_str = r"alpha + beta * x_1**3 * exp(2*x_2 + x_1**2) + gamma * cos(log(x_3))>0"
    run_demo_advanced(expr_str, params, vars_list)


    # 使用示例
