import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_flowchart():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')

    # 定义样式
    box_style = "round,pad=0.5"
    props_io = dict(boxstyle="square,pad=0.5", facecolor="#E8F0FE", edgecolor="#1A73E8", lw=1.5)
    props_process = dict(boxstyle=box_style, facecolor="#F1F3F4", edgecolor="#5F6368", lw=1.5)
    props_decision = dict(boxstyle="square,pad=0.3", facecolor="#FFF3E0", edgecolor="#F57C00", lw=1.5)
    props_sub = dict(boxstyle=box_style, facecolor="#E6F4EA", edgecolor="#1E8E3E", lw=1.5)

    # 绘制节点框
    nodes = {
        "input": ("Input: Expr, Param_Set, Var_Set", (0.5, 0.95), props_io),
        "alias": ("Step 1: Symbol Aliasing & Preprocessing\n(Mangle String, Replace LaTeX)", (0.5, 0.85),
                  props_process),
        "parse": ("Step 2: Parse to AST\n full_expr = LHS - RHS", (0.5, 0.75), props_process),
        "linearize": ("Step 3: Recursive Linearization\nTraverse(node)", (0.5, 0.65), props_sub),

        # 分支
        "is_add": ("node is Add?", (0.2, 0.52), props_decision),
        "is_mul": ("node is Mul?", (0.5, 0.52), props_decision),
        "is_pow": ("node is Pow?", (0.8, 0.52), props_decision),

        # 操作
        "do_add": ("Return Add(*args)", (0.2, 0.40), props_process),
        "do_mul": ("Binarize Variables\nPairwise Sub-expressions", (0.5, 0.40), props_process),
        "do_pow": ("If Param^Var:\nConvert to exp(Var * log(Param))", (0.8, 0.40), props_process),

        "atomize": ("Step 4: Atomization & CSE\nCreate or Reuse y_temp_k", (0.5, 0.28), props_sub),
        "output": ("Output: Definition D_New, Linear Form l_F", (0.5, 0.15), props_io)
    }

    # 添加框并记录中心点位置
    bbox_centers = {}
    for key, (text, pos, props) in nodes.items():
        bbox = ax.text(pos[0], pos[1], text, ha="center", va="center", size=10, bbox=props, zorder=3,
                       weight='bold' if 'Input' in text or 'Output' in text else 'normal')
        bbox_centers[key] = pos

    # 定义连线箭头
    def add_arrow(start_key, end_key, connectionstyle="arc3,rad=0"):
        posA = bbox_centers[start_key]
        posB = bbox_centers[end_key]
        arrow = patches.FancyArrowPatch(posA, posB, arrowstyle='-|>', mutation_scale=15, color='#5F6368', lw=1.5,
                                        zorder=2, connectionstyle=connectionstyle)
        ax.add_patch(arrow)

    # 主干连接
    add_arrow("input", "alias")
    add_arrow("alias", "parse")
    add_arrow("parse", "linearize")

    # 树状分支连接 (使用直角线段更好看，这里用曲线示意)
    add_arrow("linearize", "is_add", connectionstyle="angle3,angleA=0,angleB=90")
    add_arrow("linearize", "is_mul")
    add_arrow("linearize", "is_pow", connectionstyle="angle3,angleA=0,angleB=90")

    add_arrow("is_add", "do_add")
    add_arrow("is_mul", "do_mul")
    add_arrow("is_pow", "do_pow")

    # 汇聚到原子化
    add_arrow("do_add", "atomize", connectionstyle="angle3,angleA=-90,angleB=0")
    add_arrow("do_mul", "atomize")
    add_arrow("do_pow", "atomize", connectionstyle="angle3,angleA=-90,angleB=0")

    add_arrow("atomize", "output")

    plt.title("Algorithm Flow of NEDTree Generation", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("nedtree_flowchart.png", dpi=300, bbox_inches='tight')
    plt.show()


draw_flowchart()





