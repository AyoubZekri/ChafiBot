import ast
import operator

from langchain_core.tools import tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    # تقييم آمن: أرقام وعمليات حسابية فقط (لا eval)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("أس كبير جداً")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("عملية غير مسموحة")


def _format(value):
    # تنسيق جزائري: 100.000,00
    text = f"{value:,.2f}"
    return text.replace(",", " ").replace(".", ",").replace(" ", ".")


@tool
def calculate(expression: str) -> str:
    """احسب عملية حسابية بدقة. استعملها لكل حساب (غرامات، نسب، ضرائب).
    مثال: "250000 * 10 / 100" أو "(1200000 - 300000) * 0.19".
    الرموز المسموحة: الأرقام و + - * / % ** والأقواس."""
    expression = expression.replace("×", "*").replace("÷", "/").replace(",", ".")
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
    except ZeroDivisionError:
        return "خطأ: قسمة على صفر"
    except (ValueError, SyntaxError) as e:
        return f"خطأ في العملية: {e}"
    return f"{expression} = {_format(result)}"
