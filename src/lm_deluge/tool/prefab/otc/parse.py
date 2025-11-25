import ast

SAFE_BUILTINS = {
    # Types
    "bool": bool,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "frozenset": frozenset,
    "type": type,
    # Functions
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "chr": chr,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "format": format,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,  # Captured for output
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "slice": slice,
    "sorted": sorted,
    "sum": sum,
    "zip": zip,
    # Constants
    "True": True,
    "False": False,
    "None": None,
    # Exceptions (for try/except)
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
}

# AST nodes that are NOT allowed
FORBIDDEN_NODES = {
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.AsyncWith,  # We control async, not user code
    ast.Yield,
    ast.YieldFrom,
    ast.ClassDef,  # No class definitions
}

# Forbidden function calls
FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "breakpoint",
    "exit",
    "quit",
}

# Forbidden attribute access patterns
FORBIDDEN_ATTRIBUTES = {
    "__class__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__code__",
    "__globals__",
    "__builtins__",
    "__import__",
    "__dict__",
    "__module__",
    "__reduce__",
    "__reduce_ex__",
}


class OTCSecurityError(Exception):
    """Raised when code violates OTC security constraints."""

    pass


class OTCExecutionError(Exception):
    """Raised when code execution fails."""

    pass


class ASTValidator(ast.NodeVisitor):
    """Validates that an AST doesn't contain forbidden constructs."""

    def __init__(self, allowed_tool_names: set[str]):
        self.allowed_tool_names = allowed_tool_names
        self.errors: list[str] = []

    def visit(self, node: ast.AST) -> None:
        # Check for forbidden node types
        if type(node) in FORBIDDEN_NODES:
            self.errors.append(
                f"Forbidden construct: {type(node).__name__} at line {getattr(node, 'lineno', '?')}"
            )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check for forbidden function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                self.errors.append(
                    f"Forbidden function call: {node.func.id} at line {node.lineno}"
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Check for forbidden attribute access
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.errors.append(
                f"Forbidden attribute access: {node.attr} at line {node.lineno}"
            )

        # Also check for dunder access patterns
        if node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr not in {"__len__", "__iter__", "__next__", "__contains__"}:
                self.errors.append(
                    f"Forbidden dunder access: {node.attr} at line {node.lineno}"
                )

        self.generic_visit(node)

    def validate(self, tree: ast.AST) -> list[str]:
        """Validate the AST and return list of errors."""
        self.errors = []
        self.visit(tree)
        return self.errors


def validate_code(code: str, allowed_tool_names: set[str]) -> ast.Module:
    """Parse and validate code, returning AST if valid."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise OTCSecurityError(f"Syntax error: {e}")

    validator = ASTValidator(allowed_tool_names)
    errors = validator.validate(tree)

    if errors:
        raise OTCSecurityError(
            "Security violations:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return tree
