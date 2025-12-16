"""
RLM (Recursive Language Model) code parsing and validation.

Extends OTC's security model with additional modules for context analysis.
"""

import ast
import collections
import json
import math
import re

# Import OTC's base security definitions
from ..otc.parse import (
    FORBIDDEN_CALLS,
    SAFE_BUILTINS,
    ASTValidator,
    OTCSecurityError,
)

# RLM uses the same builtins as OTC
RLM_SAFE_BUILTINS = SAFE_BUILTINS.copy()

# Modules available in RLM - imports of these are stripped (no-ops)
RLM_ALLOWED_IMPORTS = {"re", "math", "collections", "json"}

# Modules and common imports available in RLM (injected into globals)
RLM_MODULES = {
    # Full modules
    "re": re,
    "math": math,
    "collections": collections,
    "json": json,
    # Common imports from collections
    "Counter": collections.Counter,
    "defaultdict": collections.defaultdict,
    "deque": collections.deque,
    "namedtuple": collections.namedtuple,
    "OrderedDict": collections.OrderedDict,
}


class RLMSecurityError(OTCSecurityError):
    """Raised when RLM code violates security constraints."""

    pass


class RLMExecutionError(Exception):
    """Raised when RLM code execution fails."""

    pass


class RLMASTValidator(ASTValidator):
    """Validates RLM code with additional checks.

    Import statements for allowed modules are stripped (no-ops).
    Imports of disallowed modules raise errors.
    """

    def __init__(self, allowed_names: set[str] | None = None):
        super().__init__(allowed_tool_names=set())
        self.allowed_names = allowed_names or set()

    def visit(self, node: ast.AST) -> None:
        # Check imports - allowed ones will be stripped later, disallowed ones error
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in RLM_ALLOWED_IMPORTS:
                    self.errors.append(
                        f"Forbidden import: {alias.name} at line {node.lineno}. "
                        f"Available modules: {', '.join(sorted(RLM_ALLOWED_IMPORTS))}"
                    )
            self.generic_visit(node)
            return

        if isinstance(node, ast.ImportFrom):
            if node.module not in RLM_ALLOWED_IMPORTS:
                self.errors.append(
                    f"Forbidden import: from {node.module} at line {node.lineno}. "
                    f"Available modules: {', '.join(sorted(RLM_ALLOWED_IMPORTS))}"
                )
            self.generic_visit(node)
            return

        # For all other nodes, use parent validation
        super().visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                self.errors.append(
                    f"Forbidden function call: {node.func.id} at line {node.lineno}"
                )
        self.generic_visit(node)


class ImportStripper(ast.NodeTransformer):
    """Strips import statements for allowed modules from the AST."""

    def visit_Import(self, node: ast.Import) -> ast.AST | None:
        # Keep only imports of non-allowed modules (which will error at validation)
        remaining = [
            alias for alias in node.names if alias.name not in RLM_ALLOWED_IMPORTS
        ]
        if not remaining:
            return None  # Remove the entire import statement
        node.names = remaining
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
        # Strip imports from allowed modules
        if node.module in RLM_ALLOWED_IMPORTS:
            return None  # Remove the entire import statement
        return node


def validate_rlm_code(code: str) -> ast.Module:
    """Parse and validate RLM code, returning AST if valid.

    Import statements for allowed modules (re, math, collections, json) are
    stripped from the AST since these modules are already in the namespace.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise RLMSecurityError(f"Syntax error: {e}")

    # Validate first (before stripping)
    validator = RLMASTValidator()
    errors = validator.validate(tree)

    if errors:
        raise RLMSecurityError(
            "Security violations:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Strip allowed imports (they're no-ops since modules are pre-loaded)
    stripper = ImportStripper()
    tree = stripper.visit(tree)
    ast.fix_missing_locations(tree)

    return tree
