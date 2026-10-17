"""Detect simple import-based dependencies between Python task scripts."""

import ast
import tokenize
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple


class DependencyAnalysisError(ValueError):
    """Raised when a task script cannot be parsed for static imports."""


def detect_import_dependencies(
    scripts: Mapping[str, Path],
    available_script_names: Iterable[str],
) -> Dict[str, Tuple[str, ...]]:
    """Map each script to other available scripts it statically imports.

    Only direct ``import module`` and ``from module import name`` statements
    whose module exactly matches another script's stem are considered.
    """
    script_by_module = {
        Path(name).stem: name
        for name in available_script_names
        if Path(name).suffix.lower() == ".py"
    }
    dependencies: Dict[str, Tuple[str, ...]] = {}

    for script_name, script_path in scripts.items():
        try:
            with tokenize.open(script_path) as source_file:
                tree = ast.parse(source_file.read(), filename=script_name)
        except (OSError, SyntaxError, UnicodeError) as error:
            raise DependencyAnalysisError(
                f"cannot analyze imports in {script_name!r}: {error}"
            ) from error

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_modules.add(node.module)

        parents = {
            script_by_module[module]
            for module in imported_modules
            if module in script_by_module and script_by_module[module] != script_name
        }
        dependencies[script_name] = tuple(sorted(parents))

    return dependencies
