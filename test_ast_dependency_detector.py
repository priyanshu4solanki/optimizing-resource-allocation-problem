"""Tests for simple import-based static task dependency detection."""

import tempfile
import unittest
from pathlib import Path

from ast_dependency_detector import (
    DependencyAnalysisError,
    detect_import_dependencies,
)


class AstDependencyDetectorTest(unittest.TestCase):
    def test_detects_imports_matching_other_script_stems(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            provider = root / "provider.py"
            consumer = root / "consumer.py"
            provider.write_text("def load():\n    return 1\n", encoding="utf-8")
            consumer.write_text(
                "import os\nimport provider as data\n"
                "from provider import load\n"
                "def run():\n    import missing_at_runtime\n",
                encoding="utf-8",
            )

            dependencies = detect_import_dependencies(
                {"consumer.py": consumer, "provider.py": provider},
                {"consumer.py", "provider.py"},
            )

        self.assertEqual(dependencies["consumer.py"], ("provider.py",))
        self.assertEqual(dependencies["provider.py"], ())

    def test_does_not_match_partial_module_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "consumer.py"
            source.write_text("import provider_utils\n", encoding="utf-8")

            dependencies = detect_import_dependencies(
                {"consumer.py": source},
                {"consumer.py", "provider.py"},
            )

        self.assertEqual(dependencies["consumer.py"], ())

    def test_reports_invalid_python_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "broken.py"
            source.write_text("if:\n", encoding="utf-8")

            with self.assertRaises(DependencyAnalysisError):
                detect_import_dependencies(
                    {"broken.py": source},
                    {"broken.py"},
                )


if __name__ == "__main__":
    unittest.main()
