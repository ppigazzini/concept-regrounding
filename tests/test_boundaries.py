import ast
from pathlib import Path

PACKAGE = Path(__file__).parent.parent / "regrounding"


def _imported(path: Path) -> set[str]:
    """Return every module ``path`` imports, by ``import x`` and by ``from x import y``."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
    return out


class TestBoundaries:
    def test_only_the_boundary_module_imports_valgebra(self):
        for path in PACKAGE.rglob("*.py"):
            if path.name == "_valgebra.py":
                continue
            assert "valgebra" not in {name.split(".")[0] for name in _imported(path)}, path

    def test_the_environment_module_imports_os_and_nothing_else(self):
        assert _imported(PACKAGE / "reproducible.py") == {"os"}

    def test_a_from_import_counts_as_an_import(self, tmp_path):
        module = tmp_path / "module.py"
        source = "import os\nfrom torch import nn\nfrom . import model\n"
        module.write_text(source, encoding="utf-8")
        assert _imported(module) == {"os", "torch", "."}
