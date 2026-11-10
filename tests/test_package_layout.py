"""Guard the symmetric V1/V2 layout, shared imports, module CLIs and notebooks."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def python_notebook_cell(source):
    """Replace IPython shell/magic statements, preserving surrounding Python."""
    result, continuation = [], False
    for line in source.splitlines(keepends=True):
        if continuation:
            continuation = line.rstrip().endswith("\\")
            result.append("\n")
        elif line.lstrip().startswith(("!", "%")):
            continuation = line.rstrip().endswith("\\")
            result.append(line[:len(line) - len(line.lstrip())] + "pass\n")
        else:
            result.append(line)
    return "".join(result)


class LayoutTests(unittest.TestCase):
    def test_version_filenames_are_symmetric(self):
        roles = {"model", "train", "evaluate", "learning", "transfer", "verify"}
        for version in ("v1", "v2"):
            actual = {path.name for path in (ROOT / version).glob("ai01_*.py")}
            self.assertEqual(actual, {f"ai01_{role}_{version}.py" for role in roles})
            self.assertTrue((ROOT / version / f"ai01_colab_{version}.ipynb").is_file())
            self.assertFalse((ROOT / version / f"ai01_targets_{version}.py").exists())

    def test_no_cross_version_or_shared_to_version_imports(self):
        for package in ("utils", "v1", "v2"):
            forbidden = {"v1", "v2"} if package == "utils" else {"v2" if package == "v1" else "v1"}
            for path in (ROOT / package).glob("ai01_*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    modules = ([node.module or ""] if isinstance(node, ast.ImportFrom) else
                               [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                    for module in modules:
                        self.assertNotIn(module.split(".")[0], forbidden, str(path))

    def test_v2_runtime_does_not_import_local_v1(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; import v2.ai01_train_v2, v2.ai01_evaluate_v2; "
            "assert not any(k == 'v1' or k.startswith('v1.') for k in sys.modules)",
        ], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_six_module_cli_help_commands(self):
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "USE_TF": "0"}
        for version in ("v1", "v2"):
            for role in ("train", "evaluate", "verify"):
                module = f"{version}.ai01_{role}_{version}"
                with self.subTest(module=module):
                    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                                            env=env, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_both_colab_notebooks_have_valid_sources_and_new_paths(self):
        for version in ("v1", "v2"):
            path = ROOT / version / f"ai01_colab_{version}.ipynb"
            notebook = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(notebook["nbformat"], 4)
            joined = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
            self.assertIn(f"-m {version}.ai01_train_{version}", joined)
            self.assertIn("from utils.ai01_data import", joined)
            self.assertIn("utils/ai01_requirements.txt", joined)
            self.assertNotIn("from ai01_data import", joined)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    compile(python_notebook_cell("".join(cell["source"])), str(path), "exec")

    def test_single_shared_requirements_and_original_rgb_root_paths(self):
        requirements = (ROOT / "utils" / "ai01_requirements_downstream.txt").read_text()
        self.assertIn("-r ai01_requirements.txt", requirements)
        for name in ("ai01_data.py", "ai01_targets.py", "ai01_metrics.py", "ai01_features.py",
                     "train_ai01.py", "ai01_v2_train.py", "ai01_transfer.py", "ai01_v2_transfer.py"):
            self.assertFalse((ROOT / name).exists(), name)
        for name in ("swin_mae.py", "swin_unet.py", "train.py", "visualize.py", "requirements.txt"):
            self.assertTrue((ROOT / name).is_file())
            self.assertFalse((ROOT / "v1" / "legacy_rgb" / name).exists())

    def test_original_rgb_imports_are_not_package_relative(self):
        for name in ("swin_mae.py", "swin_unet.py", "train.py", "visualize.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            tree = ast.parse(source)
            compile(tree, str(ROOT / name), "exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0, name)
        tree = ast.parse((ROOT / "swin_mae.py").read_text(encoding="utf-8"))
        self.assertTrue(any(isinstance(node, ast.ImportFrom) and node.module == "swin_unet"
                            for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
