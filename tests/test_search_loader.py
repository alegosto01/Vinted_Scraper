import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

existing_yaml = sys.modules.get("yaml")
if existing_yaml is not None and not hasattr(existing_yaml, "__file__"):
    sys.modules.pop("yaml", None)

try:
    import yaml  # noqa: F401
except ModuleNotFoundError:
    yaml = None

if yaml is not None:
    for module_name in list(sys.modules):
        if module_name == "config" or module_name.startswith("config."):
            sys.modules.pop(module_name, None)
    from config.search_loader import load_searches


@unittest.skipUnless(yaml is not None, "PyYAML is not installed in this environment")
class SearchLoaderTests(unittest.TestCase):
    def write_temp_yaml(self, content: str) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        try:
            tmp.write(content)
            return tmp.name
        finally:
            tmp.close()

    def test_load_searches_returns_empty_dict_for_empty_yaml(self):
        path = self.write_temp_yaml("")

        searches = load_searches(path)

        self.assertEqual(searches, {})

    def test_load_searches_rejects_non_mapping_root(self):
        path = self.write_temp_yaml("- search: ps5\n")

        with self.assertRaises(ValueError):
            load_searches(path)

    def test_load_searches_rejects_non_mapping_entry(self):
        path = self.write_temp_yaml("ps5: just-a-string\n")

        with self.assertRaises(ValueError):
            load_searches(path)

    def test_load_searches_ignores_deprecated_no_residential_key(self):
        path = self.write_temp_yaml(
            "ps5:\n"
            "  search: PlayStation 5\n"
            "  folder: ps5\n"
            "  no_residential: true\n"
        )

        searches = load_searches(path)

        self.assertIn("ps5", searches)
        self.assertFalse(hasattr(searches["ps5"], "no_residential"))
