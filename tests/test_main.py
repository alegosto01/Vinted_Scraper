import sys
import types
import unittest

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

sys.modules.setdefault('yaml', types.SimpleNamespace(safe_load=lambda *args, **kwargs: {}))
sys.modules.setdefault('dotenv', types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))
sys.modules.setdefault('scraping_options', types.ModuleType('scraping_options'))

import main as main_module


class MainTests(unittest.TestCase):
    def test_main_function_exists(self):
        self.assertTrue(callable(main_module.main))


if __name__ == '__main__':
    unittest.main()
