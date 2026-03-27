from __future__ import annotations

import importlib
import unittest


class LazyImportSmokeTest(unittest.TestCase):
    def test_base_trainer_import_does_not_require_peft(self) -> None:
        module = importlib.import_module("train.base_trainer")
        self.assertTrue(hasattr(module, "BaseDiffusionTrainer"))

    def test_generator_import_is_lazy(self) -> None:
        module = importlib.import_module("infer.generator")
        self.assertTrue(hasattr(module, "run_stable_diffusion_inference"))


if __name__ == "__main__":
    unittest.main()
