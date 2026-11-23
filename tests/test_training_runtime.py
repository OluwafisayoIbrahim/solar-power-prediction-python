"""Regression for a stale imported trainer during a Streamlit hot rerun."""
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
import inspect
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "deployment"), str(ROOT / "src")]
from training_runtime import load_training_backend, _load_source


@contextmanager
def replaced_modules(replacements):
    # Restore only our substitutions. patch.dict(sys.modules) removes freshly
    # imported TensorFlow extensions and can crash a later native re-import.
    originals = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class TrainingRuntimeTests(unittest.TestCase):
    def test_stale_import_does_not_supply_the_backend(self):
        stale = ModuleType("training")
        stale.train_uploaded = lambda frame: None
        with replaced_modules({"training": stale}):
            backend = load_training_backend()
            self.assertIsNot(backend, stale)
            for option in ("early_stopping", "patience", "min_delta"):
                self.assertIn(option, inspect.signature(backend.train_uploaded).parameters)
            # All families accept the UI's stopping options, including non-ANNs.
            import pandas as pd
            import numpy as np
            frame = pd.DataFrame(dict(solar_irradiance=np.arange(20.) + 100,
                temperature=np.arange(20.) + 20, humidity=50., hour=12.,
                power=np.arange(20.) * 2))
            for family in ("linear", "tuned", "pytorch", "tensorflow"):
                with self.subTest(family=family):
                    fitted, _, _, _ = backend.train_uploaded(frame, family=family,
                        epochs=1, early_stopping=True, patience=30, min_delta=.0001)
                    self.assertEqual(fitted.metadata["model_family"], family)

    def test_stale_spreadsheet_reader_is_not_reused(self):
        import batch_prediction
        stale = ModuleType('batch_prediction')
        stale.__dict__.update(batch_prediction.__dict__)
        stale.read_sheet = lambda data, filename, sheet=0, features=None: None
        stale.download_csv = lambda result: b'stale'
        with replaced_modules({'batch_prediction': stale}):
            path = ROOT / 'deployment/batch_prediction.py'
            backend = _load_source(path.read_bytes(), str(path))
            frame = backend.read_sheet(b'GHI,AC_Power\n123,45\n', 'data.csv',
                                       features=[], preserve_columns=True)
            self.assertEqual(frame.columns.tolist(), ['GHI', 'AC_Power'])
            self.assertNotEqual(load_training_backend().download_csv(frame), b'stale')

    def test_coherent_runtime_ignores_stale_registry_and_all_upload_modules(self):
        from session_runtime import load_session_runtime, _load_bundle, MODULES
        import pandas as pd
        from types import SimpleNamespace
        stale = {name: ModuleType(name) for name in MODULES}
        # Reproduce an old constructor without the input_features field.
        class LegacySpec:
            def __init__(self, key, label, track, kind):
                self.key, self.label, self.track, self.kind = key, label, track, kind
                self.features = ['wind_speed']
        stale['model_registry'].ModelSpec = LegacySpec
        _load_bundle.cache_clear()
        with replaced_modules(stale):
            runtime = load_session_runtime()
            self.assertIs(runtime.training.ModelSpec, runtime.model_registry.ModelSpec)
            self.assertIs(runtime.training.download_csv, runtime.batch_prediction.download_csv)
            self.assertIs(runtime.mapping_ui.validate_config, runtime.feature_mapping.validate_config)
            spec = runtime.model_registry.ModelSpec('custom', 'Custom', 'Custom', 'linear',
                                                   input_features=('wind_speed',))
            self.assertEqual(spec.features, ['wind_speed'])
            frame = runtime.batch_prediction.read_sheet(
                b'Wind,Output\n' + b''.join(f'{i},{2*i}\n'.encode() for i in range(20)),
                'data.csv', features=[], preserve_columns=True)
            config = dict(schema_version=1, preset='Custom', features=[
                dict(source_column='Wind', feature_id='wind_speed', display_name='Wind', unit='m/s')],
                target=dict(source_column='Output', display_name='Output', unit='W'))
            model, _, _, _ = runtime.training.train_uploaded(frame, family='linear',
                feature_config=config, epochs=1, early_stopping=True)
            self.assertEqual(model.spec.features, ['wind_speed'])
            source, _, _, _ = runtime.training.train_uploaded(frame, family='pytorch',
                feature_config=config, epochs=1)
            legacy = SimpleNamespace(spec=LegacySpec('old', 'Old model', 'Custom', 'pytorch'),
                                     estimator=source.estimator, metadata=source.metadata)
            continued, _, _, _ = runtime.training.train_uploaded(frame, original=legacy,
                feature_config=config, training_mode='continue', epochs=1)
            self.assertEqual(continued.spec.features, ['wind_speed'])
            self.assertIsInstance(continued.spec, runtime.model_registry.ModelSpec)
            self.assertIs(runtime, load_session_runtime())

    def test_registry_change_versions_the_entire_dependency_graph(self):
        from session_runtime import _load_bundle, MODULES
        folder = ROOT / 'deployment'
        sources = tuple((name, (folder / (name + '.py')).read_bytes()) for name in MODULES)
        first = _load_bundle(sources, str(folder))
        updated = tuple((name, source + b'\n# updated registry\n' if name == 'model_registry' else source)
                        for name, source in sources)
        second = _load_bundle(updated, str(folder))
        self.assertNotEqual(first.version, second.version)
        self.assertIsNot(first.model_registry.ModelSpec, second.model_registry.ModelSpec)
        self.assertIs(second.training.ModelSpec, second.model_registry.ModelSpec)
        self.assertIs(first.training.ModelSpec, first.model_registry.ModelSpec)
        self.assertIsNot(first.training, second.training)

    def test_source_changes_produce_independent_modules(self):
        first = _load_source(b"value = 1\ndef read(): return value\n", "trainer.py")
        second = _load_source(b"value = 2\ndef read(): return value\n", "trainer.py")
        self.assertEqual(first.read(), 1)
        self.assertEqual(second.read(), 2)
        self.assertIs(first, _load_source(b"value = 1\ndef read(): return value\n", "trainer.py"))


if __name__ == "__main__":
    unittest.main()
