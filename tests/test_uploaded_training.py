"""Upload experiment safeguards, independent of research artifacts."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'deployment')]
from training import train_uploaded, split_uploaded, evaluation_metrics, validate_training, export_trained
from model_registry import load_model

class TrainingTests(unittest.TestCase):
    def setUp(self):
        n=120
        rng=np.random.default_rng(10)
        self.frame=pd.DataFrame(dict(solar_irradiance=rng.uniform(0,1000,n),temperature=rng.uniform(20,40,n),humidity=rng.uniform(20,90,n),hour=rng.uniform(0,24,n),current=rng.uniform(0,10,n),voltage=rng.uniform(20,40,n),power=rng.uniform(0,1000,n),day_id=np.repeat(np.arange(12),10),timestamp=pd.date_range('2025-01-01',periods=n,freq='6h')))

    def test_splits_and_scalers_baseline(self):
        model,result,_,_=train_uploaded(self.frame,family='linear',strategy='grouped',group_column='day_id')
        tr,te,meta=split_uploaded(self.frame,'grouped','day_id')
        self.assertFalse(set(meta['training_groups']) & set(meta['evaluation_groups']))
        np.testing.assert_allclose(model.estimator[0].mean_, self.frame.loc[tr,model.spec.features].mean())
        self.assertEqual(result.source_row.tolist(),(te+2).tolist())
        model,result,_,_=train_uploaded(self.frame,family='tuned',epochs=2,baseline=True)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        tr,te,_=split_uploaded(self.frame)
        baseline=make_pipeline(StandardScaler(),LinearRegression()).fit(self.frame.loc[tr,model.spec.features].to_numpy(),self.frame.loc[tr,'power'])
        np.testing.assert_allclose(result.linear_prediction_w,baseline.predict(self.frame.loc[te,model.spec.features].to_numpy()))

    def test_chronological_and_invalid(self):
        tr,te,_=split_uploaded(self.frame,'chronological')
        self.assertLess(self.frame.loc[tr,'timestamp'].max(),self.frame.loc[te,'timestamp'].min())
        for bad in ['12:00','01/02/2025','bad']:
            with self.assertRaises(ValueError):
                split_uploaded(self.frame.assign(timestamp=bad),'chronological')
        frame=self.frame.copy()
        frame.loc[0,'power']=-1
        frame.loc[1,'humidity']=101
        frame.loc[2,'temperature']=-300
        model,_,rejected,_=train_uploaded(frame,family='linear')
        self.assertEqual(len(rejected),3)
        self.assertTrue(rejected.training_issue.str.len().gt(0).all())

    def test_metrics_and_unsupported_continuation(self):
        metrics=evaluation_metrics([0,2],[1,-2])
        self.assertEqual(metrics['MAE'],2.5)
        self.assertAlmostEqual(metrics['RMSE'],np.sqrt(8.5))
        self.assertEqual(metrics['MAPE'],200)
        self.assertEqual(metrics['WAPE'],250)
        self.assertEqual(metrics['R2'],-7.5)
        self.assertIsNone(evaluation_metrics([0,0],[1,2])['WAPE'])
        self.assertIsNone(evaluation_metrics([0,0],[1,2])['MAPE'])
        model,*_=train_uploaded(self.frame,family='linear')
        with self.assertRaisesRegex(ValueError,'Continued training'):
            train_uploaded(self.frame,model,training_mode='continue')

    def test_exports_reload_all_families_and_weights(self):
        for family in ('linear','tuned','pytorch','tensorflow'):
            with self.subTest(family=family):
                original,result,rejected,_=train_uploaded(self.frame,family=family,epochs=1)
                raw=export_trained(original,result,rejected)
                with TemporaryDirectory() as folder:
                    with ZipFile(BytesIO(raw)) as archive:
                        self.assertIn('held_out_predictions.csv',archive.namelist())
                        self.assertIn('evaluation_summary.json',archive.namelist())
                        archive.extractall(folder)
                    loaded=load_model(original.spec.key,folder)
                    np.testing.assert_allclose(loaded.predict(self.frame),original.predict(self.frame),rtol=1e-5)
                if family not in ('pytorch','tensorflow'):
                    continue
                before=original.predict(self.frame)
                scaler_mean=original.estimator[1].mean_.copy()
                # No optimizer updates: copied model must give identical pretrained predictions.
                target='torch.optim.Adam.step' if family=='pytorch' else 'keras.src.optimizers.adam.Adam.update_step'
                with patch(target):
                    copied,*_=train_uploaded(self.frame,original,training_mode='continue',epochs=1)
                np.testing.assert_allclose(copied.predict(self.frame),before,rtol=1e-5)
                self.assertIsNot(copied.estimator[0],original.estimator[0])
                adapted,*_=train_uploaded(self.frame,original,training_mode='continue',epochs=2)
                self.assertFalse(np.allclose(adapted.predict(self.frame),before))
                np.testing.assert_array_equal(original.predict(self.frame),before)
                np.testing.assert_array_equal(adapted.estimator[1].mean_,scaler_mean)
                fresh,*_=train_uploaded(self.frame,family=family,epochs=1,seed=12)
                self.assertFalse(np.allclose(fresh.predict(self.frame),before))

    def test_internal_validation_boundaries_and_scaler_isolation(self):
        for strategy, group in [('random', None), ('grouped', 'day_id'), ('chronological', None)]:
            with self.subTest(strategy=strategy):
                model, result, _, _ = train_uploaded(self.frame, family='pytorch', epochs=1,
                    strategy=strategy, group_column=group, baseline=True)
                meta = model.metadata
                fitting, validation, evaluation = [np.array(meta[key]) - 2 for key in
                    ('fitting_source_rows', 'early_stopping_source_rows', 'evaluation_source_rows')]
                self.assertFalse(set(fitting) & set(validation))
                self.assertFalse(set(fitting) & set(evaluation))
                self.assertFalse(set(validation) & set(evaluation))
                np.testing.assert_allclose(model.estimator[1].mean_, self.frame.loc[fitting, model.spec.features].mean())
                np.testing.assert_allclose(model.estimator[2].mean_, [self.frame.loc[fitting, 'power'].mean()])
                if strategy == 'grouped':
                    sets = [set(self.frame.loc[idx, 'day_id']) for idx in (fitting, validation, evaluation)]
                    self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
                if strategy == 'chronological':
                    self.assertLess(self.frame.loc[fitting, 'timestamp'].max(), self.frame.loc[validation, 'timestamp'].min())
                    self.assertLess(self.frame.loc[validation, 'timestamp'].max(), self.frame.loc[evaluation, 'timestamp'].min())
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LinearRegression
                baseline = make_pipeline(StandardScaler(), LinearRegression()).fit(
                    self.frame.loc[fitting, model.spec.features].to_numpy(), self.frame.loc[fitting, 'power'])
                np.testing.assert_allclose(result.linear_prediction_w, baseline.predict(self.frame.loc[evaluation, model.spec.features].to_numpy()))

    def test_early_stopping_restores_best_for_both_modes_and_backends(self):
        from training import EarlyStoppingState
        update = EarlyStoppingState.update
        # Force validation to worsen after epoch one while weights keep changing.
        def worsening(state, epoch, loss, snapshot):
            return update(state, epoch, float(epoch), snapshot)
        for family in ('pytorch', 'tensorflow'):
            original, _, _, _ = train_uploaded(self.frame, family=family, epochs=1)
            before = original.predict(self.frame)
            for source in (None, original):
                options = dict(family=family, original=source,
                               training_mode='continue' if source else 'new')
                one, _, _, _ = train_uploaded(self.frame, epochs=1, **options)
                with patch('training.EarlyStoppingState.update', new=worsening):
                    stopped, result, rejected, _ = train_uploaded(self.frame, epochs=1000, patience=1, **options)
                    full, _, _, _ = train_uploaded(self.frame, epochs=3, early_stopping=False, **options)
                self.assertEqual(stopped.metadata['requested_max_epochs'], 1000)
                self.assertEqual(stopped.metadata['completed_epochs'], 2)
                self.assertEqual(stopped.metadata['best_epoch'], 1)
                self.assertTrue(stopped.metadata['early_stopped'])
                self.assertEqual(full.metadata['completed_epochs'], 3)
                self.assertFalse(full.metadata['early_stopped'])
                np.testing.assert_allclose(stopped.predict(self.frame), one.predict(self.frame), rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(full.predict(self.frame), one.predict(self.frame), rtol=1e-5, atol=1e-5)
                np.testing.assert_array_equal(original.predict(self.frame), before)
                self.assertEqual(len(stopped.metadata['validation_loss']), 2)
                with TemporaryDirectory() as folder:
                    with ZipFile(BytesIO(export_trained(stopped, result, rejected))) as archive:
                        archive.extractall(folder)
                    reloaded = load_model(stopped.spec.key, folder)
                    self.assertEqual(reloaded.metadata['best_epoch'], 1)
                    np.testing.assert_allclose(reloaded.predict(self.frame), stopped.predict(self.frame), rtol=1e-5)

    def test_outer_evaluation_values_do_not_affect_training_or_stopping(self):
        options = dict(family='pytorch', epochs=5, strategy='grouped', group_column='day_id')
        first, _, _, _ = train_uploaded(self.frame, **options)
        changed = self.frame.copy()
        heldout = np.array(first.metadata['evaluation_source_rows']) - 2
        changed.loc[heldout, 'power'] = 1e8
        changed.loc[heldout, 'solar_irradiance'] = 1e8
        second, _, _, _ = train_uploaded(changed, **options)
        self.assertEqual(first.metadata['validation_loss'], second.metadata['validation_loss'])
        self.assertEqual(first.metadata['best_epoch'], second.metadata['best_epoch'])
        np.testing.assert_array_equal(first.predict(self.frame), second.predict(self.frame))

    def test_early_stopping_settings_and_small_groups(self):
        from inspect import signature
        from training import EarlyStoppingState
        defaults = signature(train_uploaded).parameters
        for name, value in [('epochs', 200), ('early_stopping', True), ('patience', 30), ('min_delta', .0001)]:
            self.assertEqual(defaults[name].default, value)
        state = EarlyStoppingState(patience=2, min_delta=.1)
        self.assertFalse(state.update(1, 1., lambda: 'first'))
        self.assertFalse(state.update(2, .99, lambda: 'second'))
        self.assertTrue(state.update(3, .98, lambda: 'third'))
        self.assertEqual(state.best_weights, 'third')
        for options in ({'epochs':1001}, {'patience':0}, {'min_delta':-1}, {'min_delta':np.nan}):
            with self.assertRaises(ValueError):
                train_uploaded(self.frame, family='pytorch', **options)
        with self.assertRaisesRegex(ValueError, 'training-only validation'):
            train_uploaded(self.frame.assign(day_id=np.repeat([0,1],60)), family='pytorch', strategy='grouped', group_column='day_id')

    def test_large_installation(self):
        frame=pd.concat([self.frame]*100,ignore_index=True)
        model,result,_,_=train_uploaded(frame,family='linear')
        self.assertEqual(model.metadata['valid_rows'],12000)
        self.assertEqual(len(result),2400)

if __name__=='__main__':
    unittest.main()
