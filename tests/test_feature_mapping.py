"""Dynamic feature matrices, reusable mappings and isolated model packages."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import sys
import unittest

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'deployment'), str(ROOT / 'src')]
from feature_mapping import (validate_config, load_mapping, dump_mapping, map_frame,
                             preset_config, mapping_warnings)
from training import train_uploaded, export_trained
from model_registry import load_session_model, load_model
from batch_prediction import read_sheet, run_batch, download_csv


def example():
    rng = np.random.default_rng(32)
    frame = pd.DataFrame({'GHI': rng.uniform(0, 1000, 60), 'AmbientTemp': rng.uniform(10, 40, 60),
        'WindSpeed': rng.uniform(0, 10, 60), 'AC_Power': rng.uniform(0, 10, 60),
        'Day': np.repeat(np.arange(6), 10), 'Time': pd.date_range('2026-01-01', periods=60, freq='3h'),
        'Bad': ['category'] * 60})
    config = dict(schema_version=1, preset='Custom', features=[
        dict(source_column='GHI', feature_id='solar_irradiance', display_name='Solar Irradiance', unit='W/m2'),
        dict(source_column='AmbientTemp', feature_id='temperature', display_name='Ambient temperature', unit='deg C'),
        dict(source_column='WindSpeed', feature_id='wind_speed', display_name='Wind Speed', unit='m/s')],
        target=dict(source_column='AC_Power', display_name='AC Power Output', unit='kW'),
        grouping_column='Day', timestamp_column='Time')
    return frame, config


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.config = example()

    def test_original_headers_csv_excel_and_json_reuse(self):
        original = self.frame.copy(deep=True)
        raw = read_sheet(self.frame.to_csv(index=False).encode(), 'site.csv', features=[], preserve_columns=True)
        self.assertIn('GHI', raw)
        reserved = self.frame.rename(columns={'WindSpeed': 'predicted_value'})
        self.assertIn('predicted_value', read_sheet(reserved.to_csv(index=False).encode(),
            'headers.csv', features=[], preserve_columns=True))
        output = BytesIO()
        self.frame.to_excel(output, index=False)
        self.assertIn('AC_Power', read_sheet(output.getvalue(), 'site.xlsx', features=[], preserve_columns=True))
        self.assertEqual(load_mapping(dump_mapping(self.config).encode(), self.frame.columns), self.config)
        mapped = map_frame(self.frame, self.config)
        self.assertEqual(list(mapped.columns), ['solar_irradiance', 'temperature', 'wind_speed', 'power', 'day_id', 'timestamp'])
        pd.testing.assert_frame_equal(original, self.frame)
        standard = preset_config(['GHI', 'AmbientTemp', 'RH', 'hour', 'AC_Power'], 'B')
        self.assertEqual(standard['features'][0]['source_column'], 'GHI')
        self.assertEqual(standard['target']['source_column'], 'AC_Power')
        validate_config(standard)

    def test_invalid_mappings_leakage_and_missing_columns(self):
        variants=[]
        for field in ('feature_id','source_column'):
            cfg=deepcopy(self.config)
            cfg['features'][1][field]=cfg['features'][0][field]
            variants.append(cfg)
        for role in ('target','grouping_column','timestamp_column'):
            cfg=deepcopy(self.config)
            if role=='target':
                cfg[role]['source_column']='GHI'
            else:
                cfg[role]='GHI'
            variants.append(cfg)
        variants += [dict(self.config, features=[])]
        for cfg in variants:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_config(cfg,self.frame.columns)
        with self.assertRaisesRegex(ValueError,'missing columns'):
            validate_config(self.config, ['GHI'])
        with self.assertRaisesRegex(ValueError,'no numerical'):
            cfg=deepcopy(self.config)
            cfg['features'][0]['source_column']='Bad'
            map_frame(self.frame,cfg)
        with self.assertRaises(ValueError):
            load_mapping(b'not json')
        with self.assertRaises(ValueError):
            load_mapping(b' '* (1024*1024+1))
        with self.assertRaises(ValueError):
            load_mapping(b'{"schema_version": 100}')

    def test_custom_add_remove_rename_and_invalid_observations(self):
        for count in (1,2,3):
            cfg=deepcopy(self.config)
            cfg['features']=cfg['features'][:count]
            cfg['features'][0]['display_name']='Romanian GHI'
            frame=self.frame.copy()
            frame.loc[0,'GHI']=np.inf
            frame.loc[1,'AC_Power']=np.nan
            model,_,rejected,_=train_uploaded(frame,family='linear',track='Custom',feature_config=cfg)
            self.assertEqual(model.estimator.n_features_in_,count)
            self.assertEqual(model.metadata['feature_config']['features'][0]['display_name'],'Romanian GHI')
            self.assertEqual(len(rejected),2)

    def test_all_families_custom_dimensions_baseline_reload_and_predictions(self):
        for family in ('tensorflow','pytorch','tuned','linear'):
            with self.subTest(family=family):
                model,result,rejected,_=train_uploaded(self.frame,family=family,track='Custom',
                    feature_config=self.config,epochs=1,baseline=True,strategy='grouped',group_column='day_id')
                self.assertEqual(model.spec.features,['solar_irradiance','temperature','wind_speed'])
                self.assertEqual(model.metadata['target']['unit'],'kW')
                expected=model.predict(self.frame)
                np.testing.assert_allclose(model.predict(self.frame[self.frame.columns[::-1]]),expected)
                canonical=map_frame(self.frame,self.config)
                np.testing.assert_allclose(model.predict(canonical),expected)
                output=run_batch(canonical,model,features=model.spec.features,predictor=lambda m,f:m.predict(f))
                np.testing.assert_allclose(output.predicted_power_w,expected)
                alternate=self.frame.rename(columns={'GHI':'Radiation','WindSpeed':'Wind'})
                mapped=map_frame(alternate,self.config,require_target=False,overrides={'solar_irradiance':'Radiation','wind_speed':'Wind'})
                np.testing.assert_allclose(model.predict(mapped),expected)
                if family!='linear':
                    from sklearn.pipeline import make_pipeline
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.linear_model import LinearRegression
                    fit=np.array(model.metadata['fitting_source_rows'])-2 if family in ('tensorflow','pytorch') else self.frame.index.difference(result.index)
                    baseline=make_pipeline(StandardScaler(),LinearRegression()).fit(canonical.loc[fit,model.spec.features].to_numpy(),canonical.loc[fit,'power'])
                    np.testing.assert_allclose(result.linear_prediction_w,baseline.predict(canonical.loc[result.index,model.spec.features].to_numpy()))
                with TemporaryDirectory() as folder:
                    with ZipFile(BytesIO(export_trained(model,result,rejected))) as archive:
                        self.assertIn('feature_config.json',archive.namelist())
                        self.assertNotIn('training_data.csv',archive.namelist())
                        archive.extractall(folder)
                    loaded=load_session_model(folder)
                    self.assertEqual(loaded.metadata['feature_config'],self.config)
                    np.testing.assert_allclose(loaded.predict(self.frame),expected,rtol=1e-5)
                    np.testing.assert_allclose(load_model('ignored',folder).predict(self.frame),expected,rtol=1e-5)
                    csv=pd.read_csv(Path(folder)/'held_out_predictions.csv')
                    self.assertIn('predicted_value',csv)
                    self.assertNotIn('predicted_power_w',csv)
                    self.assertEqual(csv.target_unit.unique().tolist(),['kW'])

    def test_continuation_requires_identical_inputs_units_and_keeps_source(self):
        original,_,_,_=train_uploaded(self.frame,family='pytorch',track='Custom',feature_config=self.config,epochs=1)
        before=original.predict(self.frame)
        good=deepcopy(self.config)
        good['features'][0]['display_name']='Renamed GHI'
        adapted,_,_,_=train_uploaded(self.frame,original,feature_config=good,training_mode='continue',epochs=1)
        self.assertEqual(adapted.spec.features,original.spec.features)
        np.testing.assert_array_equal(original.predict(self.frame),before)
        for bad in (dict(self.config,features=self.config['features'][:1]),
                    dict(self.config,target=dict(self.config['target'],unit='W'))):
            with self.assertRaises(ValueError):
                train_uploaded(self.frame,original,feature_config=bad,training_mode='continue',epochs=1)

    def test_timestamp_group_mapping_stays_distinct_from_group_identifier(self):
        model, _, _, _ = train_uploaded(self.frame, family='linear', track='Custom',
            feature_config=self.config, strategy='grouped', group_column='Time')
        self.assertEqual(model.metadata['grouping_column'], 'timestamp')
        self.assertTrue(all('-' in group for group in model.metadata['training_groups']))

    def test_prediction_duplicate_mapping_and_leakage_warnings(self):
        with self.assertRaisesRegex(ValueError,'exactly once'):
            map_frame(self.frame,self.config,require_target=False,
                      overrides={'wind_speed':'GHI'})
        cfg=preset_config(['solar_irradiance','temperature','humidity','current','voltage','power'],'A')
        self.assertTrue(any('P = I x V' in note for note in mapping_warnings(cfg)))
        frame=self.frame.copy()
        frame['GHI']=frame['AC_Power']
        self.assertTrue(any('duplicates the target' in note for note in mapping_warnings(self.config,frame)))

if __name__=='__main__':
    unittest.main()
