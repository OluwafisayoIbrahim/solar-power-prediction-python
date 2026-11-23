"""Custom training, JSON reuse and immediate session inference in Streamlit."""
from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'deployment'),str(ROOT/'src')]
from streamlit.testing.v1 import AppTest
from test_feature_mapping import example
from feature_mapping import dump_mapping


class MappingAppTests(unittest.TestCase):
    def test_import_train_predict_and_mapping_change_invalidates(self):
        frame, config=example()
        upload=BytesIO(frame.to_csv(index=False).encode()); upload.name='romania.csv'
        mapping=BytesIO(dump_mapping(config).encode()); mapping.name='site_mapping.json'
        app=AppTest.from_file(str(ROOT/'deployment/prediction_workspace.py'),default_timeout=90).run()
        app.radio(key='model_source').set_value('Train from uploaded data').run()
        next(item for item in app.selectbox if item.label=='Model family').select('linear').run()
        def files(label,**kwargs):
            return mapping if label=='Reuse mapping configuration (JSON)' else upload if label in ('Training measurements','Measurements') else None
        with patch('streamlit.file_uploader',side_effect=files):
            app.run()
            next(item for item in app.button if item.label=='Apply mapping configuration').click().run()
            self.assertEqual(app.selectbox(key='training_track').value,'Custom')
            next(item for item in app.checkbox if item.label.startswith('I confirm')).check().run()
            next(item for item in app.button if item.label=='Train selected model').click().run()
            self.assertFalse(len(app.exception),str(app.exception))
            self.assertFalse(len(app.error),str([e.value for e in app.error]))
            trained=app.session_state['trained_model']
            self.assertEqual(trained.spec.features,['solar_irradiance','temperature','wind_speed'])
            self.assertEqual(app.number_input(key='input_wind_speed').label,'Wind Speed (m/s)')
            next(item for item in app.button if item.label=='Predict power').click().run()
            self.assertTrue(any(metric.label=='Predicted AC Power Output' and 'kW' in metric.value for metric in app.metric))
            next(item for item in app.button if item.label=='Predict uploaded readings').click().run()
            self.assertFalse(len(app.error),str([e.value for e in app.error]))
            self.assertEqual(len(app.session_state['batch_result']),len(frame))
            # A malformed import must not replace an existing valid mapping/result.
            mapping.seek(0); mapping.truncate(); mapping.write(b'bad json')
            app.run()
            self.assertEqual(app.session_state['trained_model'].spec.features,trained.spec.features)
            self.assertTrue(any('Mapping configuration unavailable' in error.value for error in app.error))
            # Editing a real feature label changes the configuration and clears stale results.
            next(item for item in app.text_input if item.label=='Display name for WindSpeed').set_value('Site wind').run()
            with self.assertRaises(KeyError):
                app.session_state['trained_model']
            self.assertEqual(app.session_state['model_source'],'Train from uploaded data')


if __name__=='__main__':
    unittest.main()
