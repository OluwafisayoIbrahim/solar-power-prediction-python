"""Portable dataset mappings, independent of Streamlit and research artifacts."""
from copy import deepcopy
import json
import re

import pandas as pd
import numpy as np
from features import TRACK_A_FEATURES, TRACK_B_FEATURES

STANDARD = {
    'solar_irradiance': ('Solar Irradiance', 'W/m2', ['ghi', 'irradiance']),
    'temperature': ('Temperature', 'deg C', ['ambienttemp', 'ambienttemperature']),
    'humidity': ('Relative Humidity', '%', ['rh', 'relativehumidity']),
    'hour': ('Hour', 'hour', []),
    'current': ('Current', 'A', []),
    'voltage': ('Voltage', 'V', []),
}
RESERVED = {'power', 'timestamp', 'day_id', 'source_file', 'source_row', 'predicted_power_w',
            'predicted_value', 'prediction_status', 'prediction_issue', 'prediction_model', 'linear_prediction_w'}


def identifier(name):
    value = re.sub(r'[^a-z0-9_]+', '_', str(name).lower()).strip('_')
    return ('feature_' + value) if not value or value[0].isdigit() else value


def suggest_column(columns, feature):
    aliases = [feature] + STANDARD.get(feature, ('', '', []))[2]
    if feature == 'power':
        aliases += ['ac_power', 'activepower', 'pv_output']
    key = lambda value: re.sub(r'[^a-z0-9]', '', value.lower())
    for alias in aliases:
        matches = [column for column in columns if key(column) == key(alias)]
        if len(matches) == 1:
            return matches[0]
    return None


def preset_config(columns, preset='B', features=None):
    names = features if features is not None else TRACK_A_FEATURES if preset == 'A' else TRACK_B_FEATURES if preset == 'B' else []
    return dict(schema_version=1, preset=preset,
        features=[dict(source_column=suggest_column(columns, name), feature_id=name,
                       display_name=STANDARD.get(name, (name, '', []))[0],
                       unit=STANDARD.get(name, ('', '', []))[1]) for name in names],
        target=dict(source_column=suggest_column(columns, 'power'), display_name='Power', unit='W'),
        grouping_column=next((c for c in ('day_id', 'source_file') if c in columns), None),
        timestamp_column='timestamp' if 'timestamp' in columns else None)


def validate_config(config, columns=None, expected_features=None):
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ValueError('Use a mapping JSON with schema_version 1.')
    if config.get('preset') not in ('A', 'B', 'Custom'):
        raise ValueError('Mapping preset must be A, B or Custom.')
    entries = config.get('features')
    if not isinstance(entries, list) or not 1 <= len(entries) <= 100:
        raise ValueError('Select between 1 and 100 numerical input features.')
    sources, ids = [], []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Each feature must be a mapping object.')
        for field in ('source_column', 'feature_id', 'display_name'):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f'Every feature requires a {field}.')
        fid = entry['feature_id']
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', fid) or fid in RESERVED:
            raise ValueError(f'Invalid/reserved internal feature identifier: {fid}.')
        if not isinstance(entry.get('unit', ''), str):
            raise ValueError('Feature units must be text.')
        if fid in STANDARD and entry.get('unit', '') != STANDARD[fid][1]:
            raise ValueError(f'{fid} requires unit {STANDARD[fid][1]}. Use a new custom identifier for a measurement in different units; no conversions are performed.')
        sources.append(entry['source_column'])
        ids.append(fid)
    if len(sources) != len(set(sources)) or len(ids) != len(set(ids)):
        raise ValueError('Input source columns and internal identifiers must each be unique.')
    if config['preset'] in ('A', 'B'):
        required = TRACK_A_FEATURES if config['preset'] == 'A' else TRACK_B_FEATURES
        if ids != list(required):
            raise ValueError('Track A/B presets retain their standard feature order. Choose Custom to change the feature list.')
    target = config.get('target')
    if not isinstance(target, dict) or any(not isinstance(target.get(key), str) or not target[key].strip()
                                          for key in ('source_column', 'display_name')) or not isinstance(target.get('unit', ''), str):
        raise ValueError('Select and name the target column; supply a text unit (or leave it unknown).')
    roles = [config.get('grouping_column'), config.get('timestamp_column')]
    if any(role is not None and (not isinstance(role, str) or not role.strip()) for role in roles):
        raise ValueError('Grouping and timestamp columns must be column names or null.')
    if target['source_column'] in sources or any(role in sources for role in roles if role):
        raise ValueError('Target, grouping identifiers and raw timestamps cannot be input features. Transform timestamps into numerical features first.')
    assigned = [target['source_column']] + [role for role in roles if role]
    if len(assigned) != len(set(assigned)):
        raise ValueError('Target, grouping and timestamp roles must use separate source columns.')
    if expected_features is not None and ids != list(expected_features):
        raise ValueError('Continued training must preserve the saved feature identifiers and order. Choose Train a New Model to change inputs.')
    if columns is not None:
        missing = set(sources + assigned) - set(columns)
        if missing:
            raise ValueError('Mapping references missing columns: ' + ', '.join(sorted(missing)))
    return deepcopy(config)


def load_mapping(data, columns=None, expected_features=None):
    if len(data) > 1024 * 1024:
        raise ValueError('Mapping JSON must be no larger than 1 MB.')
    try:
        config = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Unable to read mapping JSON.') from exc
    return validate_config(config, columns, expected_features)


def dump_mapping(config):
    return json.dumps(validate_config(config), indent=2, ensure_ascii=False)


def map_frame(frame, config, require_target=True, overrides=None, allow_internal=False):
    """Construct a separate canonical frame; never rename the uploaded frame in place."""
    config = validate_config(config)
    if not frame.columns.is_unique:
        raise ValueError('Dataset columns must be unique.')
    overrides = overrides or {}
    if set(overrides) - {entry['feature_id'] for entry in config['features']}:
        raise ValueError('Prediction mapping includes unknown feature identifiers.')
    used, mapped = [], {}
    for entry in config['features']:
        fid = entry['feature_id']
        source = overrides.get(fid, entry['source_column'])
        if source not in frame and allow_internal and fid in frame and fid not in overrides:
            source = fid
        if source not in frame:
            raise ValueError(f'Missing input column for {entry["display_name"]}: {source}.')
        used.append(source)
        values = pd.to_numeric(frame[source], errors='coerce')
        if values.notna().sum() == 0:
            raise ValueError(f'Input {source} has no numerical observations. Raw timestamps/categories require explicit preprocessing.')
        if (pd.api.types.is_datetime64_any_dtype(frame[source]) or source == config.get('timestamp_column')
                or identifier(source) in ('timestamp', 'datetime', 'date')):
            raise ValueError('Raw timestamps cannot be direct model inputs.')
        mapped[fid] = values
    if len(used) != len(set(used)):
        raise ValueError('Each required input must be mapped exactly once to a distinct column.')
    target = config['target']['source_column']
    if target in used:
        raise ValueError('The target column cannot also be used as an input.')
    if any(role in used for role in (config.get('grouping_column'), config.get('timestamp_column')) if role):
        raise ValueError('Grouping identifiers and raw timestamps cannot be prediction inputs.')
    if target in frame:
        mapped['power'] = pd.to_numeric(frame[target], errors='coerce')
        if require_target and mapped['power'].notna().sum() == 0:
            raise ValueError('The selected target has no numerical observations.')
    elif require_target:
        raise ValueError(f'Missing target column: {target}.')
    for key, role in [('grouping_column', 'day_id'), ('timestamp_column', 'timestamp')]:
        source = config.get(key)
        if source is not None:
            if source not in frame and require_target:
                raise ValueError(f'Missing {key}: {source}.')
            if source in frame:
                mapped[role] = frame[source]
    return pd.DataFrame(mapped, index=frame.index)


def mapping_warnings(config, frame=None):
    ids = {entry['feature_id'] for entry in config['features']}
    notes = ['More input features do not automatically improve held-out accuracy.']
    if {'current', 'voltage'} <= ids:
        notes.append('Current and voltage can directly determine electrical power (P = I x V). This is contemporaneous estimation, not weather-only forecasting.')
    if frame is not None:
        target = pd.to_numeric(frame[config['target']['source_column']], errors='coerce')
        for entry in config['features']:
            values = pd.to_numeric(frame[entry['source_column']], errors='coerce')
            paired = target.notna() & values.notna()
            if paired.sum() >= 2 and np.array_equal(values[paired].to_numpy(float), target[paired].to_numpy(float)):
                notes.append(f'{entry["display_name"]} duplicates the target values; check for target leakage.')
            elif any(word in entry['source_column'].lower() for word in ('power', 'output', 'energy')):
                notes.append(f'{entry["display_name"]} may be derived from the target; verify that it is available independently at prediction time.')
    return notes


def target_labels(config=None):
    target = config['target'] if config else dict(display_name='Power', unit='W')
    unit = target.get('unit', '').strip() or 'unit unspecified'
    return target['display_name'], unit
