"""Native Streamlit controls for portable, explicit dataset mappings."""
from hashlib import sha256
import json

import pandas as pd
import streamlit as st

from feature_mapping import (STANDARD, identifier, preset_config, validate_config,
                             load_mapping, dump_mapping, mapping_warnings)


def training_mapping(frame, preset, clear_results, expected_features=None):
    columns = list(frame.columns)
    st.subheader('Configure input features')
    st.caption('Original headers are preserved. Suggested mappings require review and unit confirmation; no measurements or units are converted.')
    uploaded = st.file_uploader('Reuse mapping configuration (JSON)', type=['json'], key='mapping_upload')
    if uploaded is not None:
        try:
            imported = load_mapping(uploaded.getvalue(), columns, expected_features)
            def apply_import(mapping=imported, headers=columns, expected=expected_features):
                st.session_state['applied_mapping'] = mapping
                st.session_state['applied_mapping_headers'] = headers
                if expected is None:
                    st.session_state['training_track'] = mapping['preset']
                clear_results()
            st.button('Apply mapping configuration', on_click=apply_import)
        except ValueError as exc:
            st.error(f'Mapping configuration unavailable: {exc}')
    imported = st.session_state.get('applied_mapping')
    if (not imported or st.session_state.get('applied_mapping_headers') != columns
            or imported.get('preset') != preset
            or (expected_features is not None and [e['feature_id'] for e in imported['features']] != list(expected_features))):
        imported = None
    initial = imported or preset_config(columns, preset, expected_features)
    identity = sha256(json.dumps([columns, preset, initial, expected_features], sort_keys=True).encode()).hexdigest()[:16]
    def select(label, default, suffix, options=None):
        options = [None] + columns if options is None else options
        return st.selectbox(label, options, index=options.index(default) if default in options else 0,
                            key=f'mapping_{identity}_{suffix}')
    target_source = select('Target dataset column', initial['target']['source_column'], 'target')
    target_name = st.text_input('Target display name', value=initial['target']['display_name'], key=f'mapping_{identity}_target_name')
    target_unit = st.text_input('Target unit', value=initial['target'].get('unit', 'W'), key=f'mapping_{identity}_target_unit')
    grouping = select('Dataset grouping column (optional)', initial.get('grouping_column'), 'group')
    timestamp = select('Dataset timestamp column (optional)', initial.get('timestamp_column'), 'timestamp')
    if expected_features is not None:
        st.info('Input identifiers, order, meanings and units must match the existing model. To add or remove inputs, choose Train a New Model.')
    if preset == 'Custom' and expected_features is None:
        eligible = [c for c in columns if c not in (target_source, grouping, timestamp)
                    and pd.to_numeric(frame[c], errors='coerce').notna().any()
                    and not pd.api.types.is_datetime64_any_dtype(frame[c])]
        selected = st.multiselect('Input dataset columns', eligible,
            default=[e['source_column'] for e in initial['features'] if e['source_column'] in eligible],
            key=f'mapping_{identity}_selected')
        originals = {e['source_column']: e for e in initial['features']}
        entries = []
        for source in selected:
            entry = originals.get(source, dict(source_column=source, feature_id=identifier(source), display_name=source, unit=''))
            with st.expander(f'Input: {source}', expanded=True):
                definition = st.selectbox(f'Standard definition for {source}', ['Custom'] + list(STANDARD),
                    index=(['Custom'] + list(STANDARD)).index(entry['feature_id']) if entry['feature_id'] in STANDARD else 0,
                    key=f'mapping_{identity}_{source}_definition')
                fid = definition if definition != 'Custom' else st.text_input(f'Internal identifier for {source}', value=entry['feature_id'], key=f'mapping_{identity}_{source}_id')
                default_name, default_unit = STANDARD[definition][:2] if definition != 'Custom' else (entry['display_name'], entry.get('unit', ''))
                name = st.text_input(f'Display name for {source}', value=default_name, key=f'mapping_{identity}_{source}_{definition}_name')
                unit = st.text_input(f'Unit for {source}', value=default_unit, key=f'mapping_{identity}_{source}_{definition}_unit')
            entries.append(dict(source_column=source, feature_id=fid, display_name=name, unit=unit))
    else:
        entries = []
        for entry in initial['features']:
            fid = entry['feature_id']
            with st.expander(f'Input: {fid}', expanded=True):
                source = select(f'Dataset column for {fid}', entry['source_column'], fid + '_source')
                name = st.text_input(f'Display name for {fid}', value=entry['display_name'], key=f'mapping_{identity}_{fid}_name')
                unit = st.text_input(f'Unit for {fid}', value=entry.get('unit', ''), disabled=expected_features is not None, key=f'mapping_{identity}_{fid}_unit')
            entries.append(dict(source_column=source, feature_id=fid, display_name=name, unit=unit))
    config = dict(schema_version=1, preset=preset, features=entries,
                  target=dict(source_column=target_source, display_name=target_name, unit=target_unit),
                  grouping_column=grouping, timestamp_column=timestamp)
    def restore():
        st.session_state.pop('applied_mapping', None)
        for key in list(st.session_state):
            if key.startswith('mapping_' + identity + '_'):
                st.session_state.pop(key, None)
        clear_results()
    st.button('Restore feature preset', on_click=restore)
    config = validate_config(config, columns, expected_features)
    st.dataframe(pd.DataFrame(config['features']), hide_index=True, alt='Configured input order, source columns, display names and units')
    for note in mapping_warnings(config, frame):
        st.warning(note)
    st.download_button('Download mapping configuration', dump_mapping(config), 'column_mapping.json', 'application/json', key='mapping_download')
    return config


def prediction_mapping(frame, config, clear_prediction_results):
    """Require every model input once, with explicit alternatives to saved sources."""
    from feature_mapping import map_frame
    columns = list(frame.columns)
    identity = sha256(json.dumps([columns, config], sort_keys=True).encode()).hexdigest()[:16]
    overrides = {}
    with st.expander('Spreadsheet input mapping', expanded=True):
        for entry in config['features']:
            fid = entry['feature_id']
            default = entry['source_column'] if entry['source_column'] in columns else fid if fid in columns else None
            options = [None] + columns
            overrides[fid] = st.selectbox(f'Prediction column for {entry["display_name"]}', options,
                index=options.index(default), key=f'prediction_mapping_{identity}_{fid}', on_change=clear_prediction_results)
    return map_frame(frame, config, require_target=False, overrides=overrides)
