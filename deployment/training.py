"""Isolated upload experiments; research exports are read-only inputs."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from batch_prediction import validate_rows, MAX_ROWS, download_csv
from model_registry import LoadedModel, ModelSpec, MODELS
from evaluate import calculate_mape


def evaluation_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    denominator = np.abs(actual).sum()
    return dict(MAE=float(mean_absolute_error(actual, predicted)),
                RMSE=float(np.sqrt(mean_squared_error(actual, predicted))),
                MAPE=calculate_mape(actual, predicted) if np.any(actual != 0) else None,
                WAPE=float(np.abs(actual-predicted).sum()/denominator*100) if denominator else None,
                R2=float(r2_score(actual, predicted)) if len(actual)>1 and np.ptp(actual)>0 else None)


def validate_training(frame, features):
    missing = set(features + ['power']) - set(frame.columns)
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(sorted(missing)))
    X, issues = validate_rows(frame, features)
    power = pd.to_numeric(frame.power, errors='coerce')
    bad = ~np.isfinite(power) | (power < 0)
    issues.loc[bad] += '; power: must be finite and nonnegative'
    if 'temperature' in X:
        issues.loc[X.temperature < -273.15] += '; temperature: below absolute zero in degrees C'
    return X, power, issues.str.strip('; ')


def timestamps(frame):
    if 'timestamp' not in frame or pd.api.types.is_numeric_dtype(frame.timestamp):
        raise ValueError('Chronological splitting requires full ISO timestamps.')
    if not frame.timestamp.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}(?:[T\s]|$)').all():
        raise ValueError('Use full ISO dates/times, not ambiguous dates or times alone.')
    values = pd.to_datetime(frame.timestamp, errors='coerce', format='mixed', utc=True)
    if values.isna().any():
        raise ValueError('Every valid row requires a valid timestamp.')
    return values


def split_uploaded(frame, strategy='random', group_column=None, holdout=.2, seed=0):
    if not 0.05 <= holdout <= .5:
        raise ValueError('Holdout proportion must be between 0.05 and 0.5.')
    groups = None
    if strategy == 'grouped':
        if group_column not in ('day_id', 'source_file', 'timestamp') or group_column not in frame:
            raise ValueError('Select day_id, source_file or timestamp for grouped splitting.')
        groups = timestamps(frame).dt.strftime('%Y-%m-%d') if group_column == 'timestamp' else frame[group_column]
        if groups.isna().any() or groups.astype(str).str.strip().eq('').any() or groups.nunique() < 2:
            raise ValueError('Grouping requires complete identifiers and at least two groups.')
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed).split(frame, groups=groups))
    elif strategy == 'chronological':
        times = timestamps(frame)
        groups = times.dt.strftime('%Y-%m-%d')
        days = sorted(groups.unique())
        if len(days) < 2:
            raise ValueError('Chronological splitting requires at least two complete calendar days.')
        cutoff = days[max(1, min(len(days)-1, int(len(days)*(1-holdout))))]
        order = np.argsort(times.to_numpy(), kind='stable')
        tr = order[(groups.iloc[order] < cutoff).to_numpy()]
        te = order[(groups.iloc[order] >= cutoff).to_numpy()]
        group_column = 'timestamp'
    elif strategy == 'random':
        tr, te = train_test_split(np.arange(len(frame)), test_size=holdout, random_state=seed)
    else:
        raise ValueError('Unknown split strategy.')
    if min(len(tr), len(te)) < 2:
        raise ValueError('Each partition needs at least two observations.')
    details = dict(split_strategy=strategy, grouping_column=group_column if groups is not None else None,
                   holdout_proportion=holdout, training_groups=[], evaluation_groups=[])
    if groups is not None:
        details.update(training_groups=sorted(set(groups.iloc[tr].astype(str))),
                       evaluation_groups=sorted(set(groups.iloc[te].astype(str))))
        if set(details['training_groups']) & set(details['evaluation_groups']):
            raise ValueError('Groups overlap between partitions.')
    if 'timestamp' in frame:
        try:
            times = timestamps(frame)
            for name, indices in [('training', tr), ('evaluation', te)]:
                details[name+'_date_range'] = [times.iloc[indices].min().isoformat(), times.iloc[indices].max().isoformat()]
        except ValueError:
            if strategy == 'chronological':
                raise
    return frame.index[tr], frame.index[te], details


class EarlyStoppingState:
    """Track the absolute best checkpoint; min_delta controls patience resets."""

    def __init__(self, patience=30, min_delta=.0001):
        self.patience, self.min_delta = patience, min_delta
        self.best_loss = self.reference_loss = float('inf')
        self.best_epoch = self.wait = 0
        self.best_weights = None

    def update(self, epoch, loss, snapshot):
        if not np.isfinite(loss):
            raise ValueError('Validation produced non-finite loss.')
        if loss < self.best_loss:
            self.best_loss, self.best_epoch = float(loss), epoch
            self.best_weights = snapshot()
        if loss < self.reference_loss - self.min_delta:
            self.reference_loss, self.wait = float(loss), 0
        else:
            self.wait += 1
        return self.wait >= self.patience


def train_uploaded(frame, original=None, group_column=None, epochs=200, progress=None, *,
                   family=None, track='B', training_mode='new', hidden_layers=(10,),
                   learning_rate=.001, l2=.0001, batch_size=32, seed=0,
                   strategy=None, holdout=.2, baseline=False, dataset_filename='',
                   dataset_bytes=None, model_name='Session model', activation='tanh', dataset_sheet=None,
                   early_stopping=True, patience=30, min_delta=.0001, feature_config=None):
    """Fresh weights or copied pretrained weights; evaluation never enters fitting."""
    if training_mode not in ('new', 'continue'):
        raise ValueError('Unknown training mode.')
    if original is not None:
        family, track = original.spec.kind, original.spec.track
    if family not in ('tensorflow', 'pytorch', 'tuned', 'untuned', 'linear') or track not in ('A', 'B', 'Custom'):
        raise ValueError('Select a supported model family and track.')
    if training_mode == 'continue' and (original is None or family not in ('tensorflow', 'pytorch')):
        raise ValueError('Continued training supports saved TensorFlow/Keras and PyTorch networks only; saved sklearn pipelines lack a reliable incremental fit.')
    neural = family in ('tensorflow', 'pytorch')
    if not isinstance(patience, int) or patience < 1 or not np.isfinite(min_delta) or min_delta < 0:
        raise ValueError('Patience must be a positive integer and min_delta finite and nonnegative.')
    if not 1 <= epochs <= (1000 if neural else 500) or not 1 <= batch_size <= 4096 or not 0 < learning_rate <= 1 or not np.isfinite(l2) or l2 < 0:
        raise ValueError('Invalid epochs, batch size, learning rate or regularisation.')
    if not hidden_layers or len(hidden_layers)>8 or any(not isinstance(n, int) or not 1 <= n <= 1024 for n in hidden_layers):
        raise ValueError('Use 1-8 hidden layers with 1-1024 neurons each.')
    if len(frame)>MAX_ROWS:
        raise ValueError(f'Limit training uploads to {MAX_ROWS:,} rows.')
    if feature_config is not None:
        from feature_mapping import validate_config, map_frame, preset_config
        feature_config = validate_config(feature_config, frame.columns,
                                         original.spec.features if training_mode == 'continue' else None)
        if training_mode == 'continue':
            expected = original.metadata.get('feature_config') or preset_config(original.spec.features + ['power'], original.spec.track)
            for actual, required in zip(feature_config['features'], expected['features']):
                if actual.get('unit', '') != required.get('unit', ''):
                    raise ValueError('Continued training must retain the saved input units and meanings.')
            if feature_config['target'].get('unit', '') != expected['target'].get('unit', ''):
                raise ValueError('Continued training must retain the saved target unit.')
        if training_mode == 'new':
            track = feature_config['preset']
        frame = map_frame(frame, feature_config)
        if group_column and group_column == feature_config.get('timestamp_column'):
            group_column = 'timestamp'
        elif group_column and group_column == feature_config.get('grouping_column'):
            group_column = 'day_id'
    elif track == 'Custom':
        raise ValueError('Custom training requires a feature configuration.')
    frame = frame.reset_index(drop=True)
    spec = original.spec if original else ModelSpec(f'{track.lower()}_{family}', model_name, track, family,
                                                   MODELS[f'{track.lower()}_{family}'].filename if track != 'Custom' else 'session_model.joblib')
    if feature_config is not None:
        # Source models may have been cached before dynamic schemas existed.
        spec = ModelSpec(key=spec.key, label=spec.label, track=spec.track, kind=spec.kind,
                         filename=getattr(spec, 'filename', ''),
                         artifact_directory=getattr(spec, 'artifact_directory', ''),
                         research_run=getattr(spec, 'research_run', ''),
                         input_features=tuple(entry['feature_id'] for entry in feature_config['features']))
    X, power, issues = validate_training(frame, spec.features)
    valid = issues.eq('')
    if valid.sum()<10:
        raise ValueError('At least 10 valid rows with measured power are required.')
    tr, te, split = split_uploaded(frame.loc[valid], strategy or ('grouped' if group_column else 'random'), group_column, holdout, seed)
    training_partition = tr
    validation_indices = []
    validation_split = None
    if neural:
        try:
            tr, validation_indices, validation_split = split_uploaded(
                frame.loc[training_partition], split['split_strategy'], group_column, .2, seed)
        except ValueError as exc:
            raise ValueError(f'Cannot create training-only validation subset: {exc}') from exc
    x, y = X.loc[tr].to_numpy(float), power.loc[tr].to_numpy(float)
    config = dict(hidden_layers=list(hidden_layers), activation=activation, learning_rate=learning_rate,
                  l2=l2, batch_size=batch_size, seed=seed, epochs=epochs)
    if training_mode == 'continue':
        for key in ('hidden_layers', 'activation', 'l2'):
            config[key] = original.metadata[key]
    metadata = dict(config, **split, model_name=model_name, model_family=family, track=track,
                    features=list(spec.features), training_mode=training_mode,
                    existing_model_identifier=original.spec.key if training_mode=='continue' else None,
                    origin='uploaded data', rows=len(tr), validation_rows=len(te), valid_rows=int(valid.sum()),
                    rejected_rows=int((~valid).sum()), dataset_filename=Path(dataset_filename).name, dataset_sheet=dataset_sheet,
                    dataset_sha256=sha256(dataset_bytes if dataset_bytes is not None else frame.to_csv(index=False).encode()).hexdigest(),
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    preprocessing='retained from source model' if training_mode=='continue' else 'fitted on training partition only',
                    optimizer_state='new optimizer; pretrained weights retained' if training_mode=='continue' else 'fresh',
                    software_versions={})
    for package in ('numpy','pandas','scikit-learn',family if family in ('tensorflow','pytorch') else 'joblib'):
        package = 'torch' if package=='pytorch' else package
        try:
            metadata['software_versions'][package] = version(package)
        except PackageNotFoundError:
            pass
    if feature_config is not None:
        metadata.update(feature_config=feature_config, feature_preset=feature_config['preset'],
                        original_columns=[entry['source_column'] for entry in feature_config['features']],
                        target=feature_config['target'])
    if neural:
        metadata.update(requested_max_epochs=epochs, early_stopping_enabled=bool(early_stopping),
                        patience=patience, min_delta=min_delta,
                        training_partition_rows=len(training_partition),
                        early_stopping_validation_rows=len(validation_indices),
                        early_stopping_split=validation_split,
                        fitting_source_rows=(tr + 2).tolist(),
                        early_stopping_source_rows=(validation_indices + 2).tolist(),
                        evaluation_source_rows=(te + 2).tolist(),
                        loss_definition='MSE in scaled target space plus hidden-kernel L2',
                        restored_best_weights=True)
    convergence = 0
    if family in ('tensorflow','pytorch'):
        if training_mode=='continue':
            source, xs, ys, _ = original.estimator
            xs, ys = deepcopy(xs), deepcopy(ys)
        else:
            xs, ys = StandardScaler().fit(x), StandardScaler().fit(y.reshape(-1,1))
        sx, sy = xs.transform(x).astype('float32'), ys.transform(y.reshape(-1,1)).astype('float32')
        vx = xs.transform(X.loc[validation_indices].to_numpy(float)).astype('float32')
        vy = ys.transform(power.loc[validation_indices].to_numpy(float).reshape(-1, 1)).astype('float32')
        stopping = EarlyStoppingState(patience, min_delta)
        if family=='tensorflow':
            import tensorflow as tf
            from models_tf import create_tf_ann
            if training_mode=='continue':
                tf.keras.utils.set_random_seed(seed)
                model = tf.keras.models.clone_model(source)
                model.set_weights(source.get_weights())
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
            else:
                model = create_tf_ann(len(spec.features), hidden_layers, activation, learning_rate, l2, seed)
            class Progress(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if not np.isfinite(logs['loss']):
                        raise ValueError('Training produced non-finite loss.')
                    should_stop = stopping.update(epoch + 1, logs['val_loss'],
                                                  lambda: deepcopy(self.model.get_weights()))
                    if progress:
                        progress((epoch+1)/epochs)
                    if early_stopping and should_stop:
                        self.model.stop_training = True
            history = model.fit(sx, sy, validation_data=(vx, vy), epochs=epochs,
                                batch_size=batch_size, shuffle=split['split_strategy'] != 'chronological',
                                verbose=0, callbacks=[Progress()]).history
            model.set_weights(stopping.best_weights)
            if not np.isfinite(history['loss']).all():
                raise ValueError('Training produced non-finite loss.')
            metadata['training_loss'] = history['loss']
            metadata['validation_loss'] = history['val_loss']
        else:
            import torch
            from models_torch import TorchANN, seed_everything, l2_penalty
            seed_everything(seed)
            model = deepcopy(source) if training_mode=='continue' else TorchANN(len(spec.features), hidden_layers, activation, learning_rate, l2, seed)
            model.train()
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(sx), torch.from_numpy(sy)), batch_size=batch_size, shuffle=split['split_strategy'] != 'chronological', generator=torch.Generator().manual_seed(seed))
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
            losses, validation_losses = [], []
            validation_x, validation_y = torch.from_numpy(vx), torch.from_numpy(vy)
            for epoch in range(epochs):
                total=0.
                for inputs, targets in loader:
                    optimizer.zero_grad()
                    loss = torch.nn.functional.mse_loss(model(inputs), targets)+l2_penalty(model, config['l2'])
                    if not torch.isfinite(loss):
                        raise ValueError('Training produced non-finite loss.')
                    loss.backward()
                    optimizer.step()
                    total += loss.item()*len(inputs)
                losses.append(total/len(x))
                model.eval()
                with torch.no_grad():
                    validation_loss = (torch.nn.functional.mse_loss(model(validation_x), validation_y)
                                       + l2_penalty(model, config['l2'])).item()
                validation_losses.append(validation_loss)
                should_stop = stopping.update(epoch + 1, validation_loss,
                                              lambda: deepcopy(model.state_dict()))
                if progress:
                    progress((epoch+1)/epochs)
                if early_stopping and should_stop:
                    break
                model.train()
            model.load_state_dict(stopping.best_weights)
            model.eval()
            metadata['training_loss']=losses
            metadata['validation_loss']=validation_losses
        completed = len(metadata['training_loss'])
        metadata.update(completed_epochs=completed, best_epoch=stopping.best_epoch,
                        early_stopped=bool(early_stopping and stopping.wait >= patience and completed < epochs),
                        best_validation_loss=stopping.best_loss)
        if progress:
            progress(1.)
        estimator=(model,xs,ys,metadata)
    else:
        regressor = LinearRegression() if family=='linear' else MLPRegressor(hidden_layer_sizes=hidden_layers, activation=activation, learning_rate_init=learning_rate, alpha=l2, batch_size=batch_size, max_iter=epochs, random_state=seed, early_stopping=False)
        estimator=make_pipeline(StandardScaler(), regressor)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            estimator.fit(x,y)
        convergence=sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        metadata['iterations_completed']=getattr(regressor,'n_iter_',None)
        if progress:
            progress(1.)
    loaded=LoadedModel(spec,estimator,metadata)
    result=frame.loc[te].copy()
    result.insert(0,'source_row', result.index+2)
    result['predicted_power_w']=loaded.predict(X.loc[te])
    result['prediction_status']='predicted'
    result['prediction_issue']=''
    result['prediction_model']=model_name
    if feature_config is not None:
        result.attrs['feature_config'] = feature_config
    metadata['evaluation_metrics']=evaluation_metrics(power.loc[te], result.predicted_power_w)
    if baseline and family!='linear':
        linear=make_pipeline(StandardScaler(),LinearRegression()).fit(x,y)
        result['linear_prediction_w']=linear.predict(X.loc[te].to_numpy(float))
        metadata['baseline_metrics']=evaluation_metrics(power.loc[te], result.linear_prediction_w)
    rejected=frame.loc[~valid].copy()
    rejected.insert(0,'source_row',rejected.index+2)
    rejected['training_issue']=issues.loc[~valid]
    return loaded,result,rejected,convergence


def export_trained(loaded, evaluation=None, rejected=None):
    """Return a ZIP; temporary serialization files are removed immediately."""
    output = BytesIO()
    with TemporaryDirectory(prefix="solar-model-") as temporary:
        folder = Path(temporary)
        prefix = f"track_{loaded.spec.track.lower()}"
        if loaded.metadata.get('feature_config'):
            from feature_mapping import dump_mapping
            (folder / 'feature_config.json').write_text(dump_mapping(loaded.metadata['feature_config']), encoding='utf-8')
            (folder / 'session_manifest.json').write_text(json.dumps({'schema_version': 1}), encoding='utf-8')
            if loaded.spec.kind in ('tensorflow', 'pytorch'):
                model, x_scaler, y_scaler, _ = loaded.estimator
                joblib.dump(x_scaler, folder / 'x_scaler.joblib')
                joblib.dump(y_scaler, folder / 'y_scaler.joblib')
                if loaded.spec.kind == 'tensorflow':
                    model.save(folder / 'model.keras')
                else:
                    import torch
                    torch.save(model.state_dict(), folder / 'model.pt')
            else:
                joblib.dump(loaded.estimator, folder / 'model.joblib')
            metadata_name = 'metadata.json'
        elif loaded.spec.kind == "tensorflow":
            model, x_scaler, y_scaler, _ = loaded.estimator
            model.save(folder / f"{prefix}_ann_final.keras")
            joblib.dump(x_scaler, folder / f"{prefix}_x_scaler.joblib")
            joblib.dump(y_scaler, folder / f"{prefix}_y_scaler.joblib")
            metadata_name = f"{prefix}_tf_metadata.json"
        elif loaded.spec.kind == "pytorch":
            import torch
            from predict_torch import artifact_names
            names = artifact_names(loaded.spec.track)
            model, x_scaler, y_scaler, _ = loaded.estimator
            torch.save(model.state_dict(), folder / names["model"])
            joblib.dump(x_scaler, folder / names["x_scaler"])
            joblib.dump(y_scaler, folder / names["y_scaler"])
            metadata_name = names["metadata"]
        else:
            joblib.dump(loaded.estimator, folder / loaded.spec.filename)
            metadata_name = Path(loaded.spec.filename).with_suffix(".json").name
        (folder / metadata_name).write_text(json.dumps(loaded.metadata, indent=2), encoding="utf-8")
        (folder / 'evaluation_summary.json').write_text(json.dumps(dict(loaded.metadata.get('evaluation_metrics', {}), target=loaded.metadata.get('target', {'display_name': 'Power', 'unit': 'W'})), indent=2), encoding='utf-8')
        (folder / 'README.txt').write_text('Trusted local model package. If session_manifest.json is present, load with model_registry.load_session_model(directory). Otherwise use model_registry.load_model with matching track/family and this extracted directory. Neural models include input/target scalers. Never load untrusted pickle/joblib files. Research files are not overwritten.', encoding='utf-8')
        if evaluation is not None:
            (folder / 'held_out_predictions.csv').write_bytes(download_csv(evaluation))
        if rejected is not None and not rejected.empty:
            (folder / 'rejected_rows.csv').write_bytes(download_csv(rejected))
        with ZipFile(output, "w") as archive:
            for path in folder.iterdir():
                archive.write(path, path.name)
    return output.getvalue()
