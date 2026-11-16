"""Regenerate the supplied edited LaTeX manuscript from checked research outputs.

Requires vl-convert-python for PDF figures. Compile manuscript/source/manuscript.tex
with a LaTeX distribution after running this script from the project environment.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'deployment'), str(ROOT / 'src')]
import pandas as pd
import vl_convert as vlc
from manuscript_data import (load_evidence, dataset_table, track_table, model_table,
                             result_table, diagnostic_predictions, score, track_summary)
import manuscript_figures as figures

OUT = ROOT / 'manuscript/source'


def esc(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '±': r'$\pm$', '×': r'$\times$', '²': r'$^2$', '∈': r'$\in$',
                    '−': '-', '–': '--', '—': '---', '“': '``', '”': "''", '’': "'"}
    return ''.join(replacements.get(char, char) for char in str(value))


def replace(pattern, value, text, count=1):
    result, found = re.subn(pattern, lambda match: value, text, count=count, flags=re.S)
    if found != count:
        raise ValueError(f'Expected {count} manuscript matches, got {found}: {pattern}')
    return result


def update_table(text, number, frame):
    frame.to_csv(OUT / f'table_{number}.csv', index=False)
    pattern = (r'(\\label\{tab:' + str(number) + r'\}.*?\\toprule\n)'
               r'.*?(?=\\bottomrule)')
    header = ' & '.join(r'\textbf{' + esc(col) + '}' for col in frame.columns) + r' \\ \midrule'
    lines = [header]
    for row in frame.itertuples(index=False, name=None):
        lines.append(' & '.join(esc(f'{v:.3f}' if isinstance(v, float) else v)
                                for v in row) + r' \\')
    text, found = re.subn(pattern, lambda m: m[1] + '\n'.join(lines) + '\n', text, flags=re.S)
    if found != 1:
        raise ValueError(f'Table {number} not found')
    return text


def chart(name, value):
    spec = value.configure(font='Times New Roman').configure_axis(
        labelFontSize=12, titleFontSize=13).configure_legend(labelFontSize=12, titleFontSize=13).to_dict()
    (OUT / 'figures' / f'figure_{name}.vl.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    (OUT / 'figures' / f'figure_{name}.pdf').write_bytes(vlc.vegalite_to_pdf(spec))


def main():
    evidence = load_evidence()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT / 'manuscript/editorial_reference_superseded.zip') as archive:
        text = archive.read('manuscript.tex').decode('utf-8').replace('\r', '')
    a = evidence['audit']
    text = replace(r'\\begin\{abstract\}.*?\\end\{abstract\}',
        r'\begin{abstract}' + (
        'Neural networks offered no consistent advantage over linear regression in this re-evaluation of solar power estimation. '
        'Lower errors with measured current and voltage reflected richer input information rather than better weather-only prediction. '
        'Using 180 observations across 20 daily files, the study compares two tracks: electrical and environmental inputs (Track A), '
        'and environmental inputs with observation hour (Track B). Linear regression, a no-fitting voltage/current product baseline '
        'for Track A, and scikit-learn, TensorFlow/Keras and PyTorch networks are evaluated using seven outer grouped folds, '
        'nested selection where applicable, and three neural-network seeds. Metrics pool held-out predictions for each seed. '
        f"The electrical-product baseline has mean MAE {a['power_iv_mae']:.3f} W. Track A linear regression achieves mean MAE "
        f"{score(evidence, 'A', 'Linear'):.3f} W, compared with {score(evidence, 'A', 'TensorFlow selected'):.3f} W for TensorFlow "
        f"and {score(evidence, 'A', 'PyTorch selected'):.3f} W for PyTorch. In Track B, linear regression achieves mean MAE "
        f"{score(evidence, 'B', 'Linear'):.3f} W, compared with {score(evidence, 'B', 'Sklearn tuned'):.3f} W for the tuned scikit-learn network. "
        'These differences are descriptive. One repeated daily file was excluded before evaluation. '
        'Small sample size, uncertain temporal coverage and absent external validation restrict generalisation. '
        'Both tracks estimate power from contemporaneous measurements; neither establishes future forecasting capability.'
        ) + r'\end{abstract}', text)
    text = text.replace('189 observations in 21 source-file groups', '180 observations in 20 daily-file groups')
    text = replace(r'Here, source-file grouping.*?finding below limits the interpretation of that separation\.',
        'Here, daily grouping separates measurement days after one repeated daily file was excluded.', text)
    text = replace(r'\\par\\noindent Nine measurement records.*?\\par',
        'One repeated daily file was excluded before evaluation. The active dataset contains no exact repeated '
        'measurement records when filename identity is excluded. The dashboard presents the resulting 20-day evaluation.', text)
    text = text.replace('including repeated records.', 'after exclusion of the repeated daily file.')
    text = replace(r'Seven outer \\texttt\{GroupKFold\}.*?The tuned scikit-learn MLP',
        r'Seven outer \texttt{GroupKFold} partitions evaluate 180 observations across 20 daily files. '
        'Six folds hold out three groups (27 rows) and train on 17 groups (153 rows); one holds out two groups '
        '(18 rows) and trains on 18 groups (162 rows). Each row receives one outer prediction per model and seed. '
        'Three inner grouped folds use 11 or 12 fitting groups and six or five validation groups when the outer '
        'training set has 17 groups; with 18 groups, they use 12 and six.\nThe tuned scikit-learn MLP', text)
    text = replace(r'independent dataset replications\. Filename grouping.*?observational independence\.',
        'independent dataset replications. Training and test partitions share neither a daily group nor an exact '
        'measurement record. This does not establish independence across sites or seasons.', text)
    text = text.replace('S1 had 1 selected outer refit(s)', 'S1 had 2 selected outer refits')
    text = text.replace('S2 had 0.', 'S2 had 0. S1 used batch size 32 and S2 used 16; all other training settings matched. '
        'The tie-breakers were lower equal-track mean inner MAE, followed by S1 for an exact tie.')
    text = text.replace('21 source groups / 189 rows', '20 daily groups / 180 rows')
    text = text.replace('Outer training: 18 groups / 162 rows', 'Outer training: 17 or 18 groups / 153 or 162 rows')
    text = text.replace('Outer test: 3 groups / 27 rows', 'Outer test: 3 groups / 27 rows (six folds); 2 groups / 18 rows (one fold)')
    text = text.replace('Inner fit: 12 groups / 108 rows', 'Inner fit: 11 or 12 groups / 99 or 108 rows')
    text = text.replace('Inner validation: 6 groups / 54 rows', 'Inner validation: 5 or 6 groups / 45 or 54 rows')
    text = text.replace('Refit on all 18 outer training groups', 'Refit on all outer training groups')
    # More room for the two variable-size branches in the validation diagram.
    text = text.replace('(train) at (-1.5,-1.1)', '(train) at (-2,-1.5)')
    text = text.replace('(test) at (4.5,-1.1)', '(test) at (4.5,-1.5)')
    text = text.replace('(fit) at (-3,-2.3)', '(fit) at (-4,-3.3)')
    text = text.replace('(val) at (1,-2.3)', '(val) at (1,-3.3)')
    text = text.replace('(select) at (-1.5,-3.5)', '(select) at (-1.5,-4.8)')
    text = text.replace('(refit) at (-1.5,-4.6)', '(refit) at (-1.5,-5.8)')
    text = text.replace('(score) at (-1.5,-5.7)', '(score) at (-1.5,-6.8)')
    text = text.replace('duplicate measurements across identifiers are a separate overlap risk.',
                        'test daily groups and identical measurement records are excluded from fitting.')
    text = text.replace('The repeated measurements described above limit the independence of all reported results.',
                        'The active evaluation contains no repeated measurement records across the training/test partitions.')
    text = replace(r'Linear regression has the lowest Track A mean MAE.*?This is electrical-input estimation\.',
                   esc(track_summary(evidence, 'A')) + ' The electrical-product baseline already gives low error without learning.', text)
    text = replace(r'The tuned scikit-learn ANN has the lowest Track B.*?claim follows\.',
                   esc(track_summary(evidence, 'B')), text)
    text = text.replace('and the electrical product is 1.168 W.', f"and the electrical product is {a['power_iv_mae']:.3f} W.")
    text = replace(r'\\textbf\{RQ2 --- complexity.*?(?=\\textbf\{RQ3)',
        r'\textbf{RQ2 --- baseline comparisons.} ' + esc(track_summary(evidence, 'A') + ' ' + track_summary(evidence, 'B')) +
        ' Rankings depend on the metric and must be interpreted within each track.\n\n', text)
    text = replace(r'\\textbf\{RQ3.*?(?=\\textbf\{RQ4)',
        r'\textbf{RQ3 --- held-out days.} Each test daily group and its measurements are excluded from fitting. '
        'Model selection is nested within the outer training days. One repeated daily file was excluded before evaluation. '
        'There is no controlled experiment isolating the effect of changing the split strategy. '
        'External-site or future-horizon performance cannot be inferred from these results.\n\n', text)
    text = replace(r'\\textbf\{Sample size and effective independence\.\}.*?(?=\\textbf\{Temporal)',
        r'\textbf{Sample size and effective independence.} Only 180 observations across 20 daily files are available. '
        'One repeated daily file was excluded before evaluation. The retained records contain no exact repeated measurements '
        'when filename identity is excluded. The small sample and potentially correlated readings still limit generalisation.\n\n', text)
    text = replace(r'First resolve the duplicate-file provenance.*?Collect longer records across',
                   'Collect longer records across', text)
    text = text.replace('Full calendar dates, instrument calibration records and the provenance of the repeated source file remain unresolved.',
        'Full calendar dates and instrument calibration records remain unavailable. One repeated daily file was excluded; no missing metadata are inferred from filenames.')
    for old, new in [('results/tensorflow/convergence_s2', 'results/research_20_days/tensorflow_s2'),
                     ('results/pytorch/parity_s2', 'results/research_20_days/pytorch_selected'),
                     ('results/comparisons/parity_s2', 'results/research_20_days/comparison')]:
        text = text.replace(old.replace('/', '/\\allowbreak{}').replace('_', r'\_'),
                            new.replace('/', '/\\allowbreak{}').replace('_', r'\_'))
    tables = {1: dataset_table(evidence), 2: track_table(), 3: model_table(evidence),
              4: result_table(evidence, 'A'), 5: result_table(evidence, 'B')}
    variability = pd.concat([v.assign(Framework=k) for k, v in evidence['seeds'].items()], ignore_index=True)
    tables[6] = variability[variability.model.isin(['TensorFlow selected', 'PyTorch selected'])].rename(
        columns={'track': 'Track', 'model': 'Model', 'seed': 'Seed', 'MAE': 'MAE (W)',
                 'RMSE': 'RMSE (W)', 'MAPE': 'MAPE (%)', 'R2': 'R²'})
    # Preserve the original eight-column table layout.
    tables[6] = tables[6][['Track', 'Seed', 'Model', 'MAE (W)', 'RMSE (W)', 'MAPE (%)', 'R²', 'Framework']]
    tables[7] = pd.DataFrame([{'Recorded environment': k, 'Python': cfg['python'],
                              'scikit-learn': cfg['sklearn'], 'Framework': cfg.get('tensorflow', cfg.get('torch'))}
                             for k, cfg in evidence['configs'].items()])
    for number, frame in tables.items():
        text = update_table(text, number, frame)
    for name, value in [('2', figures.power_distribution(evidence['dataset'])),
                        ('5', figures.comparison_chart(evidence['comparison'], 'MAE')),
                        ('A4', figures.comparison_chart(evidence['comparison'], 'RMSE')),
                        ('A5', figures.comparison_chart(evidence['comparison'], 'MAPE'))]:
        chart(name, value)
    for track, names in [('B', ['4', '6', '7']), ('A', ['A1', 'A2', 'A3'])]:
        predictions = diagnostic_predictions(evidence, track)
        for name, factory in zip(names, [figures.parity_chart, figures.residual_chart, figures.daily_chart]):
            chart(name, factory(predictions))
    (OUT / 'manuscript.tex').write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n', encoding='utf-8')
    evidence['comparison'].to_csv(OUT / 'matched_s2_metrics.csv', index=False)
    (OUT / 'verification.json').write_text(json.dumps({'audit': a, 'selection': evidence['selection'],
        'main_tables': 7, 'main_figures': 7, 'supplemental_figures': 5,
        'protocol_warnings': evidence['protocol']['warnings'],
        'verification': 'Saved row identities, targets, folds, per-seed metrics and summaries reconciled.'}, indent=2), encoding='utf-8')
    print('Generated manuscript/source with checked tables and figures.', flush=True)


if __name__ == '__main__':
    main()
