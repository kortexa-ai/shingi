"""Create a compact, reproducible record from a completed local experiment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from shingi.calibration import replay
from shingi.decision import Calibration
from shingi.metrics import probabilities


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items() if k != 'reliability_bins'}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def paired(records, predictions, calibrations):
    counts = Counter()
    for row in records:
        correct = []
        for name in ('base', 'adapter'):
            prediction = predictions[name][row['id']]
            if prediction.get('error'):
                correct.append(False)
                continue
            answer = replay(row, prediction, calibrations[name])
            p = probabilities(row, answer)
            choice = max(p, key=p.__getitem__)
            if row['primitive'] == 'noul':
                choice = '1' if p['1'] >= .5 else '0'
            elif row['primitive'] == 'choice':
                choice = answer['choice']
            correct.append(choice == str(row['label']))
        counts[tuple(correct)] += 1
    n = len(records)
    wins, losses = counts[(False, True)], counts[(True, False)]
    # Resample paired outcomes, retaining their dependence. This is a
    # descriptive interval, not a predeclared significance test.
    draws = np.random.default_rng(20260922).multinomial(
        n, [wins / n, losses / n, (n - wins - losses) / n], size=20000)
    deltas = 100 * (draws[:, 0] - draws[:, 1]) / n
    return {'n': n, 'both_correct': counts[(True, True)],
            'both_wrong': counts[(False, False)], 'adapter_only_correct': wins,
            'base_only_correct': losses, 'accuracy_delta_pp': 100 * (wins - losses) / n,
            'paired_bootstrap_95_pp': np.quantile(deltas, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('artifacts/decision-v2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = read(args.training / 'run.json')
    selection = read(args.training / 'selection.json')
    summary = read(args.evaluation / 'summary.json')
    manifest = read(args.data / 'manifest.json')
    assert run['status'] in ('finished', 'bounded-stop')
    assert selection == summary['receipt']['selection']
    assert sha(args.training / 'selection.json') == summary['receipt']['selection_sha256']
    assert sha(args.data / 'manifest.json') == summary['receipt']['data_manifest_sha256']
    assert sha(args.data / 'test.jsonl') == manifest['test_sha256']
    selected = selection['selected']
    assert sha(Path(selected['native'])) == selected['native_sha256']
    assert sha(Path(selected['checkpoint'])) == selected['sha256']
    tokens = read(args.data / 'tokenized-manifest.json')
    assert tokens['data_manifest_sha256'] == sha(args.data / 'manifest.json')
    assert tokens['tokens_sha256'] == sha(args.data / 'tokenized.jsonl')
    assert run['skipped_optimizer_steps'] == 0
    prepared = rows(args.data / 'tokenized.jsonl')
    exposure = prepared[:selected['update'] * run['gradient_accumulation']]
    test = rows(args.data / 'test.jsonl')
    predictions = {name: {r['id']: r for r in rows(args.evaluation / name / 'test.jsonl')}
                   for name in ('base', 'adapter')}
    calibrations = {name: Calibration(**summary['models'][name]['calibration']['parameters'])
                    for name in predictions}
    for name in predictions:
        assert set(predictions[name]) == {r['id'] for r in test}
    held = [r for r in test if r['source'] in manifest['held_out_sources']]
    overall_paired = paired(test, predictions, calibrations)
    held_paired = paired(held, predictions, calibrations)
    assert abs(overall_paired['accuracy_delta_pp'] - summary['accuracy_delta_percentage_points']) < 1e-10
    held_accuracy = [summary['models'][name]['held_out_sources_calibrated']['overall']['accuracy_failures_incorrect']
                     for name in ('base', 'adapter')]
    assert abs(held_paired['accuracy_delta_pp'] - 100 * (held_accuracy[1] - held_accuracy[0])) < 1e-10
    result = {'training': run, 'data': manifest,
              'selected_exposure': {
                  'prompts': len(exposure),
                  'unique_records': len({r['record_id'] for r in exposure}),
                  'unique_states': len({r['state_sha256'] for r in exposure}),
                  'by_source': dict(Counter(r['source'] for r in exposure)),
                  'tokens': sum(r['input_tokens'] for r in exposure),
                  'minimum_tokens': min(r['input_tokens'] for r in exposure) if exposure else None,
                  'maximum_tokens': max(r['input_tokens'] for r in exposure) if exposure else None},
              'evaluation': summary,
              'paired_calibrated': {
                  'overall': overall_paired,
                  'held_out_sources': held_paired,
                  'interval_method': 'Paired percentile bootstrap, 20,000 resamples, NumPy seed 20260922. Descriptive; no multiplicity correction.'},
              'artifact_bytes': {key: Path(selected[key]).stat().st_size
                                 for key in ('checkpoint', 'native')},
              'artifact_hashes': {
                  str(path): sha(path) for path in (
                      args.training / 'run.json', args.training / 'selection.json',
                      args.evaluation / 'summary.json', args.data / 'manifest.json',
                      args.data / 'tokenized-manifest.json')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(compact(result), stream, indent=2)
        stream.write('\n')
    print(json.dumps(result['paired_calibrated'], indent=2))


if __name__ == '__main__':
    main()
