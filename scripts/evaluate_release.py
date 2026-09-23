"""Development-only readout selection, then a separately frozen release test."""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from prepare_benchmark import read_jsonl, write_jsonl
from shingi.backend import NativeReadout
from shingi.calibration import fit, replay
from shingi.decision import Calibration, DecisionEngine
from shingi.gpu import gpu_snapshot
from shingi.metrics import report, probabilities, wilson
from shingi.probes import probe_at_tokens
from shingi.release import artifact_identity, sha256, BASE_SHA256, ADAPTER_SHA256, require_release_environment
from prepare_training_data import APACHE_SOURCES


def ordered_rows(records, canonical):
    records = copy.deepcopy(records)
    if canonical:
        for row in records:
            if row['primitive'] == 'choice':
                row['question']['criteria'] = dict(sorted(row['question']['criteria'].items()))
    return records


def accept_canonical(input_metrics, canonical_metrics):
    return (canonical_metrics['nll']['mean'] <= input_metrics['nll']['mean'] + .05
            and canonical_metrics['accuracy_failures_incorrect'] >= input_metrics['accuracy_failures_incorrect'] - .02)


def infer(engine, records, path):
    predictions = {}
    with path.open('x') as stream:
        for index, row in enumerate(records):
            started = time.perf_counter()
            item = {k: row[k] for k in ('id', 'source', 'primitive', 'input_sha256')}
            item['error'] = None
            try:
                answer, traces = engine.answer(row['state'], row['question'])
                item.update(answer=answer, traces=traces)
            except ValueError as exc:
                item['error'] = str(exc)
            item['latency_ms'] = 1000 * (time.perf_counter() - started)
            stream.write(json.dumps(item, ensure_ascii=False) + '\n'); stream.flush()
            predictions[row['id']] = item
            if (index + 1) % 100 == 0:
                print(path.name, index + 1, flush=True)
    return predictions


def parity(records, actual, expected):
    differences, agreements = [], []
    if set(actual) != {r['id'] for r in records} or not set(actual) <= set(expected):
        raise ValueError('parity record inventory mismatch')
    for row in records:
        a, b = actual[row['id']], expected[row['id']]
        if a.get('error') or b.get('error'):
            raise ValueError('failed parity prediction')
        if a['input_sha256'] != b['input_sha256']:
            raise ValueError('parity input hash differs')
        if [t['prompt_sha256'] for t in a['traces']] != [t['prompt_sha256'] for t in b['traces']]:
            raise ValueError('parity prompt hashes differ')
        x, y = probabilities(row, a['answer']), probabilities(row, b['answer'])
        differences.append(sum(abs(x[k] - y[k]) for k in x) / 2)
        agreements.append(max(x, key=x.__getitem__) == max(y, key=y.__getitem__))
    return {'n': len(records), 'argmax_agreement': sum(agreements) / len(agreements),
            'mean_tvd': sum(differences) / len(differences), 'max_tvd': max(differences)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=('development', 'calibration', 'test'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--executable', type=Path, default=Path('artifacts/bin/readout'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--previous-evaluation', type=Path)
    p.add_argument('--protocol', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--historical-comparator', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require_release_environment()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('commit source before evaluation')
    identity = artifact_identity(a.model, a.adapter)
    adapter_hash = identity['adapter_sha256']
    if identity['model_sha256'] != BASE_SHA256:
        raise ValueError('release candidate weights differ')
    if a.phase == 'development' and adapter_hash != ADAPTER_SHA256:
        raise ValueError('legacy development candidate differs')
    data_manifest = json.loads((a.data / 'manifest.json').read_text())
    splits = ('test', 'shuffle') if a.phase == 'test' else ('dev', 'calibration')
    records = {}
    for split in splits:
        path = a.data / (split + '.jsonl')
        if sha256(path) != data_manifest[split + '_sha256']:
            raise ValueError('frozen split changed: ' + split)
        records[split] = read_jsonl(path)
    protocol = None
    if a.phase == 'test':
        if a.protocol is None: p.error('test requires a frozen --protocol')
        protocol = json.loads(a.protocol.read_text())
        if protocol['adapter_sha256'] != adapter_hash or protocol['model_sha256'] != BASE_SHA256:
            raise ValueError('protocol weights differ')
        if not a.historical_comparator and data_manifest['candidate_adapter_sha256'] != adapter_hash:
            raise ValueError('locked test candidate differs')
        if not a.historical_comparator and data_manifest.get('protocol_sha256') not in (None, sha256(a.protocol)):
            raise ValueError('locked test protocol differs')
    elif a.phase == 'calibration':
        if a.selection is None: p.error('calibration requires a frozen training selection')
        selected = json.loads(a.selection.read_text())
        training = json.loads((a.selection.parent/'run.json').read_text())
        if (selected.get('test_seen') is not False or not selected.get('development_gate_passed')
                or selected['selected']['native_sha256'] != adapter_hash
                or training['dataset_manifest_sha256'] != sha256(a.data/'manifest.json')):
            raise ValueError('training selection or data differs')
        if data_manifest.get('profile') != 'apache-v2' or set(data_manifest['train_sources']) != set(APACHE_SOURCES):
            raise ValueError('calibration requires audited Apache fitting data')
        if any(r['source'] not in APACHE_SOURCES for group in records.values() for r in group):
            raise ValueError('non-allowlisted calibration/development source')
    elif a.previous_evaluation is None:
        p.error('development requires --previous-evaluation')
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = {'phase': a.phase, 'identity': identity, 'gpu': gpu_snapshot(),
               'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'executable_sha256': sha256(a.executable), 'data_manifest_sha256': sha256(a.data / 'manifest.json'),
               'protocol_sha256': sha256(a.protocol) if a.protocol else None,
               'started_at': datetime.now(timezone.utc).isoformat(), 'context_tokens': 16384, 'kv': 'q8_0'}
    (a.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    summary = {'receipt': receipt, 'models': {}}
    development = {}
    for name, adapter in (('base', None), ('adapter', a.adapter)):
        dest = a.output / name; dest.mkdir()
        backend = NativeReadout(a.executable, a.model, 16384, adapter=adapter)
        try:
            engine = DecisionEngine(backend)
            # Batch-one native canary before the complete phase.
            canary = {'type': 'choice', 'instructions': 'Read the selected key.', 'criteria': {'red': None, 'blue': None}}
            result, _ = engine.answer({'selected_key': 'blue'}, canary)
            if result['choice'] != 'blue':
                raise RuntimeError('native batch-one canary failed')
            if a.phase == 'calibration':
                engine = DecisionEngine(backend, canonical_choices=True)
                canonical_dev = ordered_rows(records['dev'], True)
                native = infer(engine, canonical_dev, dest/'dev-canonical.jsonl')
                metrics = report(canonical_dev, native)
                if metrics['overall']['valid'] != len(canonical_dev): raise RuntimeError('invalid native development')
                step = selected['selected']['update'] if adapter else 0
                differentiable = json.loads((a.selection.parent/f'dev-predictions-{step:04}.json').read_text())
                agreements, tvds = [], []
                for row in canonical_dev:
                    x = probabilities(row, native[row['id']]['answer'])
                    y = probabilities(row, differentiable[row['id']]['answer'])
                    agreements.append(max(x, key=x.get) == max(y, key=y.get))
                    tvds.append(sum(abs(x[k]-y[k]) for k in x)/2)
                check = {'n':len(tvds), 'argmax_agreement':sum(agreements)/len(tvds),
                         'mean_tvd':sum(tvds)/len(tvds), 'max_tvd':max(tvds)}
                development[name] = {'canonical':metrics, 'training_native_parity':check}
                summary['models'][name] = development[name]
                calibration_rows = ordered_rows(records['calibration'], True)
                predictions = infer(engine, calibration_rows, dest/'calibration-canonical.jsonl')
                fitted = fit(calibration_rows, predictions)
                fitted['provenance'] = {'model_sha256':BASE_SHA256, 'adapter_sha256':adapter_hash if adapter else None,
                    'readout_version':'bonsai-sorted-choice-v2', 'profile':'apache-v2',
                    'calibration_data_sha256':data_manifest['calibration_sha256'],
                    'dataset_manifest_sha256':receipt['data_manifest_sha256'],
                    'predictions_sha256':sha256(dest/'calibration-canonical.jsonl'),
                    'source_revision':receipt['source_revision'], 'dataset_revision':data_manifest['revision']}
                (dest/'calibration-canonical.json').write_text(json.dumps(fitted, indent=2)+'\n')
            elif a.phase == 'development':
                native = infer(engine, records['dev'], dest / 'dev-input.jsonl')
                previous = {r['id']: r for r in read_jsonl(a.previous_evaluation / name / 'dev.jsonl')}
                check = parity(records['dev'], native, previous)
                (dest / 'reproduction.json').write_text(json.dumps(check, indent=2) + '\n')
                if check['argmax_agreement'] < .99 or check['mean_tvd'] > .01:
                    raise RuntimeError('clean-runtime reproduction gate failed')
                canonical_dev = ordered_rows(records['dev'], True)
                canonical = infer(engine, canonical_dev, dest / 'dev-canonical.jsonl')
                development[name] = {'input': report(records['dev'], native),
                                     'canonical': report(canonical_dev, canonical), 'reproduction': check}
                # Generate both calibration variants before selection; no test records are opened.
                for mode in ('input', 'canonical'):
                    calibration_rows = ordered_rows(records['calibration'], mode == 'canonical')
                    predictions = infer(engine, calibration_rows, dest / ('calibration-' + mode + '.jsonl'))
                    fitted = fit(calibration_rows, predictions)
                    fitted['provenance'] = {'model_sha256': BASE_SHA256,
                        'adapter_sha256': ADAPTER_SHA256 if adapter else None,
                        'readout_version': 'bonsai-sorted-choice-v2' if mode == 'canonical' else 'bonsai-letter-v1',
                        'calibration_data_sha256': data_manifest['calibration_sha256'],
                        'dataset_manifest_sha256': receipt['data_manifest_sha256'],
                        'predictions_sha256': sha256(dest / ('calibration-' + mode + '.jsonl')),
                        'source_revision': receipt['source_revision'], 'dataset_revision': data_manifest['revision']}
                    (dest / ('calibration-' + mode + '.json')).write_text(json.dumps(fitted, indent=2) + '\n')
                summary['models'][name] = development[name]
            else:
                canonical = protocol['choice_order'] == 'canonical'
                calibration = Calibration(**protocol['calibrations'][name]['parameters'])
                engine = DecisionEngine(backend, calibration, canonical_choices=canonical)
                test = ordered_rows(records['test'], canonical)
                shuffled = ordered_rows(records['shuffle'], canonical)
                predictions = infer(engine, test, dest / 'test.jsonl')
                permutations = infer(engine, shuffled, dest / 'shuffle.jsonl')
                pairs = []
                for row in shuffled:
                    x, y = predictions[row['id']], permutations[row['id']]
                    if x['error'] or y['error']:
                        raise RuntimeError('invalid order pair')
                    px, py = x['answer']['probabilities'], y['answer']['probabilities']
                    pairs.append({'id': row['id'], 'options': len(px), 'flip': x['answer']['choice'] != y['answer']['choice'],
                                  'tvd': sum(abs(px[k] - py[k]) for k in px) / 2})
                write_jsonl(dest / 'order-pairs.jsonl', pairs)
                # A separate input-order diagnostic retains the underlying model limitation.
                if canonical:
                    ids = {r['id'] for r in records['shuffle']}
                    input_rows = [r for r in records['test'] if r['id'] in ids]
                    diagnostic_engine = DecisionEngine(backend, calibration)
                    raw_a = infer(diagnostic_engine, input_rows, dest / 'order-diagnostic-input.jsonl')
                    raw_b = infer(diagnostic_engine, records['shuffle'], dest / 'order-diagnostic-shuffle.jsonl')
                    diagnostic_flips = sum(raw_a[r['id']]['answer']['choice'] != raw_b[r['id']]['answer']['choice'] for r in input_rows)
                else:
                    diagnostic_flips = sum(r['flip'] for r in pairs)
                context = []
                for tokens in (512, 4096, 8192, 15000):
                    for i, position in enumerate((0., .5, 1.)):
                        row, count = probe_at_tokens(backend, tokens, position, seed=tokens+i)
                        answer, traces = engine.answer(row['state'], row['question'])
                        context.append({'target_tokens': tokens, 'input_tokens': count, 'position': position,
                                        'correct': answer['choice'] == row['label'], 'answer': answer, 'traces': traces})
                (dest / 'context.json').write_text(json.dumps(context, indent=2) + '\n')
                metrics = report(test, predictions)
                if metrics['overall']['valid'] != len(test):
                    raise RuntimeError('release test contains failed predictions')
                if name == 'adapter' and not all(r['correct'] for r in context):
                    raise RuntimeError('context gate failed')
                summary['models'][name] = {'calibrated': metrics,
                    'order': {'pairs': len(pairs), 'flips': sum(r['flip'] for r in pairs),
                              'mean_tvd': sum(r['tvd'] for r in pairs) / len(pairs),
                              'input_order_diagnostic_flips': diagnostic_flips},
                    'context': {'correct': sum(r['correct'] for r in context), 'n': len(context)}}
        finally:
            backend.close()
        (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if a.phase == 'calibration':
        base, candidate = [development[name]['canonical']['overall'] for name in ('base', 'adapter')]
        if (candidate['nll']['mean'] >= base['nll']['mean']
                or candidate['accuracy_failures_incorrect'] < base['accuracy_failures_incorrect']-.02):
            raise RuntimeError('native development gate failed; test remains locked')
        protocol = {'model_sha256':BASE_SHA256, 'adapter_sha256':adapter_hash,
                    'choice_order':'canonical', 'readout_version':'bonsai-sorted-choice-v2', 'profile':'apache-v2',
                    'train_sources':list(APACHE_SOURCES), 'adapter_license':'apache-2.0',
                    'selection':'Fresh LoRA selected by uncalibrated canonical development NLL; fixed readout.',
                    'training_selection_sha256':sha256(a.selection),
                    'dev_summary_sha256':sha256(a.output/'summary.json'), 'calibrations':{}}
        for name in ('base', 'adapter'):
            protocol['calibrations'][name] = json.loads((a.output/name/'calibration-canonical.json').read_text())
        (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
        print(json.dumps({'native_development_gate_passed':True, 'models':development}, indent=2))
    elif a.phase == 'development':
        x, y = [development['adapter'][mode]['overall'] for mode in ('input', 'canonical')]
        accepted = accept_canonical(x, y)
        mode = 'canonical' if accepted else 'input'
        protocol = {'model_sha256': BASE_SHA256, 'adapter_sha256': ADAPTER_SHA256,
                    'choice_order': mode, 'readout_version': 'bonsai-sorted-choice-v2' if accepted else 'bonsai-letter-v1',
                    'selection': 'Development NLL regression <=0.05 and accuracy regression <=2 percentage points.',
                    'dev_summary_sha256': sha256(a.output / 'summary.json'), 'calibrations': {}}
        for name in ('base', 'adapter'):
            protocol['calibrations'][name] = json.loads((a.output / name / ('calibration-' + mode + '.json')).read_text())
        (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        print(json.dumps({'canonical_accepted': accepted, 'input_dev': x, 'canonical_dev': y}, indent=2))
    print('Completed', a.phase, flush=True)


if __name__ == '__main__':
    main()
