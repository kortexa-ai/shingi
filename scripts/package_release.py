"""Derive public evidence/charts and assemble an explicit, checksum-verified HF bundle."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import urllib.request

from evaluate_release import ordered_rows
from prepare_benchmark import read_jsonl
from shingi.decision import Calibration
from shingi.release import sha256, BASE_SHA256, ADAPTER_SHA256, RELEASE_MODEL_ID
from shingi.metrics import report as quality_report, probabilities
from shingi.calibration import replay
from summarize_training import paired

HELD = {'ledgar', 'go_emotions', 'mnli', 'sst5', 'fever_evidence', 'sms_spam', 'chaosnli'}
LABELS = {'banking77': 'Banking77', 'clinc150': 'CLINC150', 'ledgar': 'LEDGAR',
          'go_emotions': 'GoEmotions', 'mmlu': 'MMLU', 'arc_challenge': 'ARC Challenge',
          'mnli': 'MNLI', 'sst5': 'SST-5', 'helpsteer2_helpfulness': 'HelpSteer2',
          'measuring_hate_speech': 'Measuring Hate Speech', 'boolq': 'BoolQ',
          'fever_evidence': 'FEVER Evidence', 'civil_comments': 'Civil Comments',
          'sms_spam': 'SMS Spam', 'chaosnli': 'ChaosNLI'}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def validate_receipt(receipt, card, protocol_hash, data_hash, adapter_hash=ADAPTER_SHA256):
    if receipt['protocol_sha256'] != protocol_hash:
        raise ValueError('run protocol differs')
    if receipt.get('data_manifest_sha256', receipt.get('dataset_manifest_sha256')) != data_hash:
        raise ValueError('run data manifest differs')
    identity = receipt['identity']
    if identity['model_sha256'] != BASE_SHA256 or identity['adapter_sha256'] != adapter_hash:
        raise ValueError('run weights differ')
    if card not in receipt['gpu']['name']:
        raise ValueError('run GPU differs')
    if receipt['kv'] != 'q8_0' or receipt['context_tokens'] != 16384:
        raise ValueError('run inference settings differ')


def load_predictions(path, records):
    rows = read_jsonl(path)
    predictions = {r['id']: r for r in rows}
    if len(rows) != len(predictions) or set(predictions) != {r['id'] for r in records}:
        raise ValueError('prediction inventory differs')
    for row in records:
        prediction = predictions[row['id']]
        if prediction.get('error') or prediction['input_sha256'] != row['input_sha256']:
            raise ValueError('failed or mismatched prediction')
    return predictions


def cross_device_parity(records, actual, expected, calibration):
    """Compare final decisions while validating each adaptive readout path."""
    if set(actual) != {r['id'] for r in records} or set(actual) != set(expected):
        raise ValueError('cross-device record inventory differs')
    differences, agreements, branches = [], [], []
    for row in records:
        a, b = actual[row['id']], expected[row['id']]
        # Replay verifies every prompt hash and candidate count, including the
        # final comparison constructed from that device's own chunk winners.
        x = replay(row, a, calibration)
        y = replay(row, b, calibration)
        if not replay_matches(x, a['answer']) or not replay_matches(y, b['answer']):
            raise ValueError('recorded answer differs from trace replay')
        ta, tb = a['traces'], b['traces']
        if len(ta) != len(tb) or any(p['candidate_ids'] != q['candidate_ids'] for p,q in zip(ta,tb)):
            raise ValueError('cross-device candidate token IDs differ')
        same = [p['prompt_sha256'] == q['prompt_sha256'] for p,q in zip(ta,tb)]
        if not all(same):
            if len(same) == 1 or not all(same[:-1]):
                raise ValueError('static input prompts differ across devices')
            branches.append(row['id'])
        px, py = probabilities(row, a['answer']), probabilities(row, b['answer'])
        differences.append(sum(abs(px[k]-py[k]) for k in px)/2)
        agreements.append(max(px,key=px.__getitem__) == max(py,key=py.__getitem__))
    return {'n': len(records), 'argmax_agreement': sum(agreements)/len(records),
            'argmax_disagreements': len(records)-sum(agreements),
            'mean_tvd': sum(differences)/len(records), 'max_tvd': max(differences),
            'max_tvd_id': records[max(range(len(differences)), key=differences.__getitem__)]['id'],
            'adaptive_final_prompt_differences': len(branches), 'adaptive_difference_ids': branches,
            'replay_answer_tolerance': 1e-12,
            'method': 'Identical request and static chunk prompts; each device adaptive trace independently replayed. Saved final decision distributions compared.'}


def replay_matches(actual, expected):
    # Python/libm on different hosts can differ in the last floating-point bit.
    # Keep selected keys and structure exact; permit only tiny numeric drift.
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(replay_matches(actual[k], expected[k]) for k in actual)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(replay_matches(a,b) for a,b in zip(actual,expected))
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return actual == expected


def metric_changes(previous, current):
    """Expose every measured regression; do not hide losses behind an aggregate."""
    changes = []
    for scope in ('overall', *sorted(previous['by_source'])):
        old = previous['overall'] if scope == 'overall' else previous['by_source'][scope]
        new = current['overall'] if scope == 'overall' else current['by_source'][scope]
        if old['n'] != new['n']: raise ValueError('comparison denominators differ')
        for metric in ('accuracy_failures_incorrect', 'nll', 'brier_multiclass', 'brier_binary',
                       'score_mae', 'rps', 'human_tvd', 'ece_top_label'):
            x, y = old[metric], new[metric]
            if isinstance(x, dict):
                if x['n'] != y['n']: raise ValueError('comparison metric denominators differ')
                x, y = x['mean'], y['mean']
            if x is None or y is None: continue
            delta = y-x
            worse = delta < -1e-12 if metric == 'accuracy_failures_incorrect' else delta > 1e-12
            changes.append({'scope':scope, 'metric':metric, 'previous':x, 'current':y,
                            'delta':delta, 'worse':worse, 'n':old['n']})
    return changes


def compare_previous(a, card, records, candidate_predictions, candidate_metrics, candidate_speed, calibration):
    directory = getattr(a, 'previous_evaluation_'+card)
    speed_dir = getattr(a, 'previous_speed_'+card)
    previous_protocol = read(a.previous_protocol)
    summary = read(directory/'summary.json'); speed = read(speed_dir/'summary.json')
    for receipt in (summary['receipt'], speed['receipt']):
        validate_receipt(receipt, card, sha256(a.previous_protocol), sha256(a.data/'manifest.json'),
                         previous_protocol['adapter_sha256'])
    if previous_protocol['choice_order'] != 'canonical': raise ValueError('previous readout differs')
    predictions = load_predictions(directory/'adapter/test.jsonl', records)
    metrics = quality_report(records, predictions)
    if not replay_matches(metrics, summary['models']['adapter']['calibrated']):
        raise ValueError('previous summary differs from raw predictions')
    if speed['receipt']['mixed_ids'] != candidate_speed['receipt']['mixed_ids']:
        raise ValueError('previous speed workload differs')
    differences = metric_changes(metrics, candidate_metrics)
    uncalibrated = {}
    for name, values in (('previous', predictions), ('current', candidate_predictions)):
        raw = {row['id']:{'answer':replay(row, values[row['id']], Calibration())} for row in records}
        uncalibrated[name] = quality_report(records, raw)
    uncalibrated['metric_changes'] = metric_changes(uncalibrated['previous'], uncalibrated['current'])
    latency = []
    for group, stats in candidate_speed['groups'].items():
        for metric in ('median', 'p95'):
            x, y = speed['groups'][group]['wall_ms'][metric], stats['wall_ms'][metric]
            latency.append({'group':group, 'metric':metric, 'previous_ms':x, 'current_ms':y,
                            'delta_ms':y-x, 'delta_percent':100*(y/x-1), 'worse':y>x})
    result = {'previous_protocol':previous_protocol, 'quality':metrics, 'speed':speed,
              'paired':paired(records, {'base':predictions, 'adapter':candidate_predictions},
                 {'base':Calibration(**previous_protocol['calibrations']['adapter']['parameters']), 'adapter':calibration}),
              'metric_changes':differences, 'regressions':[r for r in differences if r['worse']],
              'uncalibrated':uncalibrated,
              'latency_changes':latency,
              'order_previous':summary['models']['adapter']['order'],
              'source_revision':summary['receipt']['source_revision'],
              'inputs':{str(path):sha256(path) for path in (a.previous_protocol, directory/'summary.json',
                  directory/'adapter/test.jsonl', speed_dir/'summary.json', speed_dir/'observations.jsonl', speed_dir/'memory-samples.json')},
              'interpretation':'Paired accuracy interval is descriptive. Small per-source and timing changes can be sampling noise; no tuning followed this test.'}
    return result


def charts(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'svg.hashsalt': 'shingi-v01'})
    colors = {'base': '#77818B', 'adapter': '#087F8C', '6000': '#087F8C', '4090': '#C66A2B'}
    figures = output / 'figures'; figures.mkdir()
    def export(fig, name):
        fig.savefig(figures / (name + '.png'), bbox_inches='tight')
        fig.savefig(figures / (name + '.svg'), bbox_inches='tight', metadata={'Date': None})
        plt.close(fig)
    models = result['quality']['6000']['models']
    sources = list(models['base']['calibrated']['by_source'])
    fig, ax = plt.subplots(figsize=(10, 9), layout='constrained')
    for name, offset, label in [('base', -.13, 'Unchanged Bonsai'), ('adapter', .13, 'Shingi v0.1')]:
        metrics = models[name]['calibrated']['by_source']
        values = np.array([100 * metrics[s]['accuracy_failures_incorrect'] for s in sources])
        low = np.array([100 * metrics[s]['accuracy_wilson_95'][0] for s in sources])
        high = np.array([100 * metrics[s]['accuracy_wilson_95'][1] for s in sources])
        ax.errorbar(values, np.arange(len(sources)) + offset, xerr=[values-low, high-values],
                    fmt='o', markersize=5, capsize=2, color=colors[name], label=label, alpha=.95)
    ax.set_yticks(np.arange(len(sources)), [LABELS[s] + (' *' if s in HELD else '') for s in sources])
    ax.invert_yaxis(); ax.set_xlim(0, 104); ax.set_xlabel('Accuracy (%) · Wilson 95% intervals')
    ax.grid(axis='x', alpha=.18); ax.legend(loc='lower left')
    base, adapter = [models[n]['calibrated']['overall']['accuracy_failures_incorrect'] * 100 for n in ('base','adapter')]
    ax.set_title(f'Shingi v0.1: {adapter:.2f}% vs {base:.2f}% unchanged base\n'
                 'Fresh 1,500 decisions · 100 per source · RTX PRO 6000', loc='left', pad=20)
    fig.supxlabel('* Source excluded from adapter training, development and calibration.', fontsize=10)
    export(fig, 'accuracy')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    axes[0].plot([0,1], [0,1], '--', color='#B9C0C5', label='Perfect calibration')
    for name, label in [('base','Unchanged Bonsai'), ('adapter','Shingi v0.1')]:
        bins = models[name]['calibrated']['overall']['reliability_bins']
        nonempty = [b for b in bins if b['n']]
        axes[0].plot([b['probability'] for b in nonempty], [b['accuracy'] for b in nonempty],
                     'o-', color=colors[name], label=label)
    axes[0].set(xlim=(0,1), ylim=(0,1), xlabel='Mean top-label probability', ylabel='Observed accuracy', title='Probability calibration')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=.15)
    metrics = [('nll','NLL\n1,500'), ('brier_multiclass','Brier\n1,500'), ('score_mae','Score MAE\n300')]
    for name, offset in [('base',-.18), ('adapter',.18)]:
        values = [models[name]['calibrated']['overall'][k]['mean'] for k, _ in metrics]
        bars = axes[1].bar(np.arange(3)+offset, values, width=.34, color=colors[name])
        axes[1].bar_label(bars, fmt='%.3f', fontsize=9, padding=3)
    axes[1].set_xticks(np.arange(3), [label for _,label in metrics])
    axes[1].set_title('Probability and score errors · lower is better')
    axes[1].set_ylabel('Mean error (different metric scales)'); axes[1].set_ylim(0, axes[1].get_ylim()[1]*1.12)
    fig.supxlabel('Separate frozen calibration per model; same fresh test. Reliability bins have unequal sample counts.', fontsize=10)
    export(fig, 'calibration')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for card in ('6000','4090'):
        groups = result['speed'][card]['groups']
        for ax, prefix, values in [(axes[0], 'context-', [512,2048,8192,15000]), (axes[1], 'options-', [4,52,53,100,255])]:
            med = [groups[prefix+str(v)]['wall_ms']['median'] for v in values]
            tail = [groups[prefix+str(v)]['wall_ms']['p95'] for v in values]
            ax.plot(values, med, 'o-', color=colors[card], label='RTX PRO 6000' if card=='6000' else 'RTX 4090')
            ax.fill_between(values, med, tail, color=colors[card], alpha=.14)
            ax.set_yscale('log'); ax.grid(alpha=.18); ax.set_ylabel('Decision latency (ms, log scale)')
    axes[0].set(xlabel='Approximate prompt tokens · four choices', title='Context length')
    axes[1].set(xlabel='Choice count · short prompt', title='Choice count and extra forward passes')
    axes[0].legend(); axes[1].axvline(52, color='#899099', linestyle=':', alpha=.7)
    fig.suptitle('Shingi v0.1 · Sequential CUDA decisions', fontsize=14)
    fig.supxlabel('Median lines; shaded to sample p95. Ten sequential repeats per case, after warmup. Model loading excluded.', fontsize=10)
    export(fig, 'latency')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    labels = ['RTX PRO 6000', 'RTX 4090']
    for offset, stat, color in [(-.18, 'median', '#087F8C'), (.18, 'p95', '#8BBDB8')]:
        values = [result['speed'][card]['groups']['mixed']['wall_ms'][stat] for card in ('6000','4090')]
        bars = axes[0].bar(np.arange(2)+offset, values, .34, label=stat, color=color)
        axes[0].bar_label(bars, fmt='%.1f', padding=3)
    axes[0].set_xticks([0,1], labels); axes[0].set(ylabel='Decision wall time (ms)', title='Mixed workload · 900 sequential requests'); axes[0].legend()
    axes[0].set_ylim(0, axes[0].get_ylim()[1]*1.18)
    for offset, phases, label, color in [(-.18, ('load',), 'Load', '#77818B'), (.18, ('mixed','synthetic','warmup'), 'Inference', '#087F8C')]:
        values = [max(result['speed'][card]['memory']['phases'][p]['max_gpu_delta_mib'] for p in phases)/1024 for card in ('6000','4090')]
        bars = axes[1].bar(np.arange(2)+offset, values, .34, label=label, color=color)
        axes[1].bar_label(bars, fmt='%.2f', padding=3)
    axes[1].set_xticks([0,1], labels); axes[1].set(ylabel='Sampled GPU allocation delta (GiB)', title='Memory including context and buffers'); axes[1].legend()
    axes[1].set_ylim(0, axes[1].get_ylim()[1]*1.24)
    fig.suptitle('Shingi v0.1 · Measured CUDA latency and memory', fontsize=14)
    fig.supxlabel('300 fixed records × three repeats; ten warmups. Memory sampled every 100 ms; these are not continuous peaks.', fontsize=10)
    export(fig, 'hardware')


def report_release(a):
    protocol = read(a.protocol); manifest = read(a.data / 'manifest.json')
    if sha256(a.data / 'test.jsonl') != manifest['test_sha256']:
        raise ValueError('test split changed')
    records = ordered_rows(read_jsonl(a.data / 'test.jsonl'), protocol['choice_order']=='canonical')
    result = {'model': RELEASE_MODEL_ID, 'protocol': protocol, 'data': manifest,
              'report_generator_sha256': sha256(Path(__file__)),
              'held_out_sources':sorted(set(LABELS)-set(protocol['train_sources'])) if 'train_sources' in protocol else sorted(HELD),
              'quality': {}, 'speed': {}, 'api': {}, 'cross_device': {}, 'paired': {}, 'inputs': {}, 'previous': {}}
    predictions = {}
    for card in ('6000','4090'):
        quality = getattr(a, 'evaluation_' + card)
        speed = getattr(a, 'speed_' + card)
        api = getattr(a, 'api_' + card)
        result['quality'][card] = read(quality / 'summary.json')
        result['speed'][card] = read(speed / 'summary.json')
        result['api'][card] = read(api / 'result.json')
        for kind, directory, filename in [('quality',quality,'summary.json'), ('speed',speed,'summary.json'), ('api',api,'result.json')]:
            result['inputs'][card+'/'+kind] = sha256(directory / filename)
        for name in ('base','adapter'):
            for filename in ('test.jsonl','shuffle.jsonl','order-pairs.jsonl','order-diagnostic-input.jsonl','order-diagnostic-shuffle.jsonl','context.json'):
                result['inputs'][card+'/quality/'+name+'/'+filename] = sha256(quality / name / filename)
        for filename in ('observations.jsonl','memory-samples.json'):
            result['inputs'][card+'/speed/'+filename] = sha256(speed / filename)
        for kind in ('quality','speed'):
            validate_receipt(result[kind][card]['receipt'], card, sha256(a.protocol), sha256(a.data / 'manifest.json'))
        if not result['api'][card].get('passed'):
            raise ValueError('API smoke did not pass')
        version = result['api'][card]['version']
        if version['model_sha256'] != BASE_SHA256 or version['adapter_sha256'] != ADAPTER_SHA256:
            raise ValueError('API weights differ')
        if version['calibration'] != protocol['calibrations']['adapter']['parameters']:
            raise ValueError('API calibration differs')
        predictions[card] = {name: load_predictions(quality / name / 'test.jsonl', records) for name in ('base','adapter')}
        for name, rows in predictions[card].items():
            recomputed = quality_report(records, rows)
            if not replay_matches(recomputed, result['quality'][card]['models'][name]['calibrated']) or recomputed['overall']['valid'] != len(records):
                raise ValueError('quality summary differs from raw predictions')
        model = result['quality'][card]['models']['adapter']
        if model['context'] != {'correct': 12, 'n': 12} or model['order']['pairs'] != 200 or model['order']['flips']:
            raise ValueError('context or API order gate failed')
        speed_result = result['speed'][card]
        if speed_result['groups']['mixed']['wall_ms']['n'] != 900:
            raise ValueError('incomplete mixed benchmark')
        if any(speed_result['groups'][name]['wall_ms']['n'] != 10 for name in (
                'context-512','context-2048','context-8192','context-15000','options-4','options-52','options-53','options-100','options-255')):
            raise ValueError('incomplete synthetic benchmark')
        floor = 10*1024 if card == '6000' else 4*1024
        if min(p['minimum_free_mib'] for p in speed_result['memory']['phases'].values()) < floor:
            raise ValueError('sampled memory headroom gate failed')
        calibrations = {name: Calibration(**protocol['calibrations'][name]['parameters']) for name in ('base','adapter')}
        result['paired'][card] = {'overall': paired(records, predictions[card], calibrations),
            'held_out_sources': paired([r for r in records if r['source'] in result['held_out_sources']], predictions[card], calibrations)}
        if protocol.get('profile') == 'apache-v2':
            if not a.previous_protocol or not getattr(a, 'previous_evaluation_'+card) or not getattr(a, 'previous_speed_'+card):
                raise ValueError('Apache replacement requires matched previous-adapter measurements')
            current = result['quality'][card]['models']['adapter']['calibrated']
            base = result['quality'][card]['models']['base']['calibrated']['overall']
            if current['overall']['accuracy_failures_incorrect'] <= base['accuracy_failures_incorrect'] or current['overall']['nll']['mean'] >= base['nll']['mean']:
                raise ValueError('locked-test quality gate against unchanged base failed')
            result['previous'][card] = compare_previous(a, card, records, predictions[card]['adapter'], current,
                                                       speed_result, calibrations['adapter'])
    if len({result[kind][card]['receipt']['source_revision'] for kind in ('quality','speed') for card in ('6000','4090')}) != 1:
        raise ValueError('measurement source revisions differ')
    for name in ('base','adapter'):
        result['cross_device'][name] = cross_device_parity(records, predictions['6000'][name], predictions['4090'][name],
                                                         Calibration(**protocol['calibrations'][name]['parameters']))
        check = result['cross_device'][name]
        if check['argmax_agreement'] < .99 or check['mean_tvd'] > .01:
            raise ValueError('cross-device parity gate failed')
    a.output.mkdir(parents=True, exist_ok=False)
    # Keep the full metrics; only raw prompts/predictions stay outside the public report.
    save(a.output / 'summary.json', result)
    charts(result, a.output)
    with (a.output / 'accuracy.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['source','held_out_from_adaptation','gpu','model','n','correct','accuracy','wilson_low','wilson_high','nll'])
        for card in ('6000','4090'):
            for name in ('base','adapter'):
                for source, m in result['quality'][card]['models'][name]['calibrated']['by_source'].items():
                    writer.writerow([source,source in result['held_out_sources'],card,name,m['n'],m['correct'],m['accuracy_failures_incorrect'],*m['accuracy_wilson_95'],m['nll']['mean']])
    write_report(result, a.output)
    print(json.dumps({'paired': result['paired'], 'cross_device': result['cross_device']}, indent=2))


def write_report(result, output):
    models = result['quality']['6000']['models']
    base, adapter = [models[n]['calibrated']['overall'] for n in ('base','adapter')]
    delta = result['paired']['6000']['overall']
    held = result['paired']['6000']['held_out_sources']
    lines = ['# Shingi CUDA v0.1 evaluation', '',
             f"Fresh test: **{adapter['correct']}/1,500 ({100*adapter['accuracy_failures_incorrect']:.2f}%)** Shingi versus **{base['correct']}/1,500 ({100*base['accuracy_failures_incorrect']:.2f}%)** unchanged Bonsai on the RTX PRO 6000.", '',
             f"The paired change is {delta['accuracy_delta_pp']:+.2f} percentage points, with a descriptive paired-bootstrap 95% interval [{delta['paired_bootstrap_95_pp'][0]:+.2f}, {delta['paired_bootstrap_95_pp'][1]:+.2f}]. The seven-source holdout change is {held['accuracy_delta_pp']:+.2f} points [{held['paired_bootstrap_95_pp'][0]:+.2f}, {held['paired_bootstrap_95_pp'][1]:+.2f}].", '',
             '![Accuracy by source](figures/accuracy.png)', '',
             '| Metric, calibrated | Unchanged Bonsai | Shingi v0.1 |', '|---|---:|---:|']
    for key, title in [('nll','NLL'), ('brier_multiclass','Multiclass Brier'), ('brier_binary','Binary Brier'), ('score_mae','Expected-score MAE'), ('rps','Ordinal RPS'), ('human_tvd','Distance to human labels')]:
        lines.append(f"| {title} (n={adapter[key]['n']}) | {base[key]['mean']:.5f} | {adapter[key]['mean']:.5f} |")
    lines += [f"| Top-label ECE (n=1,500) | {base['ece_top_label']:.5f} | {adapter['ece_top_label']:.5f} |", '',
              '![Calibration and score quality](figures/calibration.png)', '',
              '| GPU, same 1,500 inputs | Unchanged Bonsai correct | Shingi correct |', '|---|---:|---:|']
    for card in ('6000','4090'):
        m = result['quality'][card]['models']
        counts = [m[n]['calibrated']['overall']['correct'] for n in ('base','adapter')]
        lines.append(f'| {card} | {counts[0]}/1,500 ({counts[0]/15:.2f}%) | {counts[1]}/1,500 ({counts[1]/15:.2f}%) |')
    lines += ['', '| Cross-GPU comparison | Argmax agreement | Mean / maximum probability TVD |', '|---|---:|---:|']
    for name in ('base','adapter'):
        check = result['cross_device'][name]
        lines.append(f"| {name} | {100*check['argmax_agreement']:.4f}% | {check['mean_tvd']:.7f} / {check['max_tvd']:.7f} |")
    lines += ['',
              f"Both models pass the predeclared 99% agreement / 0.01 mean-TVD gates, but this is not exact numerical equivalence. Different final adaptive prompts occur in {result['cross_device']['base']['adaptive_final_prompt_differences']} base and {result['cross_device']['adapter']['adaptive_final_prompt_differences']} adapter requests after a chunk winner changes. Every static chunk prompt and candidate token ID matches across devices; every adaptive trace is verified independently by replay. Small hardware-dependent logit changes can cause a large individual probability shift in this approximate method; the maximum TVD is reported above and its record ID is in the JSON.", '',
              'Choice keys are sorted before inference. Map-order stability is an interface property, not learned invariance. The additional input-order diagnostic varies the underlying prompt order.', '',
              '| GPU | Map-order flips, base / adapter | Input-order diagnostic flips, base / adapter | Adapter context probes |', '|---|---:|---:|---:|']
    for card in ('6000','4090'):
        m = result['quality'][card]['models']
        lines.append(f"| {card} | {m['base']['order']['flips']}/200 / {m['adapter']['order']['flips']}/200 | {m['base']['order']['input_order_diagnostic_flips']}/200 / {m['adapter']['order']['input_order_diagnostic_flips']}/200 | {m['adapter']['context']['correct']}/12 |")
    lines += ['', '| GPU | Mixed median / p95 | Short HTTP median / p95 | Model initialization |', '|---|---:|---:|---:|']
    for card in ('6000','4090'):
        speed = result['speed'][card]; m = speed['groups']['mixed']['wall_ms']; api = result['api'][card]['short_http_latency_ms']
        lines.append(f"| {card} | {m['median']:.1f} / {m['p95']:.1f} ms | {api['median']:.1f} / {api['p95']:.1f} ms | {speed['model_initialization_seconds']:.2f} s |")
    lines += ['', '![Latency curves](figures/latency.png)', '', '![Hardware measurements](figures/hardware.png)', '',
              '## Method and limits', '',
              '- 100 fresh records from each of 15 sources; 6,930 prior IDs and state hashes excluded. The aggregate weights sources equally and is not a user-workload population estimate.',
              '- Existing update-128 adapter; frozen development-selected readout and separate 240-record calibration. No release-test tuning. Upstream pretraining overlap is unknown.',
              '- Choice sorting was selected with a predeclared development gate. Adapter development accuracy changed from 212/256 to 209/256; NLL from 0.509250 to 0.520816. It avoids caller map-order effects at this measured development tradeoff.',
              '- Paired intervals use 20,000 bootstrap draws, seed 20260922; descriptive, without multiplicity correction. Source slices have only 100 records.',
              '- Mixed latency: 300 fixed records, 20/source, repeated three times after ten warmups. Synthetic curves: ten repetitions per case after warmup. Short HTTP: 30 localhost SDK requests after three warmups; one three-choice question.',
              '- Latency percentiles use sorted values at index round((n−1) × p). With ten repeats, the sample p95 is the maximum. Curves are workload measurements, not population tail-latency guarantees.',
              '- Wall times include all forward passes, tokenization and memory checks; loading is separate. Model initialization times exclude artifact checksum hashing. OS filesystem cache is uncontrolled. Device memory is sampled every 100 ms; a brief peak can be missed. Host RSS is sampled only after native readiness, so it does not establish a host load-time peak. Other resident processes are recorded in private operator evidence.',
              '- Synthetic context probes test simple planted-fact retrieval through about 15K tokens. They do not establish real-document comprehension.',
              '- Text/JSON only; sequential serving. More than 52 choices use approximate multi-pass chunk-and-anchor scoring. Labels and wording can still affect decisions.',
              '- No Mac result, hosted Jev request, or hosted latency comparison is included.', '',
              'Exact metrics, GPU/driver information, file hashes, source revisions, calibration and data identifiers are in `summary.json`. `accuracy.csv` contains both devices\' per-source results. Raw predictions and traces are retained separately.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def assemble(a):
    if subprocess.check_output(['git','status','--porcelain'], text=True).strip():
        raise ValueError('commit release sources before assembling')
    if sha256(a.adapter) != ADAPTER_SHA256:
        raise ValueError('adapter checksum differs')
    report = read(a.report / 'summary.json')
    protocol = read('release/manifest.json')
    if report['protocol'] != protocol:
        raise ValueError('report and packaged protocol differ')
    if read('release/calibration.json') != protocol['calibrations']['adapter']:
        raise ValueError('packaged calibration differs')
    validation_path = Path('results/cuda-v0.1/validation.json')
    if read(validation_path)['evaluation_summary_sha256'] != sha256(a.report / 'summary.json'):
        raise ValueError('publication audit belongs to a different report')
    a.output.mkdir(parents=True, exist_ok=False)
    for source, dest in [(a.adapter,'adapter.gguf'), (Path('release/calibration.json'),'calibration.json'),
                         (Path('release/manifest.json'),'protocol.json'), (Path('release/README.md'),'README.md'),
                         (Path('NOTICE'),'NOTICE')]:
        shutil.copyfile(source, a.output / dest)
    # The checked-in card renders against repository evidence; the Hub bundle
    # carries those same figures and aggregate files beside the card.
    card = (a.output / 'README.md').read_text()
    card = card.replace('(../results/cuda-v0.1/figures/', '(figures/')
    card = card.replace('(../results/cuda-v0.1/', '(evaluation/')
    card = card.replace('(../NOTICE)', '(NOTICE)')
    (a.output / 'README.md').write_text(card)
    shutil.copytree(a.report / 'figures', a.output / 'figures')
    (a.output / 'evaluation').mkdir()
    shutil.copyfile(validation_path, a.output / 'evaluation/validation.json')
    for name in ('summary.json','REPORT.md','accuracy.csv'):
        shutil.copyfile(a.report / name, a.output / 'evaluation' / name)
    # The report's figure links are relative to its evaluation/ directory.
    report_text = (a.output / 'evaluation/REPORT.md').read_text().replace('(figures/', '(../figures/')
    (a.output / 'evaluation/REPORT.md').write_text(report_text)
    licenses = {
        'LICENSE': 'https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt',
        'licenses/Bonsai-LICENSE': 'https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/6ed5e12bf84b7a63069882c91dd9e9218647d17b/LICENSE',
        'licenses/Bonsai-NOTICE.txt': 'https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/6ed5e12bf84b7a63069882c91dd9e9218647d17b/NOTICE.txt',
        'licenses/Prism-MIT': 'https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/d8f26eec76da6d09bb708bcba51ef64b8cd868a3/LICENSE',
        'licenses/OpenJev-helper-Apache-2.0': 'https://huggingface.co/openjev/openjev/resolve/5ec9e5fd2f80a6fff386779b1e5ac7e389971889/LICENSE-APACHE-2.0',
        'licenses/TypeSafe-MIT': 'https://raw.githubusercontent.com/typesafe-ai/system-one-adapter-python/adffc2eab300a4fa3c0e92252d4ffd6ceaa53700/LICENSE'}
    license_hashes = {
        'LICENSE': '28a9529c7d0bb4dc51f4bf5c116a3d16ef247a052f7591466768ddf563fd1cf5',
        'licenses/Bonsai-LICENSE': '69849221bfb90053de2134ef5e6d540287b4b98062326492f1f96f5da685524b',
        'licenses/Bonsai-NOTICE.txt': 'de0e0c48fb6f691a31e74f338e3ccf93f9ecdfe2866ab769c4bf8b79a7636a30',
        'licenses/Prism-MIT': '94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d',
        'licenses/OpenJev-helper-Apache-2.0': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
        'licenses/TypeSafe-MIT': '835f233f1d6ed84a9b9a351aba0689b47644a4137d6316911fc7957bde523b02'}
    for name, url in licenses.items():
        path = a.output / name; path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'Shingi-release/0.1 (+https://github.com/kortexa-ai/shingi)'})
        data = urllib.request.urlopen(request, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != license_hashes[name]:
            raise ValueError('upstream license changed: ' + name)
        path.write_bytes(data)
    manifest = {'format_version': 1, 'model': RELEASE_MODEL_ID, 'adapter_license': 'cc-by-sa-4.0',
                'source_repository': 'https://github.com/kortexa-ai/shingi',
                'source_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'base': {'repository': 'prism-ml/Ternary-Bonsai-2-27B-gguf', 'revision': '6ed5e12bf84b7a63069882c91dd9e9218647d17b',
                         'filename': 'Ternary-Bonsai-2-27B-PQ2_0.gguf', 'sha256': BASE_SHA256, 'license': 'apache-2.0', 'included': False},
                'license_sources': licenses,
                'files': {str(p.relative_to(a.output)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                          for p in sorted(a.output.rglob('*')) if p.is_file()}}
    save(a.output / 'manifest.json', manifest)
    if sha256(a.output / 'adapter.gguf') != ADAPTER_SHA256:
        raise ValueError('adapter copy checksum differs')
    print(json.dumps({'bundle': str(a.output), 'files': len(manifest['files']) + 1,
                      'manifest_sha256': sha256(a.output / 'manifest.json')}, indent=2))


def verify_bundle(directory):
    manifest = read(directory / 'manifest.json')
    paths = list(directory.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('bundle contains an external path')
    actual = {str(p.relative_to(directory)) for p in paths if p.is_file()}
    if actual != set(manifest['files']) | {'manifest.json'}:
        raise ValueError('bundle file inventory differs')
    for name, expected in manifest['files'].items():
        path = directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('bundle contains an external path')
        if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError('bundle checksum differs: ' + name)
    if sha256(directory / 'adapter.gguf') != ADAPTER_SHA256:
        raise ValueError('bundle adapter differs')
    print(json.dumps({'verified': len(actual), 'manifest_sha256': sha256(directory / 'manifest.json')}, indent=2))


def main():
    p = argparse.ArgumentParser(); commands = p.add_subparsers(dest='command', required=True)
    r = commands.add_parser('report')
    for card in ('6000','4090'):
        for kind in ('evaluation','speed','api'):
            r.add_argument('--'+kind+'-'+card, type=Path, required=True)
        for kind in ('evaluation', 'speed'):
            r.add_argument('--previous-'+kind+'-'+card, type=Path)
    r.add_argument('--previous-protocol', type=Path)
    r.add_argument('--data', type=Path, required=True); r.add_argument('--protocol', type=Path, default=Path('release/manifest.json'))
    r.add_argument('--output', type=Path, required=True)
    b = commands.add_parser('bundle'); b.add_argument('--adapter', type=Path, required=True)
    b.add_argument('--report', type=Path, required=True); b.add_argument('--output', type=Path, required=True)
    v = commands.add_parser('verify'); v.add_argument('directory', type=Path)
    args = p.parse_args()
    if args.command == 'verify':
        verify_bundle(args.directory)
    else:
        (report_release if args.command=='report' else assemble)(args)


if __name__ == '__main__':
    main()
