"""Derive public evidence/charts and assemble an explicit, checksum-verified HF bundle."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

from evaluate_release import ordered_rows, parity
from prepare_benchmark import read_jsonl
from shingi.decision import Calibration
from shingi.release import sha256, BASE_SHA256, ADAPTER_SHA256, RELEASE_MODEL_ID
from shingi.metrics import report as quality_report
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


def validate_receipt(receipt, card, protocol_hash, data_hash):
    if receipt['protocol_sha256'] != protocol_hash:
        raise ValueError('run protocol differs')
    if receipt.get('data_manifest_sha256', receipt.get('dataset_manifest_sha256')) != data_hash:
        raise ValueError('run data manifest differs')
    identity = receipt['identity']
    if identity['model_sha256'] != BASE_SHA256 or identity['adapter_sha256'] != ADAPTER_SHA256:
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
    fig.supxlabel('Same frozen calibration and fresh test. Reliability bins have unequal sample counts.', fontsize=10)
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
    fig.supxlabel('300 fixed records × three repeats; ten warmups. Memory sampled every 100 ms; these are not continuous peaks.', fontsize=10)
    export(fig, 'hardware')


def report_release(a):
    protocol = read(a.protocol); manifest = read(a.data / 'manifest.json')
    if sha256(a.data / 'test.jsonl') != manifest['test_sha256']:
        raise ValueError('test split changed')
    records = ordered_rows(read_jsonl(a.data / 'test.jsonl'), protocol['choice_order']=='canonical')
    result = {'model': RELEASE_MODEL_ID, 'protocol': protocol, 'data': manifest,
              'quality': {}, 'speed': {}, 'api': {}, 'cross_device': {}, 'paired': {}, 'inputs': {}}
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
            if recomputed != result['quality'][card]['models'][name]['calibrated'] or recomputed['overall']['valid'] != len(records):
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
            'held_out_sources': paired([r for r in records if r['source'] in HELD], predictions[card], calibrations)}
    if len({result[kind][card]['receipt']['source_revision'] for kind in ('quality','speed') for card in ('6000','4090')}) != 1:
        raise ValueError('measurement source revisions differ')
    for name in ('base','adapter'):
        result['cross_device'][name] = parity(records, predictions['6000'][name], predictions['4090'][name])
        check = result['cross_device'][name]
        if check['argmax_agreement'] < .99 or check['mean_tvd'] > .01:
            raise ValueError('cross-device parity gate failed')
    a.output.mkdir(parents=True, exist_ok=False)
    # Keep the full metrics; only raw prompts/predictions stay outside the public report.
    save(a.output / 'summary.json', result)
    charts(result, a.output)
    with (a.output / 'accuracy.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source','held_out_from_adaptation','gpu','model','n','correct','accuracy','wilson_low','wilson_high','nll'])
        for card in ('6000','4090'):
            for name in ('base','adapter'):
                for source, m in result['quality'][card]['models'][name]['calibrated']['by_source'].items():
                    writer.writerow([source,source in HELD,card,name,m['n'],m['correct'],m['accuracy_failures_incorrect'],*m['accuracy_wilson_95'],m['nll']['mean']])
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
              '- Wall times include all forward passes, tokenization and memory checks; loading is separate. OS filesystem cache is uncontrolled. Device memory is sampled every 100 ms; a brief peak can be missed. Other resident processes are recorded in private operator evidence.',
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
    a.output.mkdir(parents=True, exist_ok=False)
    for source, dest in [(a.adapter,'adapter.gguf'), (Path('release/calibration.json'),'calibration.json'),
                         (Path('release/manifest.json'),'protocol.json'), (Path('release/README.md'),'README.md'),
                         (Path('NOTICE'),'NOTICE')]:
        shutil.copyfile(source, a.output / dest)
    shutil.copytree(a.report / 'figures', a.output / 'figures')
    (a.output / 'evaluation').mkdir()
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
    for name, url in licenses.items():
        path = a.output / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
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
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
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
