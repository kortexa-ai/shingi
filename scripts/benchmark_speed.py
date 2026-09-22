"""Sequential CUDA latency and sampled load/inference memory; no service control."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
import threading
import time

from prepare_benchmark import digest, read_jsonl
from shingi.backend import NativeReadout
from shingi.decision import Calibration, DecisionEngine
from shingi.gpu import gpu_snapshot, selected_gpu
from shingi.metrics import percentile
from shingi.probes import probe_at_tokens
from shingi.release import artifact_identity, sha256, require_release_environment


class MemorySampler:
    """One persistent nvidia-smi process avoids spawning a process per sample."""
    def __init__(self):
        self.samples = []
        self.phase = 'baseline'
        self.native_pid = None
        self.ready = threading.Event()
        self.started = time.perf_counter()
        self.process = subprocess.Popen(['nvidia-smi', '--id=' + selected_gpu(),
            '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits', '--loop-ms=100'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()
        if not self.ready.wait(10):
            self.close()
            raise RuntimeError('GPU memory sampler did not start')

    def read(self):
        for line in self.process.stdout:
            try:
                used, free = [int(x.strip()) for x in line.strip().split(',')]
            except ValueError:
                continue
            rss_kib = None
            if self.native_pid:
                try:
                    for row in Path(f'/proc/{self.native_pid}/status').read_text().splitlines():
                        if row.startswith('VmRSS:'):
                            rss_kib = int(row.split()[1]); break
                except (FileNotFoundError, ProcessLookupError):
                    pass
            self.samples.append({'seconds': time.perf_counter() - self.started, 'phase': self.phase,
                                 'used_mib': used, 'free_mib': free, 'native_rss_kib': rss_kib})
            self.ready.set()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        self.thread.join(timeout=5)

    def summary(self):
        if not self.samples:
            raise RuntimeError('no memory samples')
        baseline = self.samples[0]['used_mib']
        phases = {}
        for phase in ('load', 'warmup', 'mixed', 'synthetic'):
            rows = [r for r in self.samples if r['phase'] == phase]
            if rows:
                rss = [r['native_rss_kib'] for r in rows if r['native_rss_kib'] is not None]
                phases[phase] = {'samples': len(rows), 'max_gpu_used_mib': max(r['used_mib'] for r in rows),
                    'max_gpu_delta_mib': max(r['used_mib'] for r in rows) - baseline,
                    'minimum_free_mib': min(r['free_mib'] for r in rows),
                    'max_native_rss_kib': max(rss) if rss else None}
        return {'requested_sampling_interval_ms': 100, 'baseline_gpu_used_mib': baseline,
                'phases': phases, 'limitation': 'Sampled device allocation delta, not a continuous or process-isolated peak. OS filesystem cache is uncontrolled.'}


def stats(values):
    return {'n': len(values), 'median': percentile(values, .5), 'p95': percentile(values, .95)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--executable', type=Path, default=Path('artifacts/bin/readout'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require_release_environment()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('commit source before benchmarking')
    identity = artifact_identity(a.model, a.adapter)
    protocol = json.loads(a.protocol.read_text())
    for key in ('model_sha256', 'adapter_sha256'):
        if protocol[key] != identity[key]:
            raise ValueError('protocol weights differ')
    manifest = json.loads((a.data / 'manifest.json').read_text())
    if sha256(a.data / 'test.jsonl') != manifest['test_sha256']:
        raise ValueError('test records changed')
    pool = read_jsonl(a.data / 'test.jsonl')
    sources = sorted({r['source'] for r in pool})
    mixed = []
    for source in sources:
        mixed.extend(sorted((r for r in pool if r['source'] == source),
                            key=lambda r: digest('speed-v01|' + r['id']))[:20])
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = {'identity': identity, 'protocol_sha256': sha256(a.protocol),
               'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'executable_sha256': sha256(a.executable), 'dataset_manifest_sha256': sha256(a.data / 'manifest.json'),
               'gpu': gpu_snapshot(), 'started_at': datetime.now(timezone.utc).isoformat(),
               'driver': subprocess.check_output(['nvidia-smi', '--id=' + selected_gpu(), '--query-gpu=driver_version,power.limit', '--format=csv,noheader'], text=True).strip(),
               'context_tokens': 16384, 'kv': 'q8_0', 'batch': 1, 'mixed_ids': [r['id'] for r in mixed],
               'mixed_repeats': 3, 'synthetic_repeats': 10,
               'timing': 'Wall time around DecisionEngine.answer, including tokenization, safety queries and all candidate passes. Excludes model loading. No generated tokens.'}
    (a.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    sampler = MemorySampler()
    backend = None
    observations = []
    try:
        sampler.phase = 'load'
        start = time.perf_counter()
        backend = NativeReadout(a.executable, a.model, 16384, adapter=a.adapter)
        load_seconds = time.perf_counter() - start
        sampler.native_pid = backend.process.pid
        engine = DecisionEngine(backend, Calibration(**protocol['calibrations']['adapter']['parameters']),
                                canonical_choices=protocol['choice_order'] == 'canonical')
        sampler.phase = 'warmup'
        for row in mixed[:10]:
            engine.answer(row['state'], row['question'])
        with (a.output / 'observations.jsonl').open('x') as stream:
            def measure(row, group, repeat):
                start = time.perf_counter()
                answer, traces = engine.answer(row['state'], row['question'])
                elapsed = 1000 * (time.perf_counter() - start)
                item = {'group': group, 'id': row.get('id'), 'repeat': repeat,
                        'latency_ms': elapsed, 'native_prefill_ms': sum(t['prefill_ms'] for t in traces),
                        'forward_passes': len(traces), 'input_tokens': sum(t['input_tokens'] for t in traces),
                        'max_pass_tokens': max(t['input_tokens'] for t in traces),
                        'prompt_hashes': [t['prompt_sha256'] for t in traces],
                        'options': len(row['question'].get('criteria', {}))}
                stream.write(json.dumps(item) + '\n'); stream.flush(); observations.append(item)
            sampler.phase = 'mixed'
            for repeat in range(3):
                sequence = list(mixed); random.Random(20260922 + repeat).shuffle(sequence)
                for row in sequence:
                    measure(row, 'mixed', repeat)
                print('mixed repeat', repeat + 1, flush=True)
            sampler.phase = 'synthetic'
            for target in (512, 2048, 8192, 15000):
                row, count = probe_at_tokens(backend, target, .5, seed=target)
                engine.answer(row['state'], row['question'])
                for repeat in range(10):
                    measure(row, f'context-{target}', repeat)
                print('context', target, count, flush=True)
            for count in (4, 52, 53, 100, 255):
                row = {'state': {'selected_key': f'item_{count-1:03}'}, 'question': {
                    'type': 'choice', 'instructions': 'Select the exact key named in state.selected_key.',
                    'criteria': {f'item_{i:03}': None for i in range(count)}}}
                engine.answer(row['state'], row['question'])
                for repeat in range(10):
                    measure(row, f'options-{count}', repeat)
                print('options', count, flush=True)
    finally:
        if backend is not None:
            backend.close()
        sampler.close()
        (a.output / 'memory-samples.json').write_text(json.dumps(sampler.samples) + '\n')
    grouped = defaultdict(list)
    for row in observations:
        grouped[row['group']].append(row)
    result = {'receipt': receipt, 'model_initialization_seconds': load_seconds,
              'memory': sampler.summary(), 'groups': {}}
    for group, rows in grouped.items():
        result['groups'][group] = {'wall_ms': stats([r['latency_ms'] for r in rows]),
            'native_prefill_ms': stats([r['native_prefill_ms'] for r in rows]),
            'input_tokens': stats([r['input_tokens'] for r in rows]),
            'forward_passes': sorted({r['forward_passes'] for r in rows})}
    (a.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('model_initialization_seconds', 'memory')}, indent=2))


if __name__ == '__main__':
    main()
