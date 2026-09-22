import json

import pytest
from shingi.decision import Calibration, DecisionEngine
from shingi.release import artifact_identity, load_calibration, sha256, require_release_environment
from shingi.server import create_app
from fastapi.testclient import TestClient


def test_artifact_and_calibration_bind_both_files(tmp_path):
    model, adapter, config = [tmp_path / n for n in ('base', 'adapter', 'calibration.json')]
    model.write_bytes(b'base'); adapter.write_bytes(b'adapter')
    identity = artifact_identity(model, adapter)
    assert identity['trained'] is True
    assert identity['model'] == 'shingi-custom'
    config.write_text(json.dumps({'parameters': {'temperature': 1.25}, 'provenance': identity}))
    fitted, digest = load_calibration(config, identity)
    assert fitted == Calibration(temperature=1.25)
    assert digest == sha256(config)
    with pytest.raises(ValueError, match='adapter_sha256'):
        load_calibration(config, artifact_identity(model))
    adapter.write_bytes(b'changed')
    with pytest.raises(ValueError, match='adapter_sha256'):
        load_calibration(config, artifact_identity(model, adapter))
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='model_sha256'):
        load_calibration(config, artifact_identity(model, adapter))


def test_model_identity_is_consistent_across_api():
    class Backend:
        def infer(self, prompt, labels):
            return {'logits': [1.0, 0.0], 'input_tokens': 12}
    identity = {'model': 'shingi-test-adapter', 'trained': True, 'adapter_sha256': 'test'}
    engine = DecisionEngine(Backend(), model_id=identity['model'])
    with TestClient(create_app(engine, identity=identity)) as client:
        assert client.get('/v1/version').json()['trained'] is True
        assert client.get('/v1/models').json()['models'][0]['name'] == identity['model']
        body = {'model': identity['model'], 'state': 'x', 'questions': {'q': {'type': 'noul', 'instructions': 'yes?'}}}
        result = client.post('/v1/systemone', json=body)
        assert result.status_code == 200
        assert result.json()['model'] == identity['model']
        body['model'] = 'shingi-bonsai-2-27b-baseline'
        assert client.post('/v1/systemone', json=body).status_code == 422


def test_rejects_old_readout_calibration(tmp_path):
    identity = {'model_sha256': 'base', 'adapter_sha256': 'adapter'}
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({'parameters': {}, 'provenance': {**identity, 'readout_version': 'bonsai-letter-v1'}}))
    with pytest.raises(ValueError, match='readout version'):
        load_calibration(path, identity)


def test_release_rejects_experimental_runtime_flags(monkeypatch):
    monkeypatch.setenv('SHINGI_KV_F16', '1')
    with pytest.raises(RuntimeError, match='Q8 CUDA'):
        require_release_environment()


def test_report_rejects_duplicate_or_changed_predictions(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from package_release import load_predictions
    path = tmp_path / 'predictions.jsonl'
    row = {'id': 'one', 'input_sha256': 'original', 'error': None}
    path.write_text(json.dumps(row) + '\n')
    assert load_predictions(path, [row]) == {'one': row}
    path.write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(ValueError, match='inventory'):
        load_predictions(path, [row])
    path.write_text(json.dumps({**row, 'input_sha256': 'changed'}) + '\n')
    with pytest.raises(ValueError, match='mismatched'):
        load_predictions(path, [row])


def test_bundle_verification_rejects_extra_files_and_corruption(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    import package_release
    adapter = tmp_path / 'adapter.gguf'; adapter.write_bytes(b'fixture')
    monkeypatch.setattr(package_release, 'ADAPTER_SHA256', sha256(adapter))
    manifest = {'files': {'adapter.gguf': {'bytes': adapter.stat().st_size, 'sha256': sha256(adapter)}}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    package_release.verify_bundle(tmp_path)
    extra = tmp_path / 'raw-data.jsonl'; extra.write_text('private input')
    with pytest.raises(ValueError, match='inventory'):
        package_release.verify_bundle(tmp_path)
    extra.unlink(); adapter.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        package_release.verify_bundle(tmp_path)
