import json

import pytest
from shingi.decision import Calibration, DecisionEngine
from shingi.release import artifact_identity, load_calibration, sha256
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
