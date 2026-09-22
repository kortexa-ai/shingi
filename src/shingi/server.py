import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
import uvicorn

from .backend import NativeReadout
from .decision import Calibration, DecisionEngine, MODEL_ID
from .schema import Request
from .release import artifact_identity, load_calibration, require_release_environment, READOUT_VERSION


def create_app(engine=None, *, executable=None, model=None, context_tokens=16384,
               adapter=None, identity=None, calibration=Calibration(), calibration_sha256=None):
    if identity is None:
        identity = (artifact_identity(model, adapter) if engine is None
                    else {"model": engine.model_id, "trained": False})
    @asynccontextmanager
    async def lifespan(app):
        backend = None
        try:
            if engine is None:
                backend = NativeReadout(executable, model, context_tokens, adapter=adapter)
                app.state.engine = DecisionEngine(backend, calibration, model_id=identity["model"],
                    canonical_choices=identity.get("readout_version") == READOUT_VERSION)
            yield
        finally:
            if backend is not None:
                backend.close()

    app = FastAPI(title="Shingi", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    def health():
        current = app.state.engine
        process = getattr(getattr(current, "backend", None), "process", None)
        if current is None or (process is not None and process.poll() is not None):
            raise HTTPException(503, "native model unavailable")
        return {"status": "ok"}

    @app.get("/v1/version")
    def version():
        current = app.state.engine
        return {**identity, "calibration": asdict(current.calibration),
                "calibration_sha256": calibration_sha256,
                "context_tokens": context_tokens, "single_pass_options": 52,
                "choice_limit": 255, "image_input": False}

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": identity["model"], "description": "Local native ternary Bonsai decision model; text only. See /v1/version for adapter and calibration identity.",
                            "release_date": "2026-09-22"}]}

    @app.post("/v1/systemone")
    def system_one(body: Request):
        if body.model not in (identity["model"], "shingi", "shingi-latest", "jev-latest", "openjev"):
            raise HTTPException(422, "unknown model alias")
        try:
            response, _ = app.state.engine.evaluate(body.model_dump(exclude_none=True))
            return response
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, TimeoutError, BrokenPipeError) as exc:
            raise HTTPException(503, "native readout unavailable") from exc

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--executable", type=Path, default=Path("artifacts/bin/readout"))
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--adapter", type=Path, help="native GGUF LoRA adapter; omit for the unchanged base")
    args = parser.parse_args()
    require_release_environment()
    identity = artifact_identity(args.model, args.adapter)
    calibration, calibration_sha256 = load_calibration(args.calibration, identity)
    # Local serving only. Network deployment needs its own auth and resource policy.
    uvicorn.run(create_app(executable=args.executable, model=args.model,
                           adapter=args.adapter, identity=identity,
                           context_tokens=args.context_tokens, calibration=calibration,
                           calibration_sha256=calibration_sha256), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
