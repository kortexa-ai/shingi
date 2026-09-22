import argparse
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
import uvicorn

from .backend import NativeReadout
from .decision import DecisionEngine, MODEL_ID
from .schema import Request


def create_app(engine=None, *, executable=None, model=None, context_tokens=16384):
    @asynccontextmanager
    async def lifespan(app):
        backend = None
        try:
            if engine is None:
                backend = NativeReadout(executable, model, context_tokens)
                app.state.engine = DecisionEngine(backend)
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
        return {"model": MODEL_ID, "calibration": asdict(current.calibration),
                "trained": False, "context_tokens": context_tokens, "single_pass_options": 52,
                "choice_limit": 255, "image_input": False}

    @app.get("/v1/models")
    def models():
        return {"models": [{"name": MODEL_ID, "description": "Experimental native ternary Bonsai readout; text only, untrained and uncalibrated.",
                            "release_date": "2026-09-22"}]}

    @app.post("/v1/systemone")
    def system_one(body: Request):
        if body.model not in (MODEL_ID, "shingi", "shingi-latest", "jev-latest", "openjev"):
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
    args = parser.parse_args()
    # Initial investigation is local only. A public deployment needs its own auth and resource policy.
    uvicorn.run(create_app(executable=args.executable, model=args.model,
                           context_tokens=args.context_tokens), host="127.0.0.1", port=args.port)
