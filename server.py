"""FastAPI server for spoken language identification (ECAPA-TDNN, SpeechBrain)."""

import asyncio
import io
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated
from uuid import uuid4

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST

from model import SpeechBrainLID
from schemas import ResponseOutput, ServiceRequest, SuccessResponse

# Configuration (override via environment variables)
SERVICE_HOST = os.environ.get("SERVICE_HOST", "0.0.0.0")
SERVICE_PORT = int(os.environ.get("SERVICE_PORT", "8001"))
MODEL_DIR = os.environ.get("MODEL_DIR", "pretrained_models")
MODEL_NAME = os.environ.get("MODEL_NAME", "ecapa-tdnn")
COMPUTE_DEVICE = os.environ.get("COMPUTE_DEVICE", "auto")  # "cuda", "cpu" or "auto"
MAX_AUDIO_LENGTH = float(os.environ.get("MAX_AUDIO_LENGTH", "15"))  # seconds
CONCURRENCY_LIMIT = int(os.environ.get("LID_CONCURRENCY_LIMIT", "4"))
TASK = "language_identification"

logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s][%(asctime)s]: %(message)s"
)
logger = logging.getLogger("lid")

app = FastAPI(
    title="Spoken Language Identification",
    description="Identifies the spoken language of an audio file using ECAPA-TDNN.",
    version="1.0.0",
)

model = SpeechBrainLID(model_dir=MODEL_DIR, model_name=MODEL_NAME, device=COMPUTE_DEVICE)
executor = ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT)


@app.get("/health", status_code=HTTP_200_OK)
async def read_health() -> dict[str, str]:
    """Checks if the service is up."""
    return {"status": "HEALTHY"}


def encode_output(
    lid_model: SpeechBrainLID,
    out_prob: torch.Tensor,
    best_pred_idx: int,
    best_label: str,
    response_format: str,
) -> ResponseOutput:
    """Formats the model output into a ResponseOutput."""
    confs = lid_model.get_confidence_scores(out_prob)
    mapped_conf = None
    if response_format == "verbose_json":
        mapped_conf = sorted(
            lid_model.map_confidence_scores(confs), key=lambda x: x[1], reverse=True
        )
    return ResponseOutput(
        language=lid_model.parse_language_label(best_label),
        confidence=confs[best_pred_idx],
        all_confidences=mapped_conf,
    )


@app.post("/v1/audio/language_identification", response_model=SuccessResponse)
async def language_identification(
    request: Annotated[ServiceRequest, Form()],
) -> SuccessResponse:
    """
    Accepts an audio file and a response_format ("json" or "verbose_json") and
    returns the predicted language with its confidence.
    """
    request_id = "lid-" + str(uuid4())
    logger.info("New LID request | id: %s", request_id)

    if request.response_format not in ("json", "verbose_json"):
        raise HTTPException(
            HTTP_400_BAD_REQUEST, "response_format must be 'json' or 'verbose_json'"
        )

    audio_bytes = await request.file.read()
    try:
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as exc:  # soundfile raises several error types
        raise HTTPException(
            HTTP_400_BAD_REQUEST, f"Could not decode audio file: {exc}"
        ) from exc
    audio_data = audio_data.astype(np.float32)
    if audio_data.ndim > 1:  # stereo -> mono
        audio_data = audio_data.mean(axis=1)

    # Long audio: randomly pick a MAX_AUDIO_LENGTH-second segment to prevent OOM
    max_samples = int(MAX_AUDIO_LENGTH * sr)
    if len(audio_data) > max_samples:
        logger.info(
            "Audio is %.2fs, clipping to %.2fs", len(audio_data) / sr, MAX_AUDIO_LENGTH
        )
        start = random.randint(0, len(audio_data) - max_samples)
        audio_data = audio_data[start : start + max_samples]

    # Offload blocking inference to the thread pool
    loop = asyncio.get_running_loop()
    out_probs, best_pred_idx, best_labels = await loop.run_in_executor(
        executor, lambda: model(audio_data, sr)
    )
    output = encode_output(
        lid_model=model,
        out_prob=out_probs[0],
        best_pred_idx=best_pred_idx.item(),
        best_label=best_labels[0],
        response_format=request.response_format,
    )

    response = SuccessResponse(
        id=request_id,
        task=TASK,
        created_at=int(time.time()),
        model=model.get_model_name(),
        output=output.model_dump(exclude_none=True),
    )
    logger.info("Response: %s", response)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, log_level="info")
