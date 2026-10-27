"""Pydantic schemas for the language identification API."""

from typing import Any, Dict, List, Optional, Tuple

from fastapi import File, UploadFile
from pydantic import BaseModel


class ServiceRequest(BaseModel):
    """
    Form request for the LID endpoint.

    file: audio file to identify.
    response_format: "json" (top prediction only) or "verbose_json"
        (confidence for every language).
    """

    file: UploadFile = File(...)
    response_format: str = "json"


class ResponseOutput(BaseModel):
    """Prediction result."""

    language: str
    confidence: float
    all_confidences: Optional[List[Tuple[str, float]]] = None


class SuccessResponse(BaseModel):
    """Envelope returned by a successful request."""

    id: str
    task: str
    created_at: int
    model: str
    output: Dict[str, Any]
