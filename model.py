"""SpeechBrain ECAPA-TDNN wrapper for spoken language identification."""

import os
from typing import List, Tuple

import numpy as np
import torch
from speechbrain.inference.classifiers import EncoderClassifier

HF_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"


class SpeechBrainLID:
    """Loads the ECAPA-TDNN VoxLingua107 model and runs language identification."""

    def __init__(self, model_dir: str, model_name: str, device: str) -> None:
        """
        Args:
            model_dir: Directory that holds (or will hold) the model folder.
            model_name: Name of the model folder inside model_dir.
            device: "cuda" or "cpu". Anything else auto-selects.
        """
        if device not in ("cuda", "cpu"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.model_name = model_name
        self.device = torch.device(device)

        model_path = os.path.join(model_dir, model_name)
        # Downloads from HuggingFace Hub on first run, reuses the local copy afterwards.
        self.model = EncoderClassifier.from_hparams(
            source=HF_SOURCE if not os.path.isdir(model_path) else model_path,
            savedir=model_path,
            run_opts={"device": device},
        )
        self.ind2lab = self.model.hparams.label_encoder.ind2lab

    def get_model_name(self) -> str:
        """Returns the model name."""
        return self.model_name

    @staticmethod
    def parse_language_label(label: str) -> str:
        """Parses a SpeechBrain label, e.g. "en: English" -> "en"."""
        return label.split(": ")[0]

    @staticmethod
    def get_confidence_scores(logits: torch.Tensor) -> List[float]:
        """Softmax over the model output, returned as a list of floats."""
        return (torch.exp(logits) / torch.exp(logits).sum()).tolist()

    def map_confidence_scores(
        self, confidence_scores: List[float]
    ) -> List[Tuple[str, float]]:
        """Pairs each confidence score with its language code."""
        return [
            (self.parse_language_label(label), confidence_scores[ind])
            for ind, label in self.ind2lab.items()
        ]

    def __call__(
        self, wav: np.ndarray, sample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Runs inference on a mono waveform.

        Returns:
            (output scores (B, N), best prediction index (B,), best labels (B,))
        """
        torch_wav = torch.from_numpy(np.copy(wav))
        # (T,) -> (T, 1), then resample to 16kHz mono
        torch_wav = torch_wav.unsqueeze(-1)
        torch_wav = self.model.audio_normalizer(torch_wav, sample_rate)

        out_probs, _, best_pred_idx, best_labels = self.model.classify_batch(torch_wav)
        return out_probs, best_pred_idx, best_labels
