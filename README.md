# Spoken Language Identification

A FastAPI server that identifies the spoken language of an audio file using the
SpeechBrain ECAPA-TDNN model
([`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa),
107 languages), plus a small client to call it.

## Contents

| File | Purpose |
|---|---|
| `server.py` | FastAPI server (`GET /health`, `POST /v1/audio/language_identification`) |
| `model.py` | SpeechBrain model wrapper |
| `schemas.py` | Request / response schemas |
| `client.py` | Command-line client that calls the server |
| `audio/` | Sample audio (`test1.wav`, `test_1min.wav`) |
| `requirements.txt` | Python dependencies |
| `setup_venv.sh` | Creates `venv/` and installs the requirements |

## Requirements

- Python 3.10 – 3.12
- Internet access on first start (the model, ~80 MB, is downloaded from HuggingFace
  into `pretrained_models/ecapa-tdnn/` and reused afterwards)
- A GPU is optional; the server runs on CPU by default if no GPU is found

## 1. Set up the virtual environment

```bash
./setup_venv.sh            # or manually:
                           #   python3 -m venv venv
                           #   source venv/bin/activate
                           #   pip install -r requirements.txt
source venv/bin/activate
```

## 2. Download the ECAPA-TDNN model

The server loads the model from `pretrained_models/ecapa-tdnn/`. If that folder is
missing, the server downloads it automatically on first start, but you can also
download it yourself beforehand (e.g. for offline machines).

Download the whole model repo from HuggingFace:
[`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/main)

Option A: HuggingFace CLI (run inside the venv)

```bash
pip install huggingface_hub==0.36.0   # already pinned in requirements.txt; do NOT use -U
huggingface-cli download speechbrain/lang-id-voxlingua107-ecapa \
  --local-dir pretrained_models/ecapa-tdnn
```

Option B: git (needs [git-lfs](https://git-lfs.com))

```bash
git lfs install
git clone https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa \
  pretrained_models/ecapa-tdnn
```

Option C: manual download. Download every file from the
[Files tab](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa/tree/main),
put them in a folder named `ecapa-tdnn`, and place that whole folder inside
`pretrained_models/`.

The result should look like this:

```
pretrained_models/
└── ecapa-tdnn/
    ├── classifier.ckpt
    ├── embedding_model.ckpt
    ├── hyperparams.yaml
    ├── label_encoder.txt
    └── ...
```

The folder name must match `MODEL_NAME` (default `ecapa-tdnn`). `pretrained_models/`
is git-ignored, so the model is not committed to the repo.

## 3. Start the server

```bash
python3 server.py
```

The first start downloads the model, so wait until you see
`Uvicorn running on http://0.0.0.0:8001`. Interactive API docs are at
http://localhost:8001/docs.

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SERVICE_PORT` | `8001` | Port to listen on |
| `COMPUTE_DEVICE` | `auto` | `cuda`, `cpu` or `auto` |
| `MAX_AUDIO_LENGTH` | `15` | Audio longer than this (seconds) is clipped to a random segment of this length |
| `LID_CONCURRENCY_LIMIT` | `4` | Max concurrent inferences |
| `MODEL_DIR` / `MODEL_NAME` | `pretrained_models` / `ecapa-tdnn` | Where the model is stored |

Example: `COMPUTE_DEVICE=cpu SERVICE_PORT=9000 python server.py`

## 4. Call the server

In a second terminal (with the venv activated):

```bash
# top prediction
python3 client.py audio/test1.wav

# confidences for all languages (top 5 printed)
python3 client.py audio/test1.wav --verbose

# a different server address
python3 client.py audio/test1.wav --url http://localhost:9000
```

Expected output (shape; the language/confidence depend on the audio):

```
Predicted language: en (confidence 0.9xxx)
```

### Using curl instead

```bash
curl http://localhost:8001/health

curl -X POST http://localhost:8001/v1/audio/language_identification \
  -F "file=@audio/test1.wav" \
  -F "response_format=verbose_json"
```

## API

### `GET /health`
Returns `{"status": "HEALTHY"}`.

### `POST /v1/audio/language_identification`
Multipart form fields:

- `file` (required): audio file (WAV, FLAC, OGG, ... anything `soundfile` can read)
- `response_format` (optional): `json` (default) or `verbose_json`

Response:

```json
{
  "id": "lid-<uuid>",
  "task": "language_identification",
  "created_at": 1700000000,
  "model": "ecapa-tdnn",
  "output": {
    "language": "en",
    "confidence": 0.98,
    "all_confidences": [["en", 0.98], ["..."]]
  }
}
```

`all_confidences` is only present with `verbose_json`, sorted by confidence
(descending). Invalid audio or an unknown `response_format` returns HTTP 400.

## Troubleshooting

- **Model download fails**: check internet access, or manually place
  `classifier.ckpt`, `embedding_model.ckpt`, `hyperparams.yaml` and `label_encoder.txt`
  from the HuggingFace repo into `pretrained_models/ecapa-tdnn/`.
- **Port in use**: set `SERVICE_PORT` to another port and pass `--url` to the client.
