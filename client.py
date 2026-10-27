"""Client for the spoken language identification server.

Example:
    python client.py audio/test1.wav
    python client.py audio/test1.wav --verbose --url http://localhost:8001
"""

import argparse
import json
import sys

import httpx

ROUTE = "/v1/audio/language_identification"


def identify(
    audio_path: str, base_url: str = "http://localhost:8001", verbose: bool = False
) -> dict:
    """Sends an audio file to the server and returns the JSON response."""
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    with httpx.Client(timeout=120) as client:
        response = client.post(
            base_url.rstrip("/") + ROUTE,
            files={"file": (audio_path, audio_bytes)},
            data={"response_format": "verbose_json" if verbose else "json"},
        )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Spoken language identification client")
    parser.add_argument("audio", help="path to an audio file (wav, flac, ogg, ...)")
    parser.add_argument("--url", default="http://localhost:8001", help="server URL")
    parser.add_argument(
        "--verbose", action="store_true", help="return confidences for all languages"
    )
    parser.add_argument("--top", type=int, default=5, help="rows shown with --verbose")
    args = parser.parse_args()

    try:
        result = identify(args.audio, args.url, args.verbose)
    except httpx.HTTPError as exc:
        sys.exit(f"Request failed: {exc}")

    output = result["output"]
    print(f"Predicted language: {output['language']} (confidence {output['confidence']:.4f})")
    if args.verbose:
        print(f"Top {args.top}:")
        for lang, conf in output["all_confidences"][: args.top]:
            print(f"  {lang:<6}{conf:.4f}")
    print("\nFull response:")
    print(json.dumps(result, indent=2)[:2000])


if __name__ == "__main__":
    main()
