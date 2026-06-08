"""
Higgs Audio v3 TTS API client (optional module).
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import requests


class HiggsV3TTSClient:
    def __init__(self, api_key=None, base_url="https://api.boson.ai"):
        self.api_key = api_key or os.environ.get("BOSON_API_KEY", "")
        self.base_url = base_url.rstrip("/")

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def generate_speech(
        self,
        input_text: str,
        voice: str = "default",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        response_format: str = "mp3",
        stream: bool = False,
    ) -> bytes:
        payload = {
            "model": "higgs-audio-v3-tts",
            "input": input_text,
            "response_format": response_format,
            "stream": stream,
        }
        if ref_audio:
            payload["ref_audio"] = ref_audio
            if ref_text:
                payload["ref_text"] = ref_text
        else:
            payload["voice"] = voice

        resp = requests.post(
            f"{self.base_url}/v1/audio/speech",
            headers=self._headers(),
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.content

    def create_custom_voice(
        self, ref_audio: str, ref_text: str, title: str = "My Voice"
    ) -> str:
        payload = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "title": title,
        }
        resp = requests.post(
            f"{self.base_url}/v1/voices",
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["voice_id"]

    @staticmethod
    def encode_audio_file(filepath: str) -> str:
        with open(filepath, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return data

    def batch_generate(
        self,
        jsonl_path: str,
        voice: str = "default",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        output_dir: str = "batch_audio",
    ) -> List[dict]:
        os.makedirs(output_dir, exist_ok=True)
        results = []

        with open(jsonl_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("text", "")
                if not text:
                    continue

                fmt = "wav"
                out_path = os.path.join(output_dir, f"speech_{idx:06d}.{fmt}")

                try:
                    audio_bytes = self.generate_speech(
                        input_text=text,
                        voice=voice,
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        response_format=fmt,
                    )
                    with open(out_path, "wb") as wf:
                        wf.write(audio_bytes)
                    results.append({"idx": idx, "text": text, "audio_path": out_path, "status": "ok"})
                    print(f"[{idx}] Generated: {out_path}")
                except Exception as e:
                    results.append({"idx": idx, "text": text, "status": "error", "error": str(e)})
                    print(f"[{idx}] Error: {e}")

                time.sleep(0.1)

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    gen_parser = sub.add_parser("generate")
    gen_parser.add_argument("--input", required=True)
    gen_parser.add_argument("--voice", default="default")
    gen_parser.add_argument("--ref-audio")
    gen_parser.add_argument("--ref-text")
    gen_parser.add_argument("--output", default="output.wav")
    gen_parser.add_argument("--format", default="wav", choices=["mp3", "wav", "opus", "flac", "aac"])

    batch_parser = sub.add_parser("batch")
    batch_parser.add_argument("--input", required=True, help="JSONL file")
    batch_parser.add_argument("--voice", default="default")
    batch_parser.add_argument("--ref-audio")
    batch_parser.add_argument("--ref-text")
    batch_parser.add_argument("--output-dir", default="batch_audio")

    voice_parser = sub.add_parser("create-voice")
    voice_parser.add_argument("--ref-audio", required=True)
    voice_parser.add_argument("--ref-text", required=True)
    voice_parser.add_argument("--title", default="My Voice")

    args = parser.parse_args()

    client = HiggsV3TTSClient()

    if args.cmd == "generate":
        audio = client.generate_speech(
            input_text=args.input,
            voice=args.voice,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            response_format=args.format,
        )
        with open(args.output, "wb") as f:
            f.write(audio)
        print(f"Saved to {args.output}")

    elif args.cmd == "batch":
        results = client.batch_generate(
            jsonl_path=args.input,
            voice=args.voice,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            output_dir=args.output_dir,
        )
        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"Done: {ok}/{len(results)}")

    elif args.cmd == "create-voice":
        voice_id = client.create_custom_voice(
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            title=args.title,
        )
        print(f"Created voice: {voice_id}")


if __name__ == "__main__":
    main()
