"""Aawaz se Diagnosis web app.

Serves index.html and runs the same Sarvam pipeline as diagnose.py
(Speech-to-Text -> sarvam-105b -> Text-to-Speech) for each recording.
Standard library only; the API key is read from .env.

Usage:
  python3 app.py            then open http://localhost:8000
  python3 app.py --port 9000
"""

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sarvamai.core.api_error import ApiError

import diagnose as core

INDEX_HTML = Path(__file__).with_name("index.html")
MAX_BODY_BYTES = 15 * 1024 * 1024  # a 30-second recording is far smaller than this


def api_error_message(err):
    """Turn a Sarvam API error into a message someone can act on."""
    if err.status_code == 402:
        return "Your Sarvam account has no credits yet. Add credits at dashboard.sarvam.ai, then try again (no restart needed)."
    if err.status_code in (401, 403):
        return "Sarvam rejected the API key. Check SARVAM_API_KEY in .env, then restart the app."
    if err.status_code == 429:
        return "Too many requests to Sarvam right now. Wait a few seconds and try again."
    detail = err.body
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        detail = detail["error"].get("message", detail)
    return f"Sarvam returned an error ({err.status_code}): {detail}"


def run_pipeline(client, question, answer, audio):
    """Return (http_status, json_body) for one diagnosis request."""
    stage = "listening"
    try:
        transcript, language = core.transcribe(client, audio)
        if not transcript.strip():
            return 422, {"error": "We couldn't hear anything in the recording. Try again, a little louder.", "stage": stage}
        stage = "thinking"
        result = core.diagnose(client, question, answer, transcript, language)
    except ApiError as err:
        return 502, {"error": api_error_message(err), "stage": stage}
    except Exception as err:  # network errors, unreadable model output, ...
        return 500, {"error": f"Something went wrong: {err}", "stage": stage}

    reply = {"transcript": transcript, "language": language, **result, "audio": None, "audio_note": None}
    # The diagnosis is already useful, so a voice failure only adds a note
    if language not in core.TTS_LANGUAGES:
        reply["audio_note"] = "There is no voice for this language yet, so the explanation is text only."
    else:
        try:
            reply["audio"] = base64.b64encode(core.speak(client, result["explanation"], language)).decode()
        except ApiError as err:
            reply["audio_note"] = "Couldn't make the spoken explanation. " + api_error_message(err)
        except Exception as err:
            reply["audio_note"] = f"Couldn't make the spoken explanation: {err}"
    return 200, reply


class Handler(BaseHTTPRequestHandler):
    client = None  # set in main()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/diagnose":
            return self.send_json(404, {"error": "Not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY_BYTES:
            return self.send_json(413, {"error": "The recording is too large. Keep it under 30 seconds."})
        try:
            body = json.loads(self.rfile.read(length))
            question, answer = body["question"].strip(), body["answer"].strip()
            audio = (
                body.get("filename") or "recording.webm",
                base64.b64decode(body["audio"]),
                body.get("mime") or "application/octet-stream",
            )
        except (ValueError, KeyError, AttributeError):
            return self.send_json(400, {"error": "Send the question, the child's answer and a recording."})
        if not (question and answer and audio[1]):
            return self.send_json(400, {"error": "Send the question, the child's answer and a recording."})
        self.send_json(*run_pipeline(self.client, question, answer, audio))

    def send_json(self, status, data):
        self.send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def send(self, status, payload, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description="Run the Aawaz se Diagnosis web app.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    Handler.client = core.make_client()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Aawaz se Diagnosis is running at http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
