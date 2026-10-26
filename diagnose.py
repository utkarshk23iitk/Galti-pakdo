"""Aawaz se Diagnosis prototype.

A child explains their wrong maths answer out loud. This script:
  1. transcribes the recording (Speech-to-Text),
  2. finds the thinking mistake behind the answer and writes a simpler
     explanation plus practice questions (Chat, sarvam-105b),
  3. speaks the explanation back in the child's language (Text-to-Speech).

Usage:
  python diagnose.py child.m4a --question "1/2 + 1/3 = ?" --answer "2/5"
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from sarvamai import SarvamAI

# Common mistakes the model chooses from, so a diagnosis points at a known
# misconception instead of a vague guess. "other" covers anything new.
MISCONCEPTIONS = {
    "frac_add_across": "Adds the top numbers together and the bottom numbers together (1/2 + 1/3 = 2/5)",
    "frac_bigger_denominator": "Thinks a bigger bottom number means a bigger fraction (1/8 > 1/4)",
    "add_no_carry": "Forgets to carry when a column adds up to 10 or more",
    "sub_smaller_from_larger": "Subtracts the smaller digit from the larger one in every column, ignoring which number is on top (52 - 38 = 26)",
    "place_value_as_spoken": "Writes numbers the way they sound (one hundred five -> 1005)",
    "decimal_longer_is_bigger": "Thinks a decimal with more digits is bigger (0.25 > 0.5)",
    "left_to_right_only": "Works strictly left to right, doing addition before multiplication",
    "mult_as_addition": "Adds the numbers when the question asks to multiply them",
}

# Text-to-Speech languages, with a kid-style voice where one exists.
TTS_LANGUAGES = {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"}
KID_VOICES = {"hi-IN": "ritu_hi_kids", "en-IN": "nupur_en_kids"}

# Speech-to-Text content types by file extension (anything else is sent as
# application/octet-stream and Sarvam detects the codec itself).
AUDIO_TYPES = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/x-m4a", ".mp4": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".opus": "audio/opus", ".webm": "audio/webm",
    ".flac": "audio/flac", ".aiff": "audio/aiff", ".aif": "audio/aiff", ".amr": "audio/amr",
    ".wma": "audio/x-ms-wma",
}

SYSTEM_PROMPT = f"""You are a patient maths tutor for school children in India.
A child answered a maths question wrongly and then explained, out loud, how they got their answer.
Your job is to find the exact thinking mistake behind the wrong answer, not just say it is wrong.

Pick the mistake from this list when one fits, otherwise use "other":
{json.dumps(MISCONCEPTIONS, indent=2)}

Rules:
- Base the diagnosis on what the child actually said. Quote their words as evidence.
- Write "mistake" in simple English for the teacher.
- Write "explanation" and the practice questions in the same language and style the child spoke
  (mixing English maths words is fine), using the same script as the child's words (e.g. Devanagari
  for Hindi, not Roman letters) so a voice can read it correctly. The explanation is read aloud to the child, so keep it
  under 100 words, warm, and built around an everyday story (sharing pizza, rotis, sweets, cricket scores).
  Make sure the story is mathematically correct: for fractions, the wholes are the same size
  (identical pizzas); only the number of equal pieces differs.
- Give exactly 3 practice questions that target this specific mistake, easiest first."""

SCHEMA = {
    "type": "object",
    "properties": {
        "correct_answer": {"type": "string"},
        "mistake_id": {"type": "string", "enum": [*MISCONCEPTIONS, "other"]},
        "mistake": {"type": "string", "description": "The thinking mistake, in simple English for the teacher"},
        "evidence": {"type": "string", "description": "The child's own words that show this mistake"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "explanation": {"type": "string", "description": "Simpler re-explanation for the child, in their language"},
        "practice_questions": {
            "type": "array",
            "description": "Exactly 3 questions targeting this mistake",
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
                "required": ["question", "answer"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["correct_answer", "mistake_id", "mistake", "evidence", "confidence", "explanation", "practice_questions"],
    "additionalProperties": False,
}


def load_env():
    """Load .env into the environment (stdlib only, no python-dotenv)."""
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


def make_client():
    load_env()
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        sys.exit("SARVAM_API_KEY is not set. Add it to .env.")
    return SarvamAI(api_subscription_key=api_key)


def transcribe(client, filename, data):
    """Transcribe audio bytes. The content type comes from the file extension,
    because Python's own guess (e.g. audio/mp4a-latm for .m4a) is rejected by Sarvam."""
    content_type = AUDIO_TYPES.get(Path(filename).suffix.lower(), "application/octet-stream")
    result = client.speech_to_text.transcribe(
        file=(filename, data, content_type), model="saaras:v3", mode="codemix"
    )
    # Fall back to Hindi if the language could not be detected
    return result.transcript, result.language_code or "hi-IN"


def diagnose(client, question, answer, transcript, language):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\n\nReply with only a JSON object matching this JSON Schema:\n" + json.dumps(SCHEMA),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n"
            f"Child's answer: {answer}\n"
            f"Child's explanation (spoken, language {language}): {transcript}",
        },
    ]
    # Tested on real calls: strict json_schema output sometimes loops on blank lines
    # until max_tokens, and reasoning (even "low") can spend every token thinking.
    # A plain JSON object with reasoning off answered correctly in ~3 s, so use that
    # and retry once if a reply still comes back unusable.
    for _ in range(2):
        response = client.chat.completions(
            model="sarvam-105b",
            reasoning_effort=None,
            max_tokens=2000,
            response_format={"type": "json_object"},
            messages=messages,
        )
        try:
            result = json.loads(response.choices[0].message.content or "")
        except json.JSONDecodeError:
            continue
        if all(key in result for key in SCHEMA["required"]):
            # The model sometimes wraps the quote in its own quotation marks
            result["evidence"] = result["evidence"].strip().strip("\"'“”")
            return result
    raise RuntimeError("The model's reply couldn't be read, even after a retry. Please try again.")


def speak(client, text, language):
    """Return the spoken explanation as MP3 bytes."""
    voice = KID_VOICES.get(language)
    model, speaker = ("bulbul:v4-flash", voice) if voice else ("bulbul:v3", "ritu")
    result = client.text_to_speech.convert(
        text=text,
        language_code=language,
        model=model,
        speaker=speaker,
        pace=0.9,  # a little slower for children
        output_audio_codec="mp3",
    )
    return base64.b64decode(result.audios[0])


def main():
    parser = argparse.ArgumentParser(description="Find the thinking mistake behind a child's wrong maths answer.")
    parser.add_argument("audio", help="Recording of the child explaining their answer (up to 30 seconds)")
    parser.add_argument("--question", required=True, help='The maths question, e.g. "1/2 + 1/3 = ?"')
    parser.add_argument("--answer", required=True, help='The child\'s answer, e.g. "2/5"')
    parser.add_argument("--out", default="reply.mp3", help="Where to save the spoken explanation")
    args = parser.parse_args()

    client = make_client()

    audio = Path(args.audio)
    transcript, language = transcribe(client, audio.name, audio.read_bytes())
    print(f"Child said ({language}): {transcript}\n")

    try:
        result = diagnose(client, args.question, args.answer, transcript, language)
    except RuntimeError as err:
        sys.exit(str(err))
    print(f"Correct answer: {result['correct_answer']}")
    print(f"Mistake:        {result['mistake']} [{result['mistake_id']}, {result['confidence']} confidence]")
    print(f"Evidence:       \"{result['evidence']}\"\n")
    print(f"Explanation:\n{result['explanation']}\n")
    print("Practice:")
    for i, q in enumerate(result["practice_questions"], 1):
        print(f"  {i}. {q['question']}  (answer: {q['answer']})")

    if language in TTS_LANGUAGES:
        Path(args.out).write_bytes(speak(client, result["explanation"], language))
        print(f"\nSpoken explanation saved to {args.out}")
    else:
        print(f"\nNo voice available for {language}, so the explanation is text only.")


if __name__ == "__main__":
    main()
