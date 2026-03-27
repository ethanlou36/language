import argparse
import json
from pathlib import Path


def _decode_serialized_text(text: str) -> str:
    if "\n" in text:
        return text

    if text.startswith('"') and text.endswith('"'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    if any(token in text for token in (r"\n", r"\"", r"\t", r"\r")):
        try:
            return json.loads(f'"{text}"')
        except json.JSONDecodeError:
            return text

    return text


def _extract_final_code_block(text: str) -> str:
    start_tag = "<final_code>"
    end_tag = "</final_code>"

    if start_tag in text and end_tag in text:
        start = text.index(start_tag) + len(start_tag)
        end = text.index(end_tag, start)
        return text[start:end].strip()

    return text


def normalize_model_output(text: str) -> str:
    text = _extract_final_code_block(text.strip())
    text = _decode_serialized_text(text)

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()

    return text.strip() + "\n"


def _select_variant_payload(model: dict, variant: str) -> dict:
    key = f"{variant}_output"
    payload = model.get(key)
    if isinstance(payload, dict):
        return payload
    return model


def _select_model_text(model: dict, field: str, variant: str) -> str:
    payload = _select_variant_payload(model, variant)

    if field != "auto":
        return payload.get(field, "")

    if payload.get("raw_output"):
        return payload["raw_output"]
    if payload.get("generated_text"):
        return payload["generated_text"]
    return ""


def load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text

    if args.input_json is not None:
        payload = json.loads(args.input_json.read_text(encoding="utf-8"))
        models = payload.get("models", [])
        if not models:
            raise ValueError("No models found in the JSON input.")

        if args.model_key is not None:
            for model in models:
                if model.get("model_key") == args.model_key:
                    return _select_model_text(model, args.field, args.variant)
            raise ValueError(f"Model key not found: {args.model_key}")

        return _select_model_text(models[0], args.field, args.variant)

    if args.input_file is not None:
        return args.input_file.read_text(encoding="utf-8")

    raise ValueError("Provide --text, --input-file, or --input-json.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert escaped model output into a readable C++ source file."
    )
    parser.add_argument(
        "--text",
        help="Raw model output string, for example one containing escaped \\\\n sequences.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to a text file containing the raw model output.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Path to model_outputs.json. Uses the first model by default.",
    )
    parser.add_argument(
        "--model-key",
        help="Optional model_key to select when using --input-json.",
    )
    parser.add_argument(
        "--field",
        choices=("auto", "raw_output", "generated_text", "reasoning_trace"),
        default="auto",
        help="Which field to use from model_outputs.json. 'auto' prefers raw_output, then generated_text.",
    )
    parser.add_argument(
        "--variant",
        choices=("english", "chinese"),
        default="english",
        help="Which language-specific output block to use from model_outputs.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("answer.cpp"),
        help="Where to write the normalized C++ program.",
    )
    args = parser.parse_args()

    normalized = normalize_model_output(load_text(args))
    args.output.write_text(normalized, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
