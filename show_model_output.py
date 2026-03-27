import argparse
import json
from pathlib import Path


def _select_model(payload: dict, model_key: str | None) -> dict:
    models = payload.get("models", [])
    if not models:
        raise ValueError("No models found in the JSON input.")

    if model_key is None:
        return models[0]

    for model in models:
        if model.get("model_key") == model_key:
            return model

    raise ValueError(f"Model key not found: {model_key}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Display English and Chinese outputs from model_outputs.json."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("model_outputs.json"),
        help="Path to model_outputs.json.",
    )
    parser.add_argument(
        "--model-key",
        help="Optional model_key to select. Defaults to the first model entry.",
    )
    parser.add_argument(
        "--variant",
        choices=("english", "chinese", "both"),
        default="both",
        help="Which variant to display.",
    )
    parser.add_argument(
        "--field",
        choices=("raw_output", "reasoning_trace", "generated_text"),
        default="raw_output",
        help="Which field to display from the selected variant.",
    )
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    model = _select_model(payload, args.model_key)

    variants = ["english", "chinese"] if args.variant == "both" else [args.variant]
    for variant in variants:
        variant_payload = model.get(f"{variant}_output", {})
        print(f"=== {model.get('display_name')} / {variant} / {args.field} ===")
        print(variant_payload.get(args.field, ""))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
