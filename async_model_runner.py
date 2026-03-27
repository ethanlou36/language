import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from fetch import fetch_problems


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert competitive programmer. You may reason before answering, but your final answer "
    "must contain the C++ solution inside <final_code>...</final_code> tags."
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    repo_id: str


MODEL_SPECS = {
    "qwen25-coder-7b": ModelSpec(
        key="qwen25-coder-7b",
        display_name="Qwen2.5-Coder-7B-Instruct",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    ),
    "qwen25-coder-32b": ModelSpec(
        key="qwen25-coder-32b",
        display_name="Qwen2.5-Coder-32B-Instruct",
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    ),
    "qwen25-coder-14b": ModelSpec(
        key="qwen25-coder-14b",
        display_name="Qwen2.5-Coder-14B-Instruct",
        repo_id="Qwen/Qwen2.5-Coder-14B-Instruct",
    ),
    "codeqwen15-7b": ModelSpec(
        key="codeqwen15-7b",
        display_name="CodeQwen1.5-7B-Chat",
        repo_id="Qwen/CodeQwen1.5-7B-Chat",
    ),
    "deepseek-coder-v2": ModelSpec(
        key="deepseek-coder-v2",
        display_name="DeepSeek-Coder-V2-Instruct",
        repo_id="deepseek-ai/DeepSeek-Coder-V2-Instruct",
    ),
    "deepseek-coder-33b": ModelSpec(
        key="deepseek-coder-33b",
        display_name="DeepSeek-Coder-33B-Instruct",
        repo_id="deepseek-ai/deepseek-coder-33b-instruct",
    ),
    "deepseek-coder-6.7b": ModelSpec(
        key="deepseek-coder-6.7b",
        display_name="DeepSeek-Coder-6.7B-Instruct",
        repo_id="deepseek-ai/deepseek-coder-6.7b-instruct",
    ),
    "deepseek-coder-1.3b": ModelSpec(
        key="deepseek-coder-1.3b",
        display_name="DeepSeek-Coder-1.3B-Instruct",
        repo_id="deepseek-ai/deepseek-coder-1.3b-instruct",
    ),
}

DEFAULT_MODEL_KEYS = [
    "qwen25-coder-14b",
    "deepseek-coder-6.7b",
]


def _build_problem_prompt(problem: dict[str, Any], system_prompt: str) -> str:
    title = problem.get("name", "Untitled Problem")
    statement = problem.get("statement", "").strip()
    sample_tests = problem.get("sample_tests", [])

    prompt_parts = [system_prompt, "", f"Problem: {title}"]

    if problem.get("problem_url"):
        prompt_parts.append(f"Source: {problem['problem_url']}")

    prompt_parts.extend(["", "Statement:", statement or "No statement available."])

    if sample_tests:
        prompt_parts.append("")
        prompt_parts.append("Sample Tests:")
        for index, sample in enumerate(sample_tests, start=1):
            prompt_parts.append(f"Sample {index} Input:")
            prompt_parts.append(sample.get("input", ""))
            prompt_parts.append(f"Sample {index} Output:")
            prompt_parts.append(sample.get("output", ""))

    prompt_parts.append("")
    prompt_parts.append(
        "Think through edge cases before writing code. You may include brief reasoning before the final answer if needed."
    )
    prompt_parts.append(
        "Your final answer must end with <final_code> on its own line, then the complete C++ solution, then </final_code> on its own line."
    )
    return "\n".join(prompt_parts).strip()


def _extract_code_only(text: str) -> str:
    stripped = text.strip()
    start_tag = "<final_code>"
    end_tag = "</final_code>"

    if start_tag in stripped and end_tag in stripped:
        start = stripped.index(start_tag) + len(start_tag)
        end = stripped.index(end_tag, start)
        stripped = stripped[start:end].strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()
        return "\n".join(lines).strip()
    return stripped


def _load_problem_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.problem_json:
        with open(args.problem_json, "r", encoding="utf-8") as handle:
            return json.load(handle)

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    problems = fetch_problems(
        tags=tags,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        limit=args.problem_offset + 1,
    )
    if len(problems) <= args.problem_offset:
        raise ValueError("No problem matched the requested filters.")
    return problems[args.problem_offset]


def _build_parent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Asynchronously run multiple local Hugging Face coding models against one problem."
        )
    )
    parser.add_argument(
        "--problem-json",
        type=Path,
        help="Path to a JSON file containing one problem dict with statement/sample_tests.",
    )
    parser.add_argument(
        "--tags",
        default="math",
        help="Comma-separated Codeforces tags to use when fetching a problem via fetch.py.",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=800,
        help="Minimum Codeforces rating when fetching a problem.",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=800,
        help="Maximum Codeforces rating when fetching a problem.",
    )
    parser.add_argument(
        "--problem-offset",
        type=int,
        default=0,
        help="Zero-based index into the filtered problem list.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODEL_KEYS,
        choices=list(MODEL_SPECS),
        help="Model keys to run. Defaults to the smaller local-friendly models only.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum number of worker subprocesses to run at once. Defaults to the number of selected models.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens per model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p value when temperature > 0.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./hf_cache"),
        help="Directory used by snapshot_download for model caching.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("./model_outputs.json"),
        help="Where to write the aggregated JSON results.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token if needed. Defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Instruction prepended to the problem prompt.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Preferred execution device inside each worker.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Preferred torch dtype inside each worker.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the available model keys and exit.",
    )
    return parser


def _build_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--hf-token")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True)
    return parser


async def _run_worker(
    model_key: str,
    prompt_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        __file__,
        "--worker",
        "--model-key",
        model_key,
        "--prompt-file",
        str(prompt_file),
        "--cache-dir",
        str(args.cache_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        *(["--hf-token", args.hf_token] if args.hf_token else []),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        return {
            "model_key": model_key,
            "display_name": MODEL_SPECS[model_key].display_name,
            "repo_id": MODEL_SPECS[model_key].repo_id,
            "status": "error",
            "error": stderr_text or f"Worker exited with code {process.returncode}.",
        }

    try:
        return json.loads(stdout.decode("utf-8"))
    except json.JSONDecodeError:
        return {
            "model_key": model_key,
            "display_name": MODEL_SPECS[model_key].display_name,
            "repo_id": MODEL_SPECS[model_key].repo_id,
            "status": "error",
            "error": "Worker returned invalid JSON.",
            "raw_stdout": stdout.decode("utf-8", errors="replace"),
            "raw_stderr": stderr.decode("utf-8", errors="replace"),
        }


async def _run_all_models(
    prompt_file: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    max_concurrent = args.max_concurrent or len(args.models)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(model_key: str) -> dict[str, Any]:
        async with semaphore:
            return await _run_worker(model_key=model_key, prompt_file=prompt_file, args=args)

    tasks = [asyncio.create_task(run_one(model_key)) for model_key in args.models]
    return await asyncio.gather(*tasks)


def _detect_device(torch_module, requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(torch_module, device: str, requested_dtype: str):
    if requested_dtype == "bfloat16":
        return torch_module.bfloat16
    if requested_dtype == "float16":
        return torch_module.float16
    if requested_dtype == "float32":
        return torch_module.float32

    if device == "cuda":
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16
        return torch_module.float16
    if device == "mps":
        return torch_module.float16
    return torch_module.float32


def _prepare_inputs(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]

    if getattr(tokenizer, "chat_template", None):
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        model_inputs = tokenizer(prompt, return_tensors="pt")

    return {name: tensor.to(device) for name, tensor in model_inputs.items()}


def _input_device_for_model(model, requested_device: str) -> str:
    if requested_device == "cuda":
        return str(next(model.parameters()).device)
    return str(model.device)


def _run_worker_main(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    spec = MODEL_SPECS[args.model_key]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        prompt = args.prompt_file.read_text(encoding="utf-8")
        device = _detect_device(torch, args.device)
        dtype = _resolve_dtype(torch, device, args.dtype)

        local_model_path = snapshot_download(
            repo_id=spec.repo_id,
            cache_dir=str(args.cache_dir),
            token=args.hf_token,
            resume_download=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        if device == "mps":
            model_kwargs["attn_implementation"] = "eager"

        model = AutoModelForCausalLM.from_pretrained(local_model_path, **model_kwargs)
        if device in {"cpu", "mps"}:
            model = model.to(device)
        model.eval()

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = _prepare_inputs(tokenizer, prompt, _input_device_for_model(model, device))
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if args.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = model.generate(**inputs, **generation_kwargs)

        completion = outputs[0][prompt_tokens:]
        generated_text = _extract_code_only(
            tokenizer.decode(completion, skip_special_tokens=True)
        )

        result = {
            "model_key": spec.key,
            "display_name": spec.display_name,
            "repo_id": spec.repo_id,
            "status": "ok",
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "local_model_path": local_model_path,
            "elapsed_seconds": round(time.perf_counter() - started_at, 2),
            "generated_text": generated_text,
        }
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        error_result = {
            "model_key": spec.key,
            "display_name": spec.display_name,
            "repo_id": spec.repo_id,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(time.perf_counter() - started_at, 2),
        }
        print(json.dumps(error_result, ensure_ascii=False))
        return 0


async def _run_parent_main(args: argparse.Namespace) -> int:
    if args.list_models:
        for spec in MODEL_SPECS.values():
            print(f"{spec.key}: {spec.display_name} ({spec.repo_id})")
        return 0

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    problem = _load_problem_from_args(args)
    prompt = _build_problem_prompt(problem, system_prompt=args.system_prompt)

    with tempfile.TemporaryDirectory() as temp_dir:
        prompt_file = Path(temp_dir) / "problem_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        results = await _run_all_models(prompt_file=prompt_file, args=args)

    payload = {
        "problem": {
            "name": problem.get("name"),
            "contestId": problem.get("contestId"),
            "index": problem.get("index"),
            "rating": problem.get("rating"),
            "problem_url": problem.get("problem_url"),
        },
        "models": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    if "--worker" in sys.argv:
        worker_parser = _build_worker_parser()
        worker_args = worker_parser.parse_args()
        return _run_worker_main(worker_args)

    parent_parser = _build_parent_parser()
    parent_args = parent_parser.parse_args()
    return asyncio.run(_run_parent_main(parent_args))


if __name__ == "__main__":
    raise SystemExit(main())
