"""Single command line for preparation, paired runs and offline reporting."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sys

from .data import load_manifest, prepare_manifest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_new(path, value, *, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        if jsonl:
            for item in value:
                handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def summarize_directory(directory):
    from .reporting import summarize_records
    directory = Path(directory)
    protocol = read_json(directory / "protocol.json")
    manifest = load_manifest(directory / "manifest.json")
    # Per-arm files are authoritative, including a partially interrupted batch.
    records = [read_json(path) for path in sorted(directory.glob("cases/*/*/result.json"))]
    ledger_paths = {directory / "calls.jsonl"}
    for case in manifest["cases"]:
        ledger = case.get("upstream", {}).get("ledger_path")
        if ledger:
            ledger_paths.add(Path(ledger))
    events = {}
    for path in ledger_paths:
        for event in read_jsonl(path):
            call_id = event.get("call_id")
            if call_id in events and events[call_id] != event:
                raise ValueError(f"Conflicting records for call id {call_id}")
            events[call_id] = event
    return summarize_records(records, list(events.values()), planned=protocol)


def parser():
    command = argparse.ArgumentParser(description="Versioned Native/NED rebuttal experiments")
    sub = command.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Create a candidate manifest; never overwrite source data")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--format", dest="source_format", choices=["nora", "optibench"], default="nora")
    prepare.add_argument("--dataset-version")
    prepare.add_argument("--scoring", dest="scoring_name", choices=[
        "nextor_relative", "optibench_absolute", "optmath_normalized", "exact"],
        help="Explicit evaluation protocol; omission leaves the candidate awaiting review")
    prepare.add_argument("--nonlinear-only", action="store_true")
    prepare.add_argument("--output", required=True)
    inspect = sub.add_parser("inspect", help="Validate and count a manifest without model calls")
    inspect.add_argument("--manifest", required=True)
    extract = sub.add_parser("extract", help="Generate shared mathematical inputs with recorded upstream calls")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--model-config", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--no-refine", action="store_true")
    run = sub.add_parser("run", help="Run the two representations on identical frozen inputs")
    run.add_argument("--manifest", required=True)
    run.add_argument("--model-config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--framework", choices=["ours", "standard"], default="ours")
    run.add_argument("--python", dest="python_executable", default=sys.executable)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--repairs", type=int, default=0)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--solver-time-limit", type=float, default=60)
    run.add_argument("--process-timeout", type=float, default=180)
    run.add_argument("--solver-threads", type=int, default=1)
    run.add_argument("--allow-unreviewed", action="store_true")
    run.add_argument("--resume", action="store_true")
    summary = sub.add_parser("summary", help="Rebuild tables from saved outcomes and call ledgers")
    summary.add_argument("--run-dir", required=True)
    summary.add_argument("--output")
    rescore = sub.add_parser("rescore", help="Score the same saved predictions at stricter NEXTOR thresholds")
    rescore.add_argument("--run-dir", required=True)
    rescore.add_argument("--thresholds", default="0.01,0.001,0.0001")
    rescore.add_argument("--exclude-case", action="append", default=[])
    rescore.add_argument("--output", required=True)
    return command


async def online(args):
    from .models import LoggedModel, ModelConfig
    from .runner import extract_inputs, run_paired, write_json
    manifest = load_manifest(args.manifest)
    config = ModelConfig.from_dict(read_json(args.model_config))
    output = Path(args.output).resolve()
    ledger = output / "calls.jsonl" if args.command == "run" else output.parent / (output.stem + "_calls.jsonl")
    model = LoggedModel(config, ledger)
    try:
        if args.command == "extract":
            extracted = await extract_inputs(manifest, model, output, refine=not args.no_refine)
            return {"output": str(output), "cases": len(extracted["cases"]),
                    "upstream_status": dict(Counter(c["upstream"]["status"] for c in extracted["cases"]))}
        results = await run_paired(
            manifest, model, output, framework=args.framework, repeats=args.repeats,
            max_repairs=args.repairs, concurrency=args.concurrency,
            python_executable=args.python_executable, timeout_seconds=args.process_timeout,
            solver_time_limit=args.solver_time_limit, solver_threads=args.solver_threads,
            allow_unreviewed=args.allow_unreviewed, resume=args.resume)
        summary = summarize_directory(output)
        write_json(output / "summary.json", summary)
        return {"output": str(output), "completed_records": len(results), "summary": summary}
    finally:
        await model.close()


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_manifest(args.source, args.dataset, args.source_format,
                                    args.dataset_version, nonlinear_only=args.nonlinear_only,
                                    scoring_name=args.scoring_name)
        save_new(args.output, manifest)
        result = {"output": str(Path(args.output).resolve()), "cases": len(manifest["cases"]),
                  "review_status": dict(Counter(c["review_status"] for c in manifest["cases"]))}
    elif args.command == "inspect":
        manifest = load_manifest(args.manifest)
        result = {"dataset": manifest["dataset"], "dataset_version": manifest["dataset_version"],
                  "cases": len(manifest["cases"]),
                  "with_math_input": sum(c["math_input"] is not None for c in manifest["cases"]),
                  "review_status": dict(Counter(c["review_status"] for c in manifest["cases"]))}
    elif args.command in {"extract", "run"}:
        result = asyncio.run(online(args))
    elif args.command == "summary":
        result = summarize_directory(args.run_dir)
        if args.output:
            save_new(args.output, result)
    else:
        from .reporting import rescore_records
        directory = Path(args.run_dir)
        manifest = load_manifest(directory / "manifest.json")
        records = [read_json(path) for path in sorted(directory.glob("cases/*/*/result.json"))]
        thresholds = tuple(float(value) for value in args.thresholds.split(","))
        rescored = rescore_records(records, manifest, thresholds, args.exclude_case)
        save_new(args.output, rescored, jsonl=True)
        result = {"output": str(Path(args.output).resolve()), "records": len(rescored),
                  "thresholds": thresholds, "excluded_cases": args.exclude_case}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
