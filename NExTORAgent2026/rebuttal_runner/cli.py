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


def read_arm_records(directory):
    # Auxiliary/upstream logs also have result.json, but are not scored trials.
    return [read_json(path) for path in sorted(Path(directory).glob("cases/*/*/result.json"))
            if path.parent.name in {"native", "ned"}]


def summarize_directory(directory):
    from .reporting import summarize_ledger, summarize_records
    directory = Path(directory)
    protocol = read_json(directory / "protocol.json")
    manifest = load_manifest(directory / "manifest.json")
    # Per-arm files are authoritative, including a partially interrupted batch.
    records = read_arm_records(directory)
    ledger_paths = {directory / "calls.jsonl"}
    for case in manifest["cases"]:
        for stage in ("upstream", "auxiliary"):
            ledger = (case.get(stage) or {}).get("ledger_path")
            if ledger:
                ledger_paths.add(Path(ledger))
    events = {}
    for path in ledger_paths:
        for event in read_jsonl(path):
            call_id = event.get("call_id")
            if call_id in events and events[call_id] != event:
                raise ValueError(f"Conflicting records for call id {call_id}")
            events[call_id] = event
    summary = summarize_records(records, list(events.values()), planned=protocol)
    summary["current_run_ledger"] = summarize_ledger(read_jsonl(directory / "calls.jsonl"), records)
    return summary


def experiment_arguments(command, *, execute):
    command.add_argument("--manifest", required=True)
    command.add_argument("--model-config", required=True)
    command.add_argument("--output", required=execute)
    command.add_argument("--level", choices=["coding", "math", "end_to_end"], required=True)
    command.add_argument("--framework", choices=["ours", "standard", "coe", "optimus", "or_llm_agent"], default="ours")
    command.add_argument("--arms", nargs="+", choices=["native", "ned"])
    command.add_argument("--baseline-root")
    command.add_argument("--repeats", type=int, default=1)
    command.add_argument("--repairs", type=int, default=0)
    command.add_argument("--max-calls", type=int, default=64)
    command.add_argument("--max-upstream-calls", type=int, default=32)
    command.add_argument("--solver-time-limit", type=float, default=60)
    command.add_argument("--process-timeout", type=float, default=180)
    command.add_argument("--solver-threads", type=int, default=1)
    if execute:
        command.add_argument("--python", dest="python_executable", default=sys.executable)
        command.add_argument("--concurrency", type=int, default=1)
        command.add_argument("--allow-unreviewed", action="store_true")
        command.add_argument("--resume", action="store_true")


def parser():
    command = argparse.ArgumentParser(description="Versioned Native/NED rebuttal experiments")
    sub = command.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Create a candidate manifest; never overwrite source data")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--format", dest="source_format", choices=["nora", "optibench"], default="nora")
    prepare.add_argument("--dataset-version")
    prepare.add_argument("--scoring", dest="scoring_name", choices=[
        "objective_relative", "nextor_relative", "optibench_absolute", "optmath_normalized", "exact"],
        help="Explicit evaluation protocol; omission leaves the candidate awaiting review")
    prepare.add_argument("--nonlinear-only", action="store_true")
    prepare.add_argument("--output", required=True)
    inspect = sub.add_parser("inspect", help="Validate and count a manifest without model calls")
    inspect.add_argument("--manifest", required=True)
    attach = sub.add_parser("attach-inputs", help="Attach cached math only when source ID and question agree")
    attach.add_argument("--manifest", required=True)
    attach.add_argument("--cache-manifest", required=True)
    attach.add_argument("--output", required=True)
    extract = sub.add_parser("extract", help="Generate shared mathematical inputs with recorded upstream calls")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--model-config", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--max-calls", type=int, default=64)
    run = sub.add_parser("run", help="Run one explicit experiment level and workflow")
    experiment_arguments(run, execute=True)
    plan = sub.add_parser("plan", help="Check configuration and call ceilings offline; no API/solver calls")
    experiment_arguments(plan, execute=False)
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
    from .planning import resolved_arms
    from .runner import extract_inputs, run_paired, write_json
    manifest = load_manifest(args.manifest)
    config = ModelConfig.from_dict(read_json(args.model_config))
    output = Path(args.output).resolve()
    ledger = output / "calls.jsonl" if args.command == "run" else output.parent / (output.stem + "_calls.jsonl")
    model = LoggedModel(config, ledger)
    try:
        if args.command == "extract":
            extracted = await extract_inputs(manifest, model, output, max_calls=args.max_calls)
            return {"output": str(output), "cases": len(extracted["cases"]),
                    "upstream_status": dict(Counter(c["upstream"]["status"] for c in extracted["cases"]))}
        results = await run_paired(
            manifest, model, output, framework=args.framework, level=args.level,
            arms=resolved_arms(args.level, args.framework, args.arms), repeats=args.repeats,
            max_repairs=args.repairs, concurrency=args.concurrency,
            python_executable=args.python_executable, timeout_seconds=args.process_timeout,
            solver_time_limit=args.solver_time_limit, solver_threads=args.solver_threads,
            allow_unreviewed=args.allow_unreviewed, resume=args.resume,
            source_root=args.baseline_root, max_calls=args.max_calls,
            max_upstream_calls=args.max_upstream_calls)
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
    elif args.command == "attach-inputs":
        from .planning import attach_cached_inputs
        attached = attach_cached_inputs(load_manifest(args.manifest), load_manifest(args.cache_manifest),
                                        cache_path=args.cache_manifest)
        save_new(args.output, attached)
        result = {"output": str(Path(args.output).resolve()), **attached["cached_input_alignment"]}
    elif args.command == "plan":
        from .models import ModelConfig
        from .planning import plan_run
        result = plan_run(load_manifest(args.manifest), ModelConfig.from_dict(read_json(args.model_config)),
                          framework=args.framework, level=args.level, arms=args.arms,
                          repeats=args.repeats, max_repairs=args.repairs, max_calls=args.max_calls,
                          max_upstream_calls=args.max_upstream_calls, source_root=args.baseline_root,
                          solver_time_limit=args.solver_time_limit, process_timeout=args.process_timeout,
                          solver_threads=args.solver_threads)
        if args.output:
            save_new(args.output, result)
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
        records = read_arm_records(directory)
        thresholds = tuple(float(value) for value in args.thresholds.split(","))
        rescored = rescore_records(records, manifest, thresholds, args.exclude_case)
        save_new(args.output, rescored, jsonl=True)
        result = {"output": str(Path(args.output).resolve()), "records": len(rescored),
                  "thresholds": thresholds, "excluded_cases": args.exclude_case}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
