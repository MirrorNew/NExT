import copy
import unittest

from rebuttal_runner.reporting import rescore_records, summarize_ledger, summarize_records


def make_record(
    case_id,
    arm="native",
    *,
    dataset="NEXTOR",
    version="v1",
    model="m1",
    framework="f1",
    source_id="source-1",
    repeat=1,
    input_hash="hash-1",
    correct=True,
    prediction=1,
    call_ids=None,
    upstream_call_ids=None,
    elapsed=1.0,
    status="ok",
):
    record = {
        "dataset": dataset,
        "dataset_version": version,
        "case_id": case_id,
        "source_id": source_id,
        "model": model,
        "framework": framework,
        "arm": arm,
        "repeat": repeat,
        "input_hash": input_hash,
        "status": status,
        "score": {"correct": correct},
        "call_ids": call_ids or [],
        "upstream_call_ids": upstream_call_ids or [],
        "execution": {"elapsed_seconds": elapsed, "solver_status": "optimal"},
        "ned_trace": {"summary": "trace"} if arm == "ned" else {},
    }
    if prediction is not ...:
        record["prediction"] = prediction
    return record


def find_comparison(report, **scope):
    for item in report["comparisons"]:
        if all(item.get(key) == value for key, value in scope.items()):
            return item
    raise AssertionError(f"comparison scope not found: {scope}")


class SummaryTests(unittest.TestCase):
    def test_solver_runtime_is_aggregated_separately_and_missing_legacy_stays_missing(self):
        measured = make_record("measured", elapsed=5.0)
        measured["execution"] = {
            "elapsed_seconds": 5.0,
            "solver_limits_applied": True,
            "solver_status": {
                "status": 2,
                "sol_count": 1,
                "optimize_call_count": 2,
                "optimize_calls": [
                    {"runtime_seconds": 0.2},
                    {"runtime_seconds": 0.3},
                ],
            },
        }
        legacy = make_record("legacy", elapsed=2.0)
        report = summarize_records([measured, legacy])["overall"]["execution"]
        self.assertEqual(report["known_total_seconds"], 7.0)
        self.assertEqual(report["solver_runtime"]["optimize_call_count"], 2)
        self.assertAlmostEqual(report["solver_runtime"]["known_total_seconds"], 0.5)
        self.assertIsNone(report["solver_runtime"]["full_total_seconds"])
        self.assertEqual(report["solver_runtime"]["missing_runtime_runs"], 1)

    def test_missing_runtime_in_new_log_counts_call_and_keeps_partial_total(self):
        record = make_record("partial")
        record["execution"] = {
            "elapsed_seconds": 3.0,
            "solver_status": {
                "optimize_call_count": 2,
                "optimize_calls": [
                    {"runtime_seconds": 0.4},
                    {"runtime_seconds": None},
                ],
            },
        }
        runtime = summarize_records([record])["overall"]["execution"]["solver_runtime"]
        self.assertEqual(runtime["optimize_call_count"], 2)
        self.assertEqual(runtime["known_total_seconds"], 0.4)
        self.assertEqual(runtime["missing_runtime_count"], 1)
        self.assertIsNone(runtime["full_total_seconds"])

    def test_missing_execution_distinguishes_explicit_upstream_failure_from_success(self):
        upstream_failure = make_record("upstream", status="upstream_failed")
        upstream_failure["execution"] = None
        success_without_execution = make_record("success", status="completed")
        success_without_execution["execution"] = None

        failed_runtime = summarize_records([upstream_failure])["overall"]["execution"]["solver_runtime"]
        self.assertEqual(failed_runtime["missing_runtime_runs"], 0)
        self.assertEqual(failed_runtime["full_total_seconds"], 0)

        success_runtime = summarize_records([success_without_execution])["overall"]["execution"]["solver_runtime"]
        self.assertEqual(success_runtime["missing_runtime_runs"], 1)
        self.assertIsNone(success_runtime["full_total_seconds"])

    def test_ledger_retains_recorded_calls_without_finished_arm(self):
        ledger = summarize_ledger([{"call_id": "orphan", "total_tokens": 12},
                                   {"call_id": "failed", "total_tokens": None}], [])
        self.assertEqual(ledger["recorded_call_count"], 2)
        self.assertEqual(ledger["known_total_tokens"], 12)
        self.assertIsNone(ledger["full_total_tokens"])
        self.assertEqual(ledger["unattributed_call_ids"], ["failed", "orphan"])

    def test_missing_usage_is_not_zero_and_shared_calls_are_separated(self):
        records = [
            make_record("c1", "native", call_ids=["own-1", "shared"], upstream_call_ids=["shared"]),
            make_record("c1", "ned", call_ids=["own-2"], upstream_call_ids=["shared"]),
            make_record("c2", call_ids=["known", "missing"]),
        ]
        events = [
            {"call_id": "shared", "total_tokens": 7},
            {"call_id": "own-1", "total_tokens": 5},
            {"call_id": "own-2", "total_tokens": 6},
            {"call_id": "known", "total_tokens": 4},
        ]

        report = summarize_records(records, call_events=events)
        usage = report["overall"]["usage"]
        self.assertIsNone(usage["attributed"]["full_total_tokens"])
        self.assertEqual(usage["attributed"]["known_total_tokens"], 29)
        self.assertEqual(usage["attributed"]["missing"][0]["call_id"], "missing")
        self.assertEqual(usage["unique_execution"]["known_total_tokens"], 22)
        self.assertEqual(usage["unique_execution"]["missing"], ["missing"])
        self.assertIsNone(usage["unique_execution"]["full_total_tokens"])

        complete = summarize_records(records[:2], call_events=events)["overall"]["usage"]
        self.assertEqual(complete["attributed"]["full_total_tokens"], 25)
        self.assertEqual(complete["unique_execution"]["full_total_tokens"], 18)

    def test_real_nested_ledger_usage_flows_through_reports_and_shared_attribution(self):
        records = [
            make_record("c1", "native", call_ids=["own-1", "shared"],
                        upstream_call_ids=["shared"]),
            make_record("c1", "ned", call_ids=["own-2"],
                        upstream_call_ids=["shared"]),
        ]
        events = [
            {"call_id": "shared", "usage": {"total_tokens": 7}},
            {"call_id": "own-1", "usage": {"total_tokens": 5}},
            {"call_id": "own-2", "usage": {"total_tokens": 6}},
            {"call_id": "orphan", "usage": {"total_tokens": 3}},
        ]

        report = summarize_records(records, call_events=events)
        overall = report["overall"]["usage"]
        self.assertEqual(overall["attributed"]["full_total_tokens"], 25)
        self.assertEqual(overall["unique_execution"]["full_total_tokens"], 18)
        groups = {group["arm"]: group["usage"] for group in report["groups"]}
        self.assertEqual(groups["native"]["attributed"]["full_total_tokens"], 12)
        self.assertEqual(groups["native"]["unique_execution"]["full_total_tokens"], 12)
        self.assertEqual(groups["ned"]["attributed"]["full_total_tokens"], 13)
        self.assertEqual(groups["ned"]["unique_execution"]["full_total_tokens"], 13)

        ledger = summarize_ledger(events, records)
        self.assertEqual(ledger["recorded_call_count"], 4)
        self.assertEqual(ledger["known_total_tokens"], 21)
        self.assertEqual(ledger["full_total_tokens"], 21)
        self.assertEqual(ledger["unattributed_call_ids"], ["orphan"])

    def test_nested_and_legacy_usage_conflict_is_missing_not_silently_counted(self):
        events = [{"call_id": "conflict", "total_tokens": 5,
                   "usage": {"total_tokens": 6}}]
        record = make_record("c1", call_ids=["conflict"])

        usage = summarize_records([record], call_events=events)["overall"]["usage"]
        self.assertIsNone(usage["attributed"]["full_total_tokens"])
        self.assertEqual(usage["attributed"]["known_total_tokens"], 0)
        self.assertEqual(usage["attributed"]["missing"][0]["call_id"], "conflict")
        ledger = summarize_ledger(events, [record])
        self.assertIsNone(ledger["full_total_tokens"])
        self.assertEqual(ledger["missing_usage_call_ids"], ["conflict"])

    def test_missing_prediction_and_failed_score_stay_in_denominator(self):
        no_prediction = make_record("c1", prediction=..., correct=True)
        failed_score = make_record("c2", prediction=10, correct=False, status="failed")
        report = summarize_records([no_prediction, failed_score])
        self.assertEqual(report["overall"]["run_count"], 2)
        self.assertEqual(report["overall"]["question_count"], 2)
        self.assertEqual(report["overall"]["correct_run_count"], 0)
        self.assertEqual(report["overall"]["accuracy"], 0)
        self.assertEqual(report["overall"]["status_counts"], {"failed": 1, "ok": 1})

    def test_planned_subset_is_reported_incomplete_and_excluded_rows_do_not_count(self):
        records = [
            make_record("c1", "native", repeat=1),
            make_record("c1", "ned", repeat=1),
            {**make_record("c2", "native"), "excluded": True},
        ]
        report = summarize_records(
            records,
            planned={"case_ids": ["c1", "c2"], "repeats": 2, "arms": ["native", "ned"]},
        )
        self.assertEqual(report["overall"]["run_count"], 2)
        self.assertEqual(report["planning"]["planned_per_arm"], 4)
        self.assertEqual(report["planning"]["missing_records"], 6)
        self.assertFalse(report["planning"]["complete"])


class PairingTests(unittest.TestCase):
    def test_pairing_rejects_hash_mismatch_and_reports_unmatched(self):
        records = [
            make_record("c1", "native", correct=False),
            make_record("c1", "ned", correct=True),
            make_record("c2", "native", input_hash="native-hash", correct=False),
            make_record("c2", "ned", input_hash="ned-hash", correct=True),
            make_record("c3", "native", correct=True),
        ]
        pair = find_comparison(summarize_records(records), dataset="NEXTOR", dataset_version="v1", model="m1", framework="f1")
        self.assertEqual(pair["valid_pair_run_count"], 1)
        self.assertEqual(pair["paired_case_count"], 1)
        self.assertEqual(pair["ned_wins"], 1)
        self.assertEqual(pair["invalid_pair_count"], 1)
        self.assertEqual(pair["invalid_pairs"][0]["reason"], "input_hash_mismatch")
        self.assertEqual(pair["unmatched_record_count"], 1)
        self.assertEqual(pair["unmatched_records"][0]["case_id"], "c3")
        self.assertIsNone(pair["ci_95"])
        self.assertEqual(pair["ci_reason"], "fewer_than_two_independent_sources")

    def test_repeats_average_per_case_before_source_cluster_bootstrap(self):
        records = []
        for case_id, source_id, arm_correct in (
            ("c1", "source-1", [(True, False), (False, False)]),
            ("c2", "source-1", [(True, True), (True, False)]),
            ("c3", "source-2", [(False, True), (False, True)]),
        ):
            for repeat, (native_correct, ned_correct) in enumerate(arm_correct, start=1):
                records.append(make_record(case_id, "native", source_id=source_id, repeat=repeat, correct=native_correct))
                records.append(make_record(case_id, "ned", source_id=source_id, repeat=repeat, correct=ned_correct))

        report = summarize_records(records)
        pair = find_comparison(report, model="m1", framework="f1")
        self.assertEqual(pair["paired_case_count"], 3)
        self.assertEqual(pair["valid_pair_run_count"], 6)
        self.assertEqual(pair["case_differences"][0]["repeat_pair_count"], 2)
        self.assertEqual(pair["ci_95"]["source_count"], 2)
        self.assertEqual(pair["ci_95"]["iterations"], 500)
        self.assertEqual(pair["ci_95"]["seed"], 0)
        self.assertIsNone(pair["ci_reason"])
        second = find_comparison(summarize_records(records), model="m1", framework="f1")
        self.assertEqual(pair["ci_95"], second["ci_95"])


class RescoreTests(unittest.TestCase):
    def test_explicit_data_exclusion_with_missing_reference_does_not_block_rescore(self):
        manifest = {"schema_version": 1, "dataset": "NEXTOR", "dataset_version": "v1", "cases": [
            {"case_id": "ok", "reference": 1, "scoring": {"name": "objective_relative"}},
            {"case_id": "excluded", "reference": None, "scoring": {},
             "review_status": "excluded_missing_critical_definition"}]}
        self.assertEqual(len(rescore_records([make_record("ok")], manifest)), 3)

    def test_rescore_preserves_approved_zero_and_spring_protocols(self):
        manifest = {"schema_version": 1, "dataset": "NEXTOR", "dataset_version": "v1", "cases": [
            {"case_id": "zero", "reference": 0, "scoring": {"name": "objective_relative"}},
            {"case_id": "NEXTOR:11", "reference": [1.96564, 9.0],
             "scoring": {"name": "nextor_spring_legacy"}},
        ]}
        rows = [make_record("zero", prediction=1e-5), make_record("NEXTOR:11", prediction=9.0)]
        before = copy.deepcopy((rows, manifest))
        rescored = rescore_records(rows, manifest)
        self.assertEqual(len(rescored), 6)
        self.assertTrue(all(row["score"]["correct"] for row in rescored))
        self.assertEqual(rescored[3]["score"]["matched_references"][0]["reference_kind"], "effective_coils")
        self.assertEqual(rescored[3]["scoring"]["name"], "nextor_spring_legacy")
        self.assertEqual((rows, manifest), before)

    def setUp(self):
        self.manifest = {
            "schema_version": 1,
            "dataset": "NEXTOR",
            "dataset_version": "v1",
            "cases": [
                {"case_id": "c1", "reference": 100, "scoring": {"name": "nextor_relative"}},
            ],
        }

    def test_threshold_sweep_is_monotone_and_does_not_mutate_input(self):
        records = [
            make_record("c1", version="v1", prediction=100.5),
            make_record("c1", version="v1", prediction=100.0005),
            make_record("c1", version="v1", prediction=101),
            make_record("c1", version="v1", prediction=100.05),
        ]
        original = copy.deepcopy(records)
        rescored = rescore_records(records, self.manifest)
        self.assertEqual(records, original)
        self.assertEqual(len(rescored), 12)
        correct_counts = [
            sum(item["score"]["correct"] for item in rescored[offset::3])
            for offset in range(3)
        ]
        self.assertEqual(correct_counts, [4, 2, 1])
        self.assertEqual([item["rescore_threshold"] for item in rescored[:3]], [0.01, 0.001, 0.0001])

        paired = rescore_records(
            [make_record("c1", "native", version="v1", prediction=100.5),
             make_record("c1", "ned", version="v1", prediction=100.5)],
            self.manifest,
        )
        report = summarize_records(paired)
        self.assertEqual(
            {group["rescore_threshold"] for group in report["groups"]},
            {0.01, 0.001, 0.0001},
        )
        self.assertEqual(
            {pair["rescore_threshold"] for pair in report["comparisons"]},
            {0.01, 0.001, 0.0001},
        )
        self.assertEqual(len(report["comparisons"]), 3)

    def test_missing_prediction_remains_false_and_exclusion_is_marked(self):
        records = [
            make_record("c1", version="v1", prediction=...),
            make_record("c1", version="v1", prediction=100),
        ]
        rescored = rescore_records(records, self.manifest, thresholds=(0.01,), exclude_case_ids=("c1",))
        self.assertTrue(all(item["excluded"] for item in rescored))
        report = summarize_records(rescored)
        self.assertEqual(report["overall"]["run_count"], 0)

        rescored = rescore_records(records[:1], self.manifest, thresholds=(0.01,))
        self.assertFalse(rescored[0]["score"]["correct"])
        self.assertEqual(rescored[0]["score"]["reason"], "missing_prediction")

    def test_version_mismatch_and_wrong_protocol_are_rejected(self):
        with self.assertRaises(ValueError):
            rescore_records([make_record("c1", version="v3")], self.manifest, thresholds=(0.01,))
        wrong_protocol = copy.deepcopy(self.manifest)
        wrong_protocol["cases"][0]["scoring"]["name"] = "optibench_absolute"
        with self.assertRaises(ValueError):
            rescore_records([make_record("c1", version="v1")], wrong_protocol, thresholds=(0.01,))


if __name__ == "__main__":
    unittest.main()
