"""Exercise real outer/inner graphs and the actual SDK, without services or paid calls."""

from collections import Counter
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core import observability as obs
from eval import run_prototype_eval as runner


@pytest.mark.anyio
async def test_synthetic_graph_contracts_and_callback_tree() -> None:
    exporter = InMemorySpanExporter()
    key = f"pk-evaluation-{uuid4()}"
    client = obs.build_client(
        public_key=key, secret_key="test-only", base_url="http://127.0.0.1:1", exporter=exporter, synthetic=True
    )
    try:
        rows = await runner.run_offline(client, key)
        client.flush()
        spans = exporter.get_finished_spans()
        assert len(rows) == 8 and all(row["passed"] for row in rows)
        summary = runner.summarize(rows)
        assert summary["paid_api_calls"] == 0 and summary["cost_usd"] is None
        assert summary["metrics"]["source_recall"]["n"] == 5
        generations = [
            span for span in spans if (span.attributes or {}).get("langfuse.observation.type") == "generation"
        ]
        counts = Counter(f"{span.context.trace_id:032x}" for span in generations)
        # The exact same handler propagates through outer graph, inner loop and summary.
        # Duplicate callbacks would increase this above the actual fixture call count.
        assert len(generations) == sum(row["model_calls"] for row in rows) == 42
        for row in rows:
            for operation in row["operations"]:
                assert counts[operation["trace_id"]] == operation["model_calls"]
        tools = [span for span in spans if (span.attributes or {}).get("langfuse.observation.type") == "tool"]
        assert len(tools) == 8 and all(span.name == "search_rag_tool" for span in tools)
        summaries = [span for span in spans if span.name == "stage.summary"]
        assert len(summaries) == 1
        assert any(span.parent and span.parent.span_id == summaries[0].context.span_id for span in generations)
        followup = next(row for row in rows if row["case_id"] == "follow-up")
        trace_ids = {operation["trace_id"] for operation in followup["operations"]}
        roots = [
            span for span in spans if span.name == "operation.run" and f"{span.context.trace_id:032x}" in trace_ids
        ]
        assert len(roots) == len(trace_ids) == 2 and all(span.parent is None for span in roots)
        assert len({(span.attributes or {})["session.id"] for span in roots}) == 1
        assert len({(span.attributes or {})["langfuse.observation.metadata.operation_id"] for span in roots}) == 2
        assert len({(span.attributes or {})["langfuse.observation.metadata.task_id"] for span in roots}) == 2
    finally:
        obs.shutdown_client(client)


@pytest.mark.anyio
async def test_non_fixture_input_rejected_before_graph_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    build = Mock()
    monkeypatch.setattr(runner, "build_graph", build)
    with pytest.raises(ValueError, match="checked-in synthetic"):
        await runner.run_case({"id": "unapproved", "question": "PRIVATE"})
    build.assert_not_called()


def test_sdk_dataset_experiment_wiring_uses_fixed_inputs_and_excludes_missing_scores() -> None:
    """API contract wiring only; remote project availability is a separate check."""
    client = Mock()
    client.get_dataset.return_value.items = [
        SimpleNamespace(input=case, expected_output=case["expected_termination"], metadata={"fixture_version": "v1"})
        for case in runner.load_cases()
    ]
    runner.publish(client, "test-key")
    assert client.create_dataset_item.call_count == len(runner.load_cases())
    assert [call.kwargs["input"] for call in client.create_dataset_item.call_args_list] == runner.load_cases()
    options: dict[str, Any] = client.get_dataset.return_value.run_experiment.call_args.kwargs
    assert options["max_concurrency"] == 1
    evaluate = options["evaluators"][0]
    scores = evaluate(output={"checks": {"source_recall": None, "citation_id_valid": False}})
    assert len(scores) == 1 and scores[0].name == "citation_id_valid" and scores[0].value == 0.0


def test_remote_dataset_with_other_data_is_rejected_before_sdk_experiment() -> None:
    client = Mock()
    client.get_dataset.return_value.items = [SimpleNamespace(input={"question": "PRIVATE"})]
    with pytest.raises(ValueError, match="fixed synthetic fixtures"):
        runner.publish(client, "test-key")
    client.get_dataset.return_value.run_experiment.assert_not_called()
