"""Local versioned cases and corpus checks. Expected answers never enter application inputs."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.db.models.article_chunk import EMBED_DIM, ArticleChunk
from app.db.session import get_worker_db_context
from app.repository import article as article_repo
from eval.paths import FIXTURES

ROOT = Path(__file__).resolve().parent
CASES = FIXTURES / "e2e_cases_v1.jsonl"
OUTPUT = ROOT.parent / ".agent_data" / "e2e-evaluation"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


class CaseInput(BaseModel):
    questions: list[str] = Field(default_factory=list)


class Reference(BaseModel):
    requirement: str
    source_key: str
    quote: str


class Expected(BaseModel):
    terminations: list[str] = Field(default_factory=list)
    requirements: list[Reference] = Field(default_factory=list)
    manual_checks: list[str] = Field(default_factory=list)
    must_not_claim: list[str] = Field(default_factory=list)
    negative_control: bool = False
    termination_options: list[list[str]] | None = Field(default=None, exclude_if=lambda value: value is None)
    missing_information: str | None = Field(default=None, exclude_if=lambda value: value is None)


class CaseMetadata(BaseModel):
    case_id: str = Field(pattern=r"^(?:B-(?:0[1-5]|[1-9][0-9]{2})|G-0[1-5])$")
    kind: Literal[
        "single",
        "compare",
        "followup_artifact",
        "followup",
        "memory",
        "insufficient",
        "boundary",
        "missing_goal",
        "empty",
        "tool_error",
        "budget",
        "recovery",
    ]
    mode: Literal["real_corpus", "fixture"]
    approval: Literal["draft", "approved"] = "draft"
    version: str = "v1"
    corpus_version: str | None = None
    source_keys: list[str] = Field(default_factory=list)
    split: Literal["development", "holdout"] | None = Field(default=None, exclude_if=lambda value: value is None)
    requires_artifact: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    search_scope: Literal["snapshot"] | None = Field(default=None, exclude_if=lambda value: value is None)

    @property
    def exports_artifact(self) -> bool:
        return self.kind == "followup_artifact" or self.requires_artifact is True


class E2ECase(BaseModel):
    input: CaseInput
    expected_output: Expected
    metadata: CaseMetadata


class Source(BaseModel):
    key: str
    article_id: int
    url: str
    title: str
    published_at: str | None = None
    body: str
    body_sha256: str
    embedding_done: bool
    chunks: int
    dimensions: list[int]


class Corpus(BaseModel):
    version: str
    sources: list[Source]


def corpus_version(sources: list[Source]) -> str:
    return fingerprint(sorted((row.url, row.title, row.published_at or "", row.body_sha256) for row in sources))


def load_cases(path: Path = CASES) -> list[E2ECase]:
    cases = [
        E2ECase.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = [case.metadata.case_id for case in cases]
    if len(ids) != len(set(ids)) or not cases:
        raise ValueError("Empty dataset or duplicate case IDs")
    for case in cases:
        if case.metadata.case_id.startswith("B-") != (case.metadata.mode == "real_corpus"):
            raise ValueError("Business and fixture case labels must not be mixed")
        fixture_kind = case.metadata.kind in {"missing_goal", "empty", "tool_error", "budget", "recovery"}
        if fixture_kind != (case.metadata.mode == "fixture"):
            raise ValueError("Business and fixture scenario kinds must not be mixed")
    return cases


async def read_corpus() -> Corpus:
    """Read-only bounded snapshot, never recrawl, index or alter articles."""
    async with get_worker_db_context() as db:
        articles, total = await article_repo.list_articles(db, limit=100)
        if total > 100:
            raise ValueError("Select a separately reviewed corpus scope before evaluating more than 100 articles")
        dimensions = await db.execute(
            select(
                ArticleChunk.article_id,
                func.count(),
                func.min(func.vector_dims(ArticleChunk.embedding)),
                func.max(func.vector_dims(ArticleChunk.embedding)),
            ).group_by(ArticleChunk.article_id)
        )
        indexed = {row[0]: (row[1], [row[2], row[3]]) for row in dimensions}
        sources = []
        for article in articles:
            body = article.body_text or ""
            count, dims = indexed.get(article.id, (0, []))
            sources.append(
                Source(
                    key="S-" + hashlib.sha256(article.url.encode()).hexdigest()[:12],
                    article_id=article.id,
                    url=article.url,
                    title=article.title,
                    published_at=str(article.publish_date) if article.publish_date else None,
                    body=body,
                    body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                    embedding_done=article.embedding_done,
                    chunks=count,
                    dimensions=dims,
                )
            )
    return Corpus(version=corpus_version(sources), sources=sources)


def preflight_case(case: E2ECase, snapshot: Corpus, current: Corpus) -> list[str]:
    problems = []
    if case.metadata.approval != "approved":
        problems.append("labels_not_approved")
    if snapshot.version != corpus_version(snapshot.sources) or any(
        hashlib.sha256(row.body.encode()).hexdigest() != row.body_sha256 for row in snapshot.sources
    ):
        problems.append("snapshot_corrupted")
    if case.metadata.corpus_version != snapshot.version or snapshot.version != current.version:
        problems.append("corpus_version_mismatch")
    by_key = {row.key: row for row in snapshot.sources}
    live = {row.url: row for row in current.sources}
    insufficient = case.metadata.kind == "insufficient"
    if insufficient and (
        case.metadata.search_scope != "snapshot"
        or not (case.expected_output.missing_information or "").strip()
        or not any(item.strip() for item in case.expected_output.must_not_claim)
        or "missing_information_disclosed" not in case.expected_output.manual_checks
    ):
        problems.append("absence_rubric_missing")
    if not insufficient and (not case.metadata.source_keys or not case.expected_output.requirements):
        problems.append("references_missing")
    if set(case.metadata.source_keys) != {row.source_key for row in case.expected_output.requirements}:
        problems.append("references_incomplete")
    if case.metadata.kind == "compare" and len(set(case.metadata.source_keys)) < 2:
        problems.append("comparison_needs_two_sources")
    # A claim that information is absent needs the entire stated search scope indexed.
    scoped_keys = list(by_key) if insufficient else case.metadata.source_keys
    if insufficient and not scoped_keys:
        problems.append("search_scope_empty")
    for key in set(scoped_keys) | set(case.metadata.source_keys):
        source = by_key.get(key)
        found = live.get(source.url) if source else None
        if not source or not found or found.body_sha256 != source.body_sha256:
            problems.append("source_missing_or_changed")
        elif not found.embedding_done or found.chunks < 1 or found.dimensions != [EMBED_DIM, EMBED_DIM]:
            problems.append("source_not_indexed")
    for reference in case.expected_output.requirements:
        source = by_key.get(reference.source_key)
        if (
            reference.source_key not in case.metadata.source_keys
            or not reference.requirement.strip()
            or reference.requirement.startswith("待人工")
            or not reference.quote.strip()
            or not source
            or reference.quote not in source.body
        ):
            problems.append("support_quote_invalid")
    required_questions = 2 if case.metadata.kind in {"followup_artifact", "followup", "memory"} else 1
    if len(case.input.questions) != required_questions or any(
        not text.strip() or "{source_" in text for text in case.input.questions
    ):
        problems.append("questions_unprepared")
    options = case.expected_output.termination_options
    if options is None:
        terminations = [[value] for value in case.expected_output.terminations]
    else:
        terminations = options
        if case.expected_output.terminations:
            problems.append("termination_labels_ambiguous")
    if len(terminations) != required_questions or any(
        not values or any(not value.strip() for value in values) for values in terminations
    ):
        problems.append("termination_labels_incomplete")
    if (case.metadata.exports_artifact and case.metadata.kind not in {"followup", "followup_artifact"}) or (
        case.metadata.kind == "followup_artifact" and case.metadata.requires_artifact is False
    ):
        problems.append("artifact_configuration_invalid")
    if not {"goal_completion", "evidence_support"}.issubset(case.expected_output.manual_checks):
        problems.append("manual_rubric_missing")
    return sorted(set(problems))


def prepare_cases(cases: list[E2ECase], corpus: Corpus) -> list[E2ECase]:
    usable = [row for row in corpus.sources if row.body.strip() and row.embedding_done and row.chunks]
    usable.sort(key=lambda row: ("e 962" not in row.title.lower(), row.url))
    if not usable:
        raise ValueError("No indexed source available; do not fabricate labels")
    selected = list({row.body_sha256: row for row in reversed(usable)}.values())
    selected.sort(key=lambda row: ("e 962" not in row.title.lower(), row.url))
    selected = selected[:2]
    bindings = {"source_a": selected[0]}
    if len(selected) > 1:
        bindings["source_b"] = selected[1]
    prepared = []
    for original in cases:
        case = original.model_copy(deep=True)
        if case.metadata.mode == "real_corpus":
            keys = list(case.metadata.source_keys)
            case.metadata.corpus_version = corpus.version
            case.metadata.approval = "draft"
            case.metadata.source_keys = [bindings[key].key if key in bindings else key for key in keys]
            for i, question in enumerate(case.input.questions):
                for key, source in bindings.items():
                    question = question.replace("{" + key + "_title}", source.title)
                case.input.questions[i] = question
            case.expected_output.requirements = [
                Reference(
                    requirement="待人工核对：从本地原文确认本题应覆盖的结论与限制，不据此自动判真。",
                    source_key=bindings[key].key,
                    quote=bindings[key].body[:300],
                )
                for key in keys
                if key in bindings
            ]
        prepared.append(case)
    return prepared
