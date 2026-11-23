"""Fetch the official BEIR SciFact archive and prepare its complete retrieval data.

Run: uv run python -m eval.datasets.prepare_scifact
No database, embedding, generation, labels modification or corpus sampling.
"""

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
ARCHIVE_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
SOURCE = "https://github.com/beir-cellar/beir#-available-datasets"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / ".agent_data" / "benchmarks" / "scifact"
MAX_BYTES = 64 * 1024 * 1024
MEMBERS = ("corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv")
HF_CORPUS_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
HF_QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
HF_PARQUET = {
    "corpus": "243324b35f03d82bd6d98a5f575966876e86cad7ce16e5333a35b1b793dc4f45",
    "queries": "1c37956c5dc8b810b60302323c24d1a9e79e26411ba8f5ad9d0888642e2a9034",
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(part)
    return hasher.hexdigest()


def download_archive(destination: Path) -> None:
    if destination.exists():
        if digest(destination, "md5") != ARCHIVE_MD5:
            raise ValueError("Existing SciFact archive has an unexpected checksum; preserve it for inspection")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")
    request = urllib.request.Request(URL, headers={"User-Agent": "IntelligenceAnalysis-BenchmarkPreparation/1"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
        length = 0
        while part := response.read(1024 * 1024):
            length += len(part)
            if length > MAX_BYTES:
                raise ValueError("Archive exceeds the expected size budget")
            output.write(part)
    if digest(partial, "md5") != ARCHIVE_MD5:
        raise ValueError("Downloaded archive does not match the official BEIR checksum")
    partial.replace(destination)


def extract_required(archive: Path, destination: Path) -> None:
    """Copy only expected data members; never execute or blindly extract the archive."""
    with zipfile.ZipFile(archive) as zipped:
        if sum(entry.file_size for entry in zipped.infolist()) > MAX_BYTES:
            raise ValueError("Expanded archive exceeds its size budget")
        for entry in zipped.infolist():
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in entry.filename:
                raise ValueError("Unsafe path in dataset archive")
        for name in MEMBERS:
            source = f"scifact/{name}"
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(source) as incoming, target.open("wb") as output:
                shutil.copyfileobj(incoming, output)


def download_file(url: str, path: Path, *, sha256: str | None = None) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as incoming, partial.open("wb") as output:
            total = 0
            while part := incoming.read(1024 * 1024):
                total += len(part)
                if total > MAX_BYTES:
                    raise ValueError("Dataset file exceeds its size budget")
                output.write(part)
        if sha256 and digest(partial) != sha256:
            raise ValueError("Official Hugging Face LFS SHA256 mismatch")
        partial.replace(path)
    if sha256 and digest(path) != sha256:
        raise ValueError("Cached official data has an unexpected SHA256")


def from_official_huggingface(raw: Path, extracted: Path) -> dict[str, Any]:
    """Official BEIR distribution fallback, pinned independently of mutable main."""
    import pyarrow.parquet as pq

    extracted.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, sha256 in HF_PARQUET.items():
        filename = f"{name}/{name}-00000-of-00001.parquet"
        url = f"https://huggingface.co/datasets/BeIR/scifact/resolve/{HF_CORPUS_REVISION}/{filename}?download=true"
        target = raw / "huggingface" / f"{name}.parquet"
        download_file(url, target, sha256=sha256)
        write_jsonl(extracted / f"{name}.jsonl", pq.read_table(target).to_pylist())
        hashes[target.name] = digest(target)
    for split in ["train", "test"]:
        filename = f"{split}.tsv"
        url = f"https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/{HF_QRELS_REVISION}/{filename}"
        target = raw / "huggingface" / filename
        download_file(url, target)
        (extracted / "qrels").mkdir(exist_ok=True)
        shutil.copyfile(target, extracted / "qrels" / filename)
        hashes[filename] = digest(target)
    source = {
        "distribution": "official_beir_huggingface",
        "corpus_revision": HF_CORPUS_REVISION,
        "qrels_revision": HF_QRELS_REVISION,
        "raw_sha256": hashes,
        "integrity": "pinned repository revisions; parquet LFS SHA256 verified",
    }
    (raw / "huggingface" / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    return source


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def qrels(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["query-id", "corpus-id", "score"]:
            raise ValueError("Unexpected qrels columns")
        return [
            {"query-id": row["query-id"], "corpus-id": row["corpus-id"], "score": int(row["score"])} for row in reader
        ]


def validate(
    corpus: list[dict[str, Any]], queries: list[dict[str, Any]], labels: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    corpus_ids = {row["_id"] for row in corpus}
    query_by_id = {row["_id"]: row for row in queries}
    if len(corpus_ids) != len(corpus) or len(query_by_id) != len(queries):
        raise ValueError("Duplicate corpus or query IDs")
    if not corpus or any(not isinstance(row["_id"], str) or not row.get("text", "").strip() for row in corpus):
        raise ValueError("Empty corpus or invalid source text")
    if any(not isinstance(row["_id"], str) or not row.get("text", "").strip() for row in queries):
        raise ValueError("Invalid query text")
    split_ids: dict[str, set[str]] = {}
    selected: dict[str, list[dict[str, Any]]] = {}
    for split, rows in labels.items():
        pairs = {(row["query-id"], row["corpus-id"]) for row in rows}
        if not rows or len(pairs) != len(rows):
            raise ValueError("Empty qrels or duplicate judged pairs")
        if any(row["query-id"] not in query_by_id or row["corpus-id"] not in corpus_ids for row in rows):
            raise ValueError("Qrels reference a missing query or corpus document")
        if any(row["score"] < 0 for row in rows):
            raise ValueError("Invalid negative relevance score")
        split_ids[split] = {row["query-id"] for row in rows}
        if any(not any(r["score"] > 0 and r["query-id"] == q for r in rows) for q in split_ids[split]):
            raise ValueError("Query has no positive judgment")
        selected[split] = [row for row in queries if row["_id"] in split_ids[split]]
    if split_ids.get("train", set()) & split_ids.get("test", set()):
        raise ValueError("Train and test query IDs overlap")
    return selected


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    root = root.resolve()
    archive = root / "raw" / "scifact.zip"
    extracted = root / "raw" / "scifact"
    if (root / "raw" / "huggingface" / "source.json").exists():
        distribution = from_official_huggingface(root / "raw", extracted)
    else:
        try:
            download_archive(archive)
        except (urllib.error.URLError, TimeoutError):
            distribution = from_official_huggingface(root / "raw", extracted)
        else:
            extract_required(archive, extracted)
            distribution = {
                "distribution": "official_beir_zip",
                "archive_md5": digest(archive, "md5"),
                "archive_sha256": digest(archive),
            }
    corpus, queries = records(extracted / "corpus.jsonl"), records(extracted / "queries.jsonl")
    labels = {split: qrels(extracted / f"qrels/{split}.tsv") for split in ["train", "test"]}
    split_queries = validate(corpus, queries, labels)
    if len(corpus) != 5183 or len(split_queries["test"]) != 300:
        raise ValueError("Dataset counts differ from the reviewed SciFact release")
    output = root / "prepared"
    staging = root / f"prepared-{uuid4().hex}"
    staging.mkdir(parents=True)
    write_jsonl(staging / "corpus.jsonl", corpus)
    for split in ["train", "test"]:
        write_jsonl(staging / f"queries.{split}.jsonl", split_queries[split])
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer, fieldnames=["query-id", "corpus-id", "score"], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(labels[split])
        (staging / f"qrels.{split}.tsv").write_text(buffer.getvalue(), encoding="utf-8", newline="\n")
    hashes = {path.name: digest(path) for path in sorted(staging.iterdir())}
    if output.exists():
        if any(not (output / name).exists() or digest(output / name) != value for name, value in hashes.items()):
            raise ValueError("Existing prepared data differs; refusing to overwrite a benchmark snapshot")
        for path in staging.iterdir():
            path.unlink()
        staging.rmdir()
    else:
        staging.replace(output)
    manifest = {
        "dataset": "BEIR/SciFact",
        "preparation_version": "v1",
        "source": SOURCE,
        "download_url": URL,
        "input_distribution": distribution,
        "license": "CC-BY-SA-4.0",
        "license_source": "https://huggingface.co/datasets/BeIR/scifact",
        "counts": {
            "corpus": len(corpus),
            "all_queries": len(queries),
            **{f"{split}_queries": len(split_queries[split]) for split in labels},
            **{f"{split}_qrels": len(rows) for split, rows in labels.items()},
        },
        "prepared_sha256": hashes,
        "complete_candidate_corpus": True,
        "sampling": False,
        "database_imported": False,
        "embedding_index_built": False,
        "evaluation_run": False,
        "test_policy": "Keep official labels unchanged. Do not tune on test results.",
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if {k: v for k, v in prior.items() if k != "prepared_at_utc"} != manifest:
            raise ValueError("Existing manifest does not match prepared data")
    else:
        manifest["prepared_at_utc"] = datetime.now(UTC).isoformat()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dataset": "SciFact",
        "root": str(root),
        **manifest["counts"],
        "distribution": distribution["distribution"],
        "download_verified": True,
        "complete_candidate_corpus": True,
        "evaluation_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    sys.stdout.write(json.dumps(prepare(args.output_dir), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
