"""PostgreSQL storage, immutable snapshots and a checksum-based artifact registry."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from collections import defaultdict

from dotenv import dotenv_values
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from data_contract import CONTEXT_FIELDS, validate_observation

ROOT = Path(__file__).resolve().parent.parent
WRITE_LOCK = 6407633103


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def database_config() -> dict:
    local = dotenv_values(ROOT / ".env.db.local")
    def value(key, default=None):
        return os.environ.get(key, local.get(key, default))
    password = value("NIRS_DB_PASSWORD")
    if not password:
        raise ValueError("Set NIRS_DB_PASSWORD in .env.db.local or the environment")
    return {"host": value("NIRS_DB_HOST", "127.0.0.1"),
            "port": int(value("NIRS_DB_PORT", "55432")),
            "dbname": value("NIRS_DB_NAME", "nirs"),
            "user": value("NIRS_DB_USER", "nirs"), "password": password,
            "connect_timeout": 10}


def unknown_context() -> dict:
    return {field: {"availability": "unknown", "value": None, "evidence_ref": None,
                    "available_before_decision": False} for field in CONTEXT_FIELDS}


def legacy_record(row: dict, index: int, source_id: str, version_id: str, raw_hash: str) -> dict:
    observation_id = "obs_" + digest(f"{version_id}:{index}".encode())
    annotation_id = observation_id + "_a1"
    source_name = row.get("source") or source_id
    annotation = {
        "annotation_id": annotation_id, "revision": 1, "supersedes": None,
        "instruction_version": "1.0.0", "status": "needs_review", "target_label": None,
        "threat_tags": [], "representation_tags": [], "exfiltration_channels": [],
        "reason": "Legacy source label retained separately; target and SQL unit require source audit.",
        "evidence": [{"kind": "source_label", "reference": f"{version_id}:{index}",
                      "statement": "Imported original label without treating it as verified ground truth."}],
        "reviewer": {"kind": "unreviewed", "reviewer_id": "legacy-import-v1"},
    }
    return {
        "schema_version": "1.0.0", "observation_id": observation_id,
        "sql": {"text": row.get("query", row.get("Query")), "unit_kind": "unknown"},
        "context": unknown_context(),
        "source": {"source_id": source_id, "source_version": version_id,
                   "source_record_id": str(index), "source_file_sha256": raw_hash,
                   "origin": "llm_generated" if source_name.startswith("generated_") else "public",
                   "purpose": "research",
                   "source_label": {"label": row.get("label", row.get("Label")),
                                    "label_name": row.get("label_name")},
                   "generator": None},
        "annotations": [annotation], "current_annotation_id": annotation_id,
        "grouping": {"group_id": None, "scenario_id": None, "template_id": None,
                     "campaign_id": None, "parent_observation_ids": []},
        "execution": {"outcome": "unknown", "evidence_ref": None},
    }


class Storage:
    def __init__(self, *, dbname: str | None = None, artifact_root: Path | None = None):
        self.config = database_config()
        if dbname is not None:
            self.config["dbname"] = dbname
        local = dotenv_values(ROOT / ".env.db.local")
        configured = os.environ.get("NIRS_ARTIFACT_ROOT", local.get("NIRS_ARTIFACT_ROOT", ".local/artifacts"))
        self.artifact_root = (artifact_root or ROOT / configured).resolve()

    def connect(self, *, autocommit=False):
        return psycopg.connect(**self.config, row_factory=dict_row, autocommit=autocommit)

    @contextmanager
    def write(self):
        with self.connect() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(%s)", (WRITE_LOCK,))
            yield connection

    def migrate(self) -> list[str]:
        applied = []
        with self.write() as connection:
            connection.execute("CREATE SCHEMA IF NOT EXISTS nirs")
            connection.execute("""CREATE TABLE IF NOT EXISTS nirs.schema_migrations (
                version text PRIMARY KEY, checksum text NOT NULL,
                applied_at timestamptz NOT NULL DEFAULT now())""")
            known = {row["version"]: row["checksum"] for row in connection.execute(
                "SELECT version, checksum FROM nirs.schema_migrations")}
            for path in sorted((ROOT / "migrations").glob("*.sql")):
                content = path.read_bytes().replace(b"\r\n", b"\n")
                checksum = digest(content)
                if path.stem in known:
                    if known[path.stem] != checksum:
                        raise ValueError(f"Applied migration changed: {path.name}")
                    continue
                connection.execute(content.decode("utf-8"), prepare=False)
                connection.execute("INSERT INTO nirs.schema_migrations(version,checksum) VALUES (%s,%s)",
                                   (path.stem, checksum))
                applied.append(path.stem)
        return applied

    @staticmethod
    def _copy(connection, table: str, columns: list[str], rows: list[tuple]) -> None:
        if not rows:
            return
        statement = sql.SQL("COPY nirs.{} ({}) FROM STDIN").format(
            sql.Identifier(table), sql.SQL(",").join(map(sql.Identifier, columns)))
        with connection.cursor().copy(statement) as writer:
            for row in rows:
                writer.write_row(row)

    def _store_batch(self, connection, batch: list[tuple[dict, dict, str, int]], version_id: str) -> None:
        observations, annotations, occurrences = [], [], []
        for record, raw, source_name, ordinal in batch:
            oid = record["observation_id"]
            observations.append((oid, record["sql"]["text"], digest(record["sql"]["text"].encode()),
                record["sql"]["unit_kind"], Jsonb(record["context"]), Jsonb(record["grouping"]),
                Jsonb(record["execution"]), record["source"]["purpose"], record["current_annotation_id"]))
            for annotation in record["annotations"]:
                annotations.append((annotation["annotation_id"], oid, annotation["revision"],
                    annotation["supersedes"], annotation["status"], annotation["target_label"], Jsonb(annotation)))
            occurrences.append((version_id, record["source"]["source_record_id"], oid, source_name,
                Jsonb(record["source"]["source_label"]), Jsonb(record["source"]), Jsonb(raw), ordinal))
        self._copy(connection, "observations", ["observation_id", "sql_text", "sql_sha256", "unit_kind",
            "context_json", "grouping_json", "execution_json", "purpose", "current_annotation_id"], observations)
        self._copy(connection, "annotations", ["annotation_id", "observation_id", "revision", "supersedes",
            "status", "target_label", "data"], annotations)
        self._copy(connection, "occurrences", ["source_version_id", "record_id", "observation_id", "source_name",
            "source_label", "source_json", "raw_record", "ordinal"], occurrences)

    def import_csv(self, path: Path, *, source_id: str | None = None, batch_size: int = 1000) -> dict:
        path = path.resolve()
        raw = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        source_id = source_id or "legacy:" + relative
        raw_hash = digest(raw)
        version_id = "src_" + digest((source_id + ":" + raw_hash).encode())
        job_id = uuid.uuid4().hex
        with self.write() as connection:
            connection.execute("INSERT INTO nirs.import_jobs(job_id,source_id,status,details) VALUES (%s,%s,'running',%s)",
                (job_id, source_id, Jsonb({"input_sha256": raw_hash, "path": relative})))
        try:
            with self.write() as connection:
                existing = connection.execute("SELECT * FROM nirs.source_versions WHERE source_version_id=%s",
                                              (version_id,)).fetchone()
                if existing:
                    result = {"source_version_id": version_id, "rows": existing["row_count"],
                              "accepted": existing["accepted_count"], "rejected": existing["rejected_count"], "reused": True}
                else:
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        reader = csv.DictReader(stream, strict=True)
                        headers = reader.fieldnames
                        if not headers or len(headers) != len(set(headers)) or not ({"query", "Query"} & set(headers)):
                            raise ValueError("CSV needs unique column names and a query/Query column")
                        connection.execute("INSERT INTO nirs.sources VALUES (%s,%s) ON CONFLICT DO NOTHING",
                            (source_id, Jsonb({"kind": "legacy_csv", "license": None, "url": None,
                                              "verification": "source audit deferred to stage 4"})))
                        # Deferred counts are finalized within this same transaction.
                        connection.execute("""INSERT INTO nirs.source_versions
                            (source_version_id,source_id,file_path,sha256_bytes,sha256_lf,columns_json,row_count,accepted_count,rejected_count)
                            VALUES (%s,%s,%s,%s,%s,%s,0,0,0)""",
                            (version_id, source_id, relative, raw_hash, digest(raw.replace(b"\r\n", b"\n")), Jsonb(headers)))
                        batch, total, accepted, rejected = [], 0, 0, 0
                        for index, row in enumerate(reader):
                            total += 1
                            reason = None
                            if None in row or any(value is None for value in row.values()):
                                reason = "column_count_mismatch"
                            elif any("\x00" in value for value in row.values()):
                                reason = "nul_not_supported_by_postgresql_text"
                            if reason:
                                connection.execute("INSERT INTO nirs.rejected_records VALUES (%s,%s,%s,%s)",
                                    (version_id, str(index), reason, json.dumps(row, ensure_ascii=True)))
                                rejected += 1
                                continue
                            record = legacy_record(row, index, source_id, version_id, raw_hash)
                            if accepted == 0:
                                validate_observation(record)
                            batch.append((record, row, row.get("source") or source_id, index))
                            accepted += 1
                            if len(batch) >= batch_size:
                                self._store_batch(connection, batch, version_id)
                                batch.clear()
                        self._store_batch(connection, batch, version_id)
                        if file_hash(path) != raw_hash:
                            raise ValueError("Input CSV changed during import")
                        connection.execute("""UPDATE nirs.source_versions SET row_count=%s,accepted_count=%s,rejected_count=%s
                            WHERE source_version_id=%s""", (total, accepted, rejected, version_id))
                    result = {"source_version_id": version_id, "rows": total, "accepted": accepted,
                              "rejected": rejected, "reused": False}
                connection.execute("""UPDATE nirs.import_jobs SET source_version_id=%s,status='complete',
                    details=details || %s,finished_at=now() WHERE job_id=%s""",
                    (version_id, Jsonb(result), job_id))
            return result
        except Exception as error:
            with self.write() as connection:
                connection.execute("""UPDATE nirs.import_jobs SET status='failed',finished_at=now(),
                    details=details || %s WHERE job_id=%s""", (Jsonb({"error_type": type(error).__name__}), job_id))
            raise

    def import_records(self, records: list[dict], *, source_id: str, version: str) -> str:
        for record in records:
            validate_observation(record)
        if len({record["observation_id"] for record in records}) != len(records):
            raise ValueError("Observation IDs must be unique within an import")
        raw_hash = digest(canonical(records).encode())
        version_id = "src_" + digest((source_id + ":" + version + ":" + raw_hash).encode())
        with self.write() as connection:
            if connection.execute("SELECT 1 FROM nirs.source_versions WHERE source_version_id=%s", (version_id,)).fetchone():
                return version_id
            connection.execute("INSERT INTO nirs.sources VALUES (%s,%s) ON CONFLICT DO NOTHING",
                               (source_id, Jsonb({"kind": "contract_records", "version": version})))
            connection.execute("""INSERT INTO nirs.source_versions
                (source_version_id,source_id,file_path,sha256_bytes,sha256_lf,columns_json,row_count,accepted_count,rejected_count)
                VALUES (%s,%s,%s,%s,%s,'[]',%s,%s,0)""",
                (version_id, source_id, source_id, raw_hash, raw_hash, len(records), len(records)))
            batch = [(record, record, record["source"]["source_id"], index) for index, record in enumerate(records)]
            self._store_batch(connection, batch, version_id)
            for record in records:
                for kind in ("group_id", "scenario_id", "template_id", "campaign_id"):
                    group = record["grouping"][kind]
                    if group is not None:
                        key = source_id + ":" + kind + ":" + group
                        connection.execute("INSERT INTO nirs.sample_groups VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                                           (key, kind, Jsonb({"original_group_id": group})))
                        connection.execute("INSERT INTO nirs.group_members VALUES (%s,%s) ON CONFLICT DO NOTHING",
                                           (key, record["observation_id"]))
        return version_id

    def create_dataset(self, source_version_id: str, name: str) -> str:
        with self.write() as connection:
            members = connection.execute("""SELECT o.observation_id,o.current_annotation_id,c.record_id,c.ordinal
                FROM nirs.occurrences c JOIN nirs.observations o USING(observation_id)
                WHERE c.source_version_id=%s ORDER BY c.ordinal""", (source_version_id,)).fetchall()
            if not members:
                raise ValueError("Cannot freeze an empty dataset")
            dataset_id = "ds_" + digest(canonical({"source": source_version_id, "members": members}).encode())
            if connection.execute("SELECT 1 FROM nirs.dataset_versions WHERE dataset_version_id=%s", (dataset_id,)).fetchone():
                return dataset_id
            connection.execute("INSERT INTO nirs.dataset_versions(dataset_version_id,name,status,specification) VALUES (%s,%s,'building',%s)",
                               (dataset_id, name, Jsonb({"source_version_id": source_version_id,
                                "annotation_policy": "pinned revisions; legacy targets unresolved", "rows": len(members)})))
            self._copy(connection, "dataset_members", ["dataset_version_id", "observation_id", "annotation_id",
                "source_version_id", "record_id", "ordinal"], [(dataset_id, row["observation_id"],
                row["current_annotation_id"], source_version_id, row["record_id"], row["ordinal"]) for row in members])
            connection.execute("UPDATE nirs.dataset_versions SET status='frozen' WHERE dataset_version_id=%s", (dataset_id,))
        return dataset_id

    def dataset_records(self, dataset_id: str) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("""SELECT o.*,a.data AS annotation,c.source_json
                FROM nirs.dataset_members d JOIN nirs.observations o USING(observation_id)
                JOIN nirs.annotations a ON a.annotation_id=d.annotation_id
                JOIN nirs.occurrences c ON c.source_version_id=d.source_version_id AND c.record_id=d.record_id
                WHERE d.dataset_version_id=%s ORDER BY d.ordinal""", (dataset_id,)).fetchall()
            histories = defaultdict(list)
            for row in connection.execute("""SELECT a.observation_id,a.data
                FROM nirs.dataset_members d JOIN nirs.annotations pinned ON pinned.annotation_id=d.annotation_id
                JOIN nirs.annotations a ON a.observation_id=d.observation_id AND a.revision<=pinned.revision
                WHERE d.dataset_version_id=%s ORDER BY a.observation_id,a.revision""", (dataset_id,)):
                histories[row["observation_id"]].append(row["data"])
            results = []
            for row in rows:
                results.append({"schema_version": "1.0.0", "observation_id": row["observation_id"],
                    "sql": {"text": row["sql_text"], "unit_kind": row["unit_kind"]}, "context": row["context_json"],
                    "source": row["source_json"], "annotations": histories[row["observation_id"]],
                    "current_annotation_id": row["annotation"]["annotation_id"],
                    "grouping": row["grouping_json"], "execution": row["execution_json"]})
            return results

    def append_annotation(self, observation_id: str, annotation: dict) -> None:
        with self.write() as connection:
            row = connection.execute("SELECT * FROM nirs.observations WHERE observation_id=%s FOR UPDATE", (observation_id,)).fetchone()
            if not row:
                raise ValueError("Unknown observation")
            history = [item["data"] for item in connection.execute(
                "SELECT data FROM nirs.annotations WHERE observation_id=%s ORDER BY revision", (observation_id,))]
            source = connection.execute("SELECT source_json FROM nirs.occurrences WHERE observation_id=%s LIMIT 1", (observation_id,)).fetchone()
            record = {"schema_version": "1.0.0", "observation_id": observation_id,
                "sql": {"text": row["sql_text"], "unit_kind": row["unit_kind"]}, "context": row["context_json"],
                "source": source["source_json"], "annotations": history + [annotation],
                "current_annotation_id": annotation["annotation_id"], "grouping": row["grouping_json"], "execution": row["execution_json"]}
            validate_observation(record)
            connection.execute("INSERT INTO nirs.annotations VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (annotation["annotation_id"], observation_id, annotation["revision"], annotation["supersedes"],
                 annotation["status"], annotation["target_label"], Jsonb(annotation)))
            connection.execute("UPDATE nirs.observations SET current_annotation_id=%s WHERE observation_id=%s",
                               (annotation["annotation_id"], observation_id))

    def artifact_path(self, relative: str) -> Path:
        path = (self.artifact_root / relative).resolve()
        if not path.is_relative_to(self.artifact_root) or Path(relative).is_absolute():
            raise ValueError("Artifact path escapes configured root")
        return path

    def register_artifact(self, source: Path) -> str:
        source = source.resolve()
        checksum, size = file_hash(source), source.stat().st_size
        suffix = source.suffix.lower()
        if not suffix or not suffix[1:].isalnum():
            suffix = ".bin"
        relative = f"objects/{checksum[:2]}/{checksum}{suffix}"
        with self.write() as connection:
            existing = connection.execute("SELECT * FROM nirs.artifacts WHERE artifact_id=%s", (checksum,)).fetchone()
            if existing:
                relative = existing["relative_path"]
                if existing["byte_size"] != size:
                    raise ValueError("Artifact hash/size conflict")
            else:
                connection.execute("INSERT INTO nirs.artifacts(artifact_id,relative_path,byte_size,status) VALUES (%s,%s,%s,'pending')",
                                   (checksum, relative, size))
        target = self.artifact_path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_hash(target) != checksum:
                raise ValueError("Stored artifact checksum mismatch")
        else:
            temporary = self.artifact_path(relative + "." + uuid.uuid4().hex + ".tmp")
            shutil.copyfile(source, temporary)
            if file_hash(temporary) != checksum or temporary.stat().st_size != size:
                raise ValueError("Artifact source changed while copying; registration remains pending")
            temporary.replace(target)
        with self.write() as connection:
            connection.execute("UPDATE nirs.artifacts SET status='ready' WHERE artifact_id=%s", (checksum,))
        return checksum

    def verify_artifacts(self) -> dict:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM nirs.artifacts ORDER BY artifact_id").fetchall()
        failures, ready, pending = [], 0, 0
        for row in rows:
            if row["status"] != "ready":
                pending += 1
                continue
            path = self.artifact_path(row["relative_path"])
            if not path.is_file() or path.stat().st_size != row["byte_size"] or file_hash(path) != row["artifact_id"]:
                failures.append(row["artifact_id"])
            else:
                ready += 1
        return {"ready_verified": ready, "not_ready": pending, "failures": failures}

    def get_artifact(self, artifact_id: str) -> Path:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM nirs.artifacts WHERE artifact_id=%s AND status='ready'",
                                     (artifact_id,)).fetchone()
        if not row:
            raise ValueError("Artifact is not ready")
        path = self.artifact_path(row["relative_path"])
        if not path.is_file() or path.stat().st_size != row["byte_size"] or file_hash(path) != artifact_id:
            raise ValueError("Artifact is missing or changed")
        return path

    def fingerprints(self, connection=None) -> dict:
        if connection is None:
            with self.connect() as opened:
                return self.fingerprints(opened)
        result = {}
        tables = connection.execute("SELECT tablename FROM pg_tables WHERE schemaname='nirs' ORDER BY tablename").fetchall()
        for item in tables:
            rows = connection.execute(sql.SQL("SELECT to_jsonb(t) AS value FROM nirs.{} t").format(sql.Identifier(item["tablename"])))
            serialized = sorted(canonical(row["value"]) for row in rows)
            result[item["tablename"]] = {"rows": len(serialized), "sha256": digest("\n".join(serialized).encode())}
        return result
