"""Consistent PostgreSQL dump plus the ready files in the artifact registry."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from psycopg import sql

from storage import ROOT, Storage, WRITE_LOCK, file_hash


def docker_client(arguments: list[str], *, output=None, input_file=None) -> None:
    command = ["docker", "compose", "--project-directory", str(ROOT), "--env-file",
        str(ROOT / ".env.db.local"), "-f", str(ROOT / "compose.yaml"), "exec", "-T", "postgres", *arguments]
    subprocess.run(command, stdin=input_file, stdout=output, check=True)


def backup(store: Storage, destination: Path) -> dict:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with store.connect() as connection:
        # All application writers use this transaction-scoped lock, including
        # pending -> ready transitions. pg_dump takes its own consistent snapshot.
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (WRITE_LOCK,))
        artifacts = connection.execute("SELECT * FROM nirs.artifacts WHERE status='ready' ORDER BY artifact_id").fetchall()
        fingerprints = store.fingerprints(connection)
        server = connection.execute("SHOW server_version").fetchone()["server_version"]
        dump = destination / "database.dump"
        with dump.open("xb") as output:
            docker_client(["pg_dump", "-U", store.config["user"], "-d", store.config["dbname"],
                "--format=custom", "--no-owner", "--no-acl"], output=output)
        files = []
        for row in artifacts:
            source = store.artifact_path(row["relative_path"])
            if not source.is_file() or file_hash(source) != row["artifact_id"]:
                raise ValueError("Cannot back up a missing or changed ready artifact")
            target = destination / "artifacts" / row["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if file_hash(target) != row["artifact_id"]:
                raise ValueError("Backup artifact checksum mismatch")
            files.append({"artifact_id": row["artifact_id"], "relative_path": row["relative_path"], "byte_size": row["byte_size"]})
        manifest = {"backup_version": "1.0.0", "created_utc": datetime.now(timezone.utc).isoformat(),
            "database_name": store.config["dbname"], "server_version": server,
            "dump_sha256": file_hash(dump), "database_fingerprints": fingerprints,
            "artifacts": files, "non_ready_artifacts_not_copied": True,
            "consistency": "All storage module writers locked until dump and file copying completed."}
        (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)
                                                  + "\n", encoding="utf-8", newline="\n")
    return manifest


def create_scratch_database(store: Storage, *, prefix="nirs_test_") -> str:
    if prefix not in ("nirs_test_", "nirs_restore_"):
        raise ValueError("Unexpected scratch database prefix")
    name = prefix + uuid.uuid4().hex[:16]
    with store.connect(autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name)))
    return name


def drop_scratch_database(store: Storage, name: str) -> None:
    if not re.fullmatch(r"nirs_(?:test|restore)_[0-9a-f]{12,32}", name) or name == store.config["dbname"]:
        raise ValueError("Refusing to drop a non-scratch database")
    with store.connect(autocommit=True) as connection:
        connection.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def restore_check(store: Storage, directory: Path, artifact_root: Path) -> dict:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(directory / "database.dump") != manifest["dump_sha256"]:
        raise ValueError("Database dump checksum mismatch")
    name = create_scratch_database(store, prefix="nirs_restore_")
    restored = Storage(dbname=name, artifact_root=artifact_root)
    try:
        with (directory / "database.dump").open("rb") as input_file:
            docker_client(["pg_restore", "-U", store.config["user"], "-d", name,
                           "--no-owner", "--no-acl", "--exit-on-error"], input_file=input_file)
        restored.artifact_root.mkdir(parents=True, exist_ok=False)
        for item in manifest["artifacts"]:
            source = directory / "artifacts" / item["relative_path"]
            if not source.resolve().is_relative_to((directory / "artifacts").resolve()):
                raise ValueError("Backup path escapes its root")
            if not source.is_file() or source.stat().st_size != item["byte_size"] or file_hash(source) != item["artifact_id"]:
                raise ValueError("Backup file is missing or changed")
            target = restored.artifact_path(item["relative_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        actual = restored.fingerprints()
        if actual != manifest["database_fingerprints"]:
            raise ValueError("Restored table contents differ from the backup snapshot")
        files = restored.verify_artifacts()
        if files["failures"]:
            raise ValueError("Restored artifacts differ")
        # Verify the archived historical pipelines on the restored SQL, not just
        # the byte copies. Only locally created, hash-verified joblib files load.
        import joblib
        import numpy as np
        checked_models = 0
        with restored.connect() as connection:
            runs = connection.execute("""SELECT r.run_id,a.artifact_id FROM nirs.experiment_runs r
                JOIN nirs.run_artifacts a ON a.run_id=r.run_id AND a.role=r.configuration->>'model_artifact_role'
                WHERE r.label_space='legacy_five_class' ORDER BY r.run_id""").fetchall()
            for run in runs:
                rows = connection.execute("""SELECT o.sql_text,p.predicted_label FROM nirs.predictions p
                    JOIN nirs.observations o USING(observation_id)
                    JOIN nirs.dataset_members d ON d.dataset_version_id=p.dataset_version_id AND d.observation_id=p.observation_id
                    WHERE p.run_id=%s AND p.protocol='holdout' AND p.status='ok' ORDER BY d.ordinal""",
                    (run["run_id"],)).fetchall()
                model = joblib.load(restored.get_artifact(run["artifact_id"]))
                predicted = model.predict([row["sql_text"] for row in rows])
                np.testing.assert_array_equal(predicted, [row["predicted_label"] for row in rows])
                checked_models += 1
        return {"status": "passed", "tables_checked": len(actual),
                "ready_artifacts_checked": files["ready_verified"],
                "legacy_holdout_models_checked": checked_models,
                "dump_sha256": manifest["dump_sha256"], "database_fingerprints_match": True,
                "scratch_database_dropped_after_check": True}
    finally:
        drop_scratch_database(store, name)
