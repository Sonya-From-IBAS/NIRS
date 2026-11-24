"""CLI for PostgreSQL migrations, imports, artifact checks and backups."""

import argparse
import json
from pathlib import Path

from storage import ROOT, Storage


def save(path: Path | None, result: dict) -> None:
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate")
    import_csv = commands.add_parser("import-csv")
    import_csv.add_argument("path", type=Path)
    import_csv.add_argument("--source-id")
    import_csv.add_argument("--dataset-name")
    import_csv.add_argument("--report", type=Path)
    stage1 = commands.add_parser("import-stage1")
    stage1.add_argument("archive", type=Path)
    stage1.add_argument("--dataset", required=True)
    stage1.add_argument("--name", required=True)
    stage1.add_argument("--report", type=Path)
    commands.add_parser("check-artifacts")
    backup = commands.add_parser("backup")
    backup.add_argument("destination", type=Path)
    restore = commands.add_parser("restore-check")
    restore.add_argument("backup", type=Path)
    restore.add_argument("--artifact-root", type=Path, required=True)
    restore.add_argument("--report", type=Path)
    args = parser.parse_args()
    store = Storage()
    report = getattr(args, "report", None)
    if args.command == "migrate":
        result = {"applied": store.migrate()}
    elif args.command == "import-csv":
        result = store.import_csv(args.path, source_id=args.source_id)
        artifact = store.register_artifact(args.path)
        with store.write() as connection:
            connection.execute("INSERT INTO nirs.source_artifacts VALUES (%s,%s,'input_csv') ON CONFLICT DO NOTHING",
                               (result["source_version_id"], artifact))
        if args.dataset_name:
            result["dataset_version_id"] = store.create_dataset(result["source_version_id"], args.dataset_name)
    elif args.command == "import-stage1":
        from storage_experiments import import_stage1
        result = import_stage1(store, args.archive.resolve(), args.dataset, args.name)
    elif args.command == "check-artifacts":
        result = store.verify_artifacts()
        if result["failures"]:
            save(report, result)
            raise SystemExit(1)
    elif args.command == "backup":
        from storage_backup import backup
        manifest = backup(store, args.destination)
        result = {"dump_sha256": manifest["dump_sha256"], "ready_artifacts": len(manifest["artifacts"]),
                  "table_count": len(manifest["database_fingerprints"])}
    else:
        from storage_backup import restore_check
        result = restore_check(store, args.backup, args.artifact_root)
    save(report, result)


if __name__ == "__main__":
    main()
