"""Download immutable, static public SQLStateGuard files; never execute SQL."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "dstsmallbird/SQLStateGuard_Dataset"
COMMIT = "87c7cc7cbdbf27df9eae095b9800be7e27039ca4"
DESTINATION = ROOT / ".local/public_sources/sqlstateguard" / COMMIT


def get_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "NIRS-static-dataset-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def git_blob_hash(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def fetch_sqlstateguard() -> dict:
    tree_url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
    tree = json.loads(get_bytes(tree_url))
    if tree.get("truncated"):
        raise ValueError("Incomplete repository tree")
    entries = [entry for entry in tree["tree"] if entry["type"] == "blob" and (
        entry["path"] in ("LICENSE", "README.md") or entry["path"].endswith(".txt"))]
    if len(entries) != 14:
        raise ValueError("Unexpected pinned release contents")
    DESTINATION.mkdir(parents=True, exist_ok=True)

    def download(entry: dict) -> dict:
        relative = Path(entry["path"])
        target = (DESTINATION / relative).resolve()
        if not target.is_relative_to(DESTINATION.resolve()):
            raise ValueError("Source path escapes destination")
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/" + urllib.parse.quote(entry["path"], safe="/")
        raw = target.read_bytes() if target.exists() else get_bytes(url)
        if len(raw) != entry["size"] or git_blob_hash(raw) != entry["sha"]:
            raise ValueError(f"Pinned Git blob differs: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(raw)
            temporary.replace(target)
        print(f"Verified {entry['path']}: {len(raw)} bytes", flush=True)
        return {"path": entry["path"], "url": url, "bytes": len(raw),
                "git_blob_sha1": entry["sha"], "sha256_bytes": hashlib.sha256(raw).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, entries))
    result = {"source": REPOSITORY, "commit": COMMIT, "tree_api": tree_url, "files": files,
              "sql_executed": False, "downloaded_generators_or_attack_tools": False}
    (DESTINATION / "download_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)
                                                       + "\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    result = fetch_sqlstateguard()
    print(json.dumps({"verified_files": len(result["files"]),
                      "bytes": sum(item["bytes"] for item in result["files"])}, indent=2))
