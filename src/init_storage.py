"""Generate an ignored local database configuration without printing passwords."""

import argparse
from pathlib import Path
import secrets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=55432)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    path = Path(__file__).resolve().parent.parent / ".env.db.local"
    if path.exists():
        print("Existing .env.db.local retained.")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("NIRS_DB_HOST=127.0.0.1\n"
                     f"NIRS_DB_PORT={args.port}\n"
                     "NIRS_DB_NAME=nirs\nNIRS_DB_USER=nirs\n"
                     f"NIRS_DB_PASSWORD={secrets.token_urlsafe(36)}\n"
                     "NIRS_ARTIFACT_ROOT=.local/artifacts\n")
    print("Created .env.db.local with a random password; configuration is excluded from Git.")


if __name__ == "__main__":
    main()
