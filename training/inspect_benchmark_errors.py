from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    query = "SELECT sample_id, error FROM predictions WHERE error IS NOT NULL"
    parameters: list[object] = []
    if args.dataset:
        query += " AND dataset=?"
        parameters.append(args.dataset)
    query += " ORDER BY completed_at DESC LIMIT ?"
    parameters.append(args.limit)
    for sample_id, error in connection.execute(query, parameters):
        print(f"===== {sample_id} =====\n{error}")
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
