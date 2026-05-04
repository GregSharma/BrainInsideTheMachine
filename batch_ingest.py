#!/usr/bin/env python3
"""Batch ingest all BITM sources into the KG.

Runs pipeline.run sequentially on:
  1. All cleaned web exports (Chats/clean/*.md)
  2. All Vega transcripts (vega_md/*.md)

All sources use --transcript mode for timestamp extraction.
One source at a time at full RPM (they share the proxy budget).

Usage:
    python3 batch_ingest.py                    # ingest all, passes 1 2 4
    python3 batch_ingest.py --passes 1 2 4     # explicit passes
    python3 batch_ingest.py --wipe             # wipe Neo4j first
    python3 batch_ingest.py --dry-run          # show what would run
"""

import argparse
import subprocess
import sys
from pathlib import Path

BITM_DIR = Path(__file__).parent
PIPELINE_DIR = Path.home() / "Desktop/Projects/llm-graph-builder"

WEB_DIR = BITM_DIR / "Chats" / "clean"
VEGA_DIR = BITM_DIR / "vega_md"

DOMAIN = "bitm"
RPM = 1000
MAX_CONCURRENT = 64


def find_sources() -> list[tuple[str, Path]]:
    """Find all sources to ingest, return (title, path) pairs."""
    sources = []

    # Web exports
    for f in sorted(WEB_DIR.glob("*.md")):
        name = f.stem.replace(".clean", "")
        sources.append((f"Claude Web: {name}", f))

    # Vega transcripts
    for f in sorted(VEGA_DIR.glob("*.md")):
        # Use first 8 chars of UUID as short name
        short = f.stem[:8] if len(f.stem) > 8 else f.stem
        sources.append((f"Vega: {short}", f))

    return sources


def run_pipeline(title: str, file_path: Path, passes: list[int], dry_run: bool) -> bool:
    """Run pipeline.run for one source. Returns True on success."""
    pass_str = " ".join(str(p) for p in passes)
    cmd = [
        sys.executable, "-m", "pipeline.run",
        "--file", str(file_path),
        "--title", title,
        "--domain", DOMAIN,
        "--transcript",
        "--rpm", str(RPM),
        "--max-concurrent", str(MAX_CONCURRENT),
        "--passes", *[str(p) for p in passes],
    ]

    if dry_run:
        print(f"  [dry-run] {title}")
        print(f"    {' '.join(cmd)}")
        return True

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  {file_path.name} ({file_path.stat().st_size // 1024}KB)")
    print(f"  Passes: {pass_str}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PIPELINE_DIR))
    if result.returncode != 0:
        print(f"  [FAILED] {title} — exit code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch ingest BITM sources into KG")
    parser.add_argument("--passes", nargs="+", type=int, default=[1, 2, 4],
                        help="Pipeline passes to run (default: 1 2 4)")
    parser.add_argument("--wipe", action="store_true",
                        help="Wipe Neo4j before ingesting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    args = parser.parse_args()

    sources = find_sources()
    if not sources:
        print("No sources found. Run prepare_sources.sh first.")
        sys.exit(1)

    print(f"Found {len(sources)} sources:")
    for title, path in sources:
        print(f"  {title:40s}  {path.stat().st_size // 1024:>6}KB")

    if args.wipe and not args.dry_run:
        print("\nWiping Neo4j...")
        subprocess.run([
            sys.executable, "-c",
            "from neo4j import GraphDatabase; "
            "d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword')); "
            "d.execute_query('MATCH (n) DETACH DELETE n'); "
            "d.close(); "
            "print('  Wiped.')"
        ], cwd=str(PIPELINE_DIR))

    succeeded = 0
    failed = 0
    for title, path in sources:
        ok = run_pipeline(title, path, args.passes, args.dry_run)
        if ok:
            succeeded += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Batch complete: {succeeded} succeeded, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
