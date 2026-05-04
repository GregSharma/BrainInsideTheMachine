#!/bin/bash
# Prepare all BITM sources for KG ingestion.
# 1. Clean web exports (strip base64, flag missing attachments)
# 2. Convert Vega JSONL transcripts to markdown (verbosity 3)
#
# Output:
#   Chats/raw/     — move raw exports here
#   Chats/clean/   — cleaned web exports
#   vega_md/       — converted Vega transcripts

set -euo pipefail
cd "$(dirname "$0")"

CONVERTER=~/Desktop/Projects/Claude_Transcripts_to_Md
CLEANER=./clean_web_exports.py
JSONL_DIR=~/.claude/projects/-home-greg-Desktop-Projects-BrainInsideTheMachine

# ── Web exports ──────────────────────────────────────────
echo "=== Cleaning web exports ==="
mkdir -p Chats/raw Chats/clean

# Move raw exports to raw/ if not already there
for f in Chats/Web-*.md; do
    [ -f "$f" ] || continue
    # Don't move .clean.md files
    [[ "$f" == *.clean.md ]] && continue
    base=$(basename "$f")
    [ -f "Chats/raw/$base" ] || mv "$f" "Chats/raw/$base"
done

# Clean all raw exports -> clean/
if ls Chats/raw/Web-*.md 1>/dev/null 2>&1; then
    python3 "$CLEANER" --dir Chats/raw/ --out Chats/clean/
else
    echo "  No raw web exports found in Chats/raw/"
fi

# ── Vega transcripts ────────────────────────────────────
echo ""
echo "=== Converting Vega JSONL transcripts ==="
mkdir -p vega_md

# Convert all BITM project sessions at verbosity 3
cd "$CONVERTER"
python3 cli.py -p BrainInsideTheMachine -v 3 -o /home/greg/Desktop/Projects/BrainInsideTheMachine/vega_md
cd -

# Flatten — the converter creates subdirs per project, move md files up
find vega_md -name "*.md" -mindepth 2 -exec mv {} vega_md/ \;
find vega_md -type d -empty -delete 2>/dev/null || true

VEGA_COUNT=$(ls vega_md/*.md 2>/dev/null | wc -l)
WEB_COUNT=$(ls Chats/clean/*.md 2>/dev/null | wc -l)

echo ""
echo "=== Done ==="
echo "  Web exports cleaned: $WEB_COUNT  (Chats/clean/)"
echo "  Vega transcripts:    $VEGA_COUNT (vega_md/)"
echo ""
echo "Next: python3 batch_ingest.py"
