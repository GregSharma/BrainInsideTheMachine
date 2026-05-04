"""Clean Claude Web chat export markdown files for KG ingestion.

Fixes two export artifacts:
1. Base64-encoded images — replaced with [IMAGE: filename]
2. Empty `> File:` attachment blocks — flagged with [ATTACHMENT MISSING]

Usage:
    python clean_web_exports.py Chats/Web-1.md          # writes Chats/Web-1.clean.md
    python clean_web_exports.py Chats/Web-*.md           # batch
    python clean_web_exports.py --dir Chats/ --out clean/ # dir mode
"""

import re
import sys
from pathlib import Path


# Matches: ![any alt text](data:image/TYPE;base64,DATA)
# DATA can be very long (hundreds of KB)
_BASE64_IMG = re.compile(
    r'!\[([^\]]*)\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=\n]+\)',
    re.DOTALL,
)

# Matches a `> File:` line with nothing (or only whitespace) after the colon.
# The line must be followed by a blank line (typical for empty attachment slots).
_EMPTY_FILE_BLOCK = re.compile(
    r'^> File:\s*$',
    re.MULTILINE,
)

# Matches `> File: /mnt/...` style — these have real paths, leave them alone.
_FILE_WITH_PATH = re.compile(r'^> File:\s+\S', re.MULTILINE)


def clean(text: str) -> tuple[str, dict]:
    """Clean one web export and return (cleaned_text, stats)."""
    stats = {"images_stripped": 0, "attachments_flagged": 0}

    # 1. Strip base64 images
    def _strip_img(m: re.Match) -> str:
        alt = m.group(1) or "image"
        stats["images_stripped"] += 1
        return f"[IMAGE: {alt}]"

    text = _BASE64_IMG.sub(_strip_img, text)

    # 2. Flag empty File: blocks
    def _flag_empty(m: re.Match) -> str:
        stats["attachments_flagged"] += 1
        return "> File: [ATTACHMENT MISSING — content not captured by Chrome exporter]"

    text = _EMPTY_FILE_BLOCK.sub(_flag_empty, text)

    return text, stats


def process_file(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    cleaned, stats = clean(text)
    dst.write_text(cleaned, encoding="utf-8")

    orig_kb = len(text) / 1024
    clean_kb = len(cleaned) / 1024
    saved_kb = orig_kb - clean_kb
    print(
        f"{src.name} → {dst.name}  "
        f"({orig_kb:.0f}KB → {clean_kb:.0f}KB, -{saved_kb:.0f}KB)  "
        f"images={stats['images_stripped']}  "
        f"missing_attachments={stats['attachments_flagged']}"
    )


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        sys.exit(1)

    # Parse --dir / --out flags
    src_files: list[Path] = []
    out_dir: Path | None = None

    i = 0
    while i < len(argv):
        if argv[i] == "--dir":
            i += 1
            src_files = sorted(Path(argv[i]).glob("Web-*.md"))
        elif argv[i] == "--out":
            i += 1
            out_dir = Path(argv[i])
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            src_files.append(Path(argv[i]))
        i += 1

    if not src_files:
        print("No files found.")
        sys.exit(1)

    for src in src_files:
        if not src.exists():
            print(f"SKIP (not found): {src}")
            continue
        if out_dir:
            dst = out_dir / src.name
        else:
            dst = src.with_suffix(".clean.md")
        process_file(src, dst)


if __name__ == "__main__":
    main(sys.argv[1:])
