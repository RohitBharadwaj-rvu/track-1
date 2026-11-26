"""Zero-dependency helper to update docs/BRIEF.md from eval JSON and export 2-page HTML/PDF (Role 4).

Usage:
    python -m training.brief_tools --update-row "Stage 2 MVP Recurrent PPO" --report eval_dev.json
    python -m training.brief_tools --export-html docs/BRIEF.html
    python -m training.brief_tools --export-pdf docs/BRIEF.pdf
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRIEF = ROOT / "docs" / "BRIEF.md"


def format_outcomes_row(stage_name: str, report: dict, suite_label: str | None = None) -> str:
    """Format one Markdown table row for Page 2 of docs/BRIEF.md."""
    overall = report["summary"]["overall"]
    manifest = report.get("metadata", {}).get("maps_manifest", [])
    if suite_label is None:
        if len(manifest) == 1:
            seeds = manifest[0].get("seeds", [])
            seed_str = ", ".join(str(s) for s in seeds)
            suite_label = f"`{manifest[0]['wad']}` (seeds `{seed_str}`)"
        else:
            suite_label = f"{len(manifest)} maps"
    return (
        f"| {stage_name} | {suite_label} | {overall['episodes']} | {overall['exits']} "
        f"| {overall['keys_picked_up']} | {overall['locked_doors_unlocked']} "
        f"| {overall['kills']} | {overall['mean_unique_cells']} | {overall['errors']} |"
    )


def update_brief_outcomes_row(
    brief_path: Path,
    stage_name: str,
    report: dict,
    suite_label: str | None = None,
) -> str:
    """Replace an existing Outcomes table row matching `stage_name` or insert a new one."""
    text = brief_path.read_text(encoding="utf-8")
    new_row = format_outcomes_row(stage_name, report, suite_label=suite_label)
    lines = text.splitlines()

    prefix = f"| {stage_name} |"
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[i] = new_row
            updated = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
            brief_path.write_text(updated, encoding="utf-8")
            return updated

    # Insert after the table separator inside Section 1 of Page 2 if not already present.
    in_outcomes = False
    for i, line in enumerate(lines):
        if line.strip().startswith("### 1. Baseline vs. Final Outcomes"):
            in_outcomes = True
        elif in_outcomes and line.strip().startswith("|---"):
            lines.insert(i + 1, new_row)
            updated = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
            brief_path.write_text(updated, encoding="utf-8")
            return updated

    raise ValueError(f"Could not find Outcomes table in {brief_path}")


def _render_inline(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def markdown_to_two_page_html(markdown_text: str, title: str = "Track 1 Brief") -> str:
    """Convert `docs/BRIEF.md` into clean styled 2-page HTML ready for print/PDF."""
    lines = markdown_text.splitlines()
    body: list[str] = []
    i = 0
    in_ul = False
    in_ol = False

    def _close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            body.append("</ul>")
            in_ul = False
        if in_ol:
            body.append("</ol>")
            in_ol = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            _close_lists()
            i += 1
            continue

        if stripped == "---":
            _close_lists()
            body.append("<hr />")
            i += 1
            continue

        if stripped.startswith("# "):
            _close_lists()
            body.append(f"<h1>{_render_inline(stripped[2:])}</h1>")
            i += 1
            continue

        if stripped.startswith("## "):
            _close_lists()
            heading = stripped[3:]
            if heading.lower().startswith("page 2"):
                body.append('<div class="page-break"></div>')
            body.append(f"<h2>{_render_inline(heading)}</h2>")
            i += 1
            continue

        if stripped.startswith("### "):
            _close_lists()
            body.append(f"<h3>{_render_inline(stripped[4:])}</h3>")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|---"):
            _close_lists()
            headers = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().split("|")[1:-1]])
                i += 1
            body.append("<table><thead><tr>")
            for h in headers:
                body.append(f"<th>{_render_inline(h)}</th>")
            body.append("</tr></thead><tbody>")
            for row in rows:
                body.append("<tr>")
                for cell in row:
                    body.append(f"<td>{_render_inline(cell)}</td>")
                body.append("</tr>")
            body.append("</tbody></table>")
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            if in_ol:
                body.append("</ol>")
                in_ol = False
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{_render_inline(stripped[2:])}</li>")
            i += 1
            continue

        ol_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ol_match:
            if in_ul:
                body.append("</ul>")
                in_ul = False
            if not in_ol:
                body.append("<ol>")
                in_ol = True
            body.append(f"<li>{_render_inline(ol_match.group(1))}</li>")
            i += 1
            continue

        _close_lists()
        body.append(f"<p>{_render_inline(stripped)}</p>")
        i += 1

    _close_lists()

    css = """
    @page { size: letter; margin: 0.45in; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           font-size: 10.2pt; line-height: 1.38; color: #1a1a1a; max-width: 7.6in; margin: 0 auto; }
    h1 { font-size: 16pt; margin: 0 0 6pt 0; border-bottom: 1.5px solid #222; padding-bottom: 4pt; }
    h2 { font-size: 12.5pt; margin: 10pt 0 4pt 0; color: #0f3d5e; }
    h3 { font-size: 10.8pt; margin: 8pt 0 3pt 0; }
    p { margin: 3pt 0 5pt 0; }
    ul, ol { margin: 2pt 0 6pt 16pt; padding: 0; }
    li { margin-bottom: 2.5pt; }
    code { font-family: Consolas, "Courier New", monospace; font-size: 9pt;
           background: #f3f5f7; padding: 0.5pt 3pt; border-radius: 2pt; }
    table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt 0; font-size: 8.8pt; }
    th, td { border: 1px solid #cbd5e1; padding: 3.5pt 5pt; text-align: left; }
    th { background: #eef3f8; font-weight: 600; }
    hr { border: none; border-top: 1px solid #d1d5db; margin: 8pt 0; }
    .page-break { break-before: page; page-break-before: always; }
    """
    return (
        f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{css}</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )


def find_headless_browser() -> str | None:
    """Locate Edge or Chrome binary for headless HTML->PDF printing."""
    for name in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(name)
        if found:
            return found
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def export_pdf(brief_path: Path, pdf_path: Path) -> bool:
    """Render `brief_path` to HTML and print to `pdf_path` via headless browser."""
    browser = find_headless_browser()
    html_text = markdown_to_two_page_html(brief_path.read_text(encoding="utf-8"))
    if browser is None:
        return False

    with tempfile.TemporaryDirectory(prefix="brief_html_") as td:
        tmp_html = Path(td) / "brief.html"
        tmp_profile = Path(td) / "profile"
        tmp_html.write_text(html_text, encoding="utf-8")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--disable-extensions",
            f"--user-data-dir={tmp_profile}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path.resolve()}",
            tmp_html.resolve().as_uri(),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return res.returncode == 0 and pdf_path.is_file()
        except (OSError, subprocess.TimeoutExpired):
            return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Role 4 brief updater & HTML/PDF exporter.")
    parser.add_argument("--brief", type=Path, default=DEFAULT_BRIEF, help="Path to BRIEF.md")
    parser.add_argument("--update-row", default=None, help="Row stage name to update in BRIEF.md")
    parser.add_argument("--suite-label", default=None, help="Optional custom Suite column label")
    parser.add_argument("--report", type=Path, default=None, help="Eval JSON report for --update-row")
    parser.add_argument("--export-html", nargs="?", const=str(ROOT / "docs" / "BRIEF.html"), default=None)
    parser.add_argument("--export-pdf", nargs="?", const=str(ROOT / "docs" / "BRIEF.pdf"), default=None)
    args = parser.parse_args(argv)

    if args.update_row:
        if args.report is None:
            parser.error("--update-row requires --report <eval.json>")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        update_brief_outcomes_row(args.brief, args.update_row, report, suite_label=args.suite_label)
        print(f"Updated '{args.update_row}' row in {args.brief}")

    if args.export_html:
        out_html = Path(args.export_html)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        html_text = markdown_to_two_page_html(args.brief.read_text(encoding="utf-8"))
        out_html.write_text(html_text, encoding="utf-8")
        print(f"Exported 2-page HTML to {out_html}")

    if args.export_pdf:
        out_pdf = Path(args.export_pdf)
        if export_pdf(args.brief, out_pdf):
            print(f"Exported 2-page PDF to {out_pdf}")
        else:
            print(
                "Could not find headless Edge/Chrome to generate PDF automatically. "
                "Export HTML with `--export-html` and print to PDF from your browser."
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
