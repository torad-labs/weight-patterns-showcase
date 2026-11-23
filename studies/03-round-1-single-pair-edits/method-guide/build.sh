#!/usr/bin/env bash
# Rebuild the internal method guide: figures, HTML, PDF with page numbers.
set -euo pipefail
cd "$(dirname "$0")"
uv run --quiet --with matplotlib --with numpy python make_figures.py
{
  printf '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
  printf '<title>How the Trace-Derived Edit Works (internal guide)</title>\n'
  cat style.css
  printf '</head>\n'
  cat body-1.html body-2.html body-3.html body-4.html
} > guide.html
# Rendered with Playwright Chromium (page.pdf, Letter, preferCSSPageSize); host chromium runs behind Host Shield.
uv run --quiet --with pypdf --with reportlab python stamp.py draft.pdf how-the-trace-derived-edit-works.pdf
rm -f draft.pdf
