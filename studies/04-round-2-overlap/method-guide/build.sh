#!/usr/bin/env bash
# Rebuild the round-two report on the path edit: figures and checked numbers, HTML, PDF with page numbers.
set -euo pipefail
cd "$(dirname "$0")"
uv run --quiet --with matplotlib --with numpy python make_figures.py > /dev/null
{
  printf '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
  printf '<title>Editing the Weights from the Overlap of Many Answers</title>\n'
  cat style.css
  printf '</head>\n'
  cat body-1.html body-2.html body-3.html body-4.html
} > guide.html
# Rendered with Playwright's headless Chromium (Letter, the page's own margins), bounded by the host resource governor.
# The body font is Liberation Serif, the face the two earlier PDFs were rendered in.
shell=$(ls -d "$HOME"/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell | sort -V | tail -1)
timeout --kill-after=10 120 "$shell" --headless --no-pdf-header-footer --print-to-pdf=draft.pdf "file://$PWD/guide.html"
uv run --quiet --with pypdf --with reportlab python stamp.py draft.pdf how-the-path-edit-works.pdf
rm -f draft.pdf
