"""Typeset the PDF and export a self-contained, reflowable HTML reading copy."""
from pathlib import Path

import typst

HERE = Path(__file__).resolve().parent
SOURCE = str(HERE / "paper.typ")
STYLE = """
<style>
:root {color-scheme:light;color:#192f37;background:#fff;font-family:Georgia,serif}
body {max-width:43rem;margin:3rem auto;padding:0 clamp(1rem,4vw,2rem);font-size:1.12rem;line-height:1.57}
h1,h2,h3,body>p:first-child,th {font-family:system-ui,sans-serif;line-height:1.2}
h1 {font-size:clamp(2rem,5vw,2.8rem);letter-spacing:-.035em}
h2 {margin:2.3rem 0 .75rem;font-size:1.6rem}
h3 {margin:1.45rem 0 .45rem;color:#08665f;font-size:1.13rem}
p {margin:.7rem 0}
body>p:first-child {color:#08665f;font-size:.8rem;font-weight:700;letter-spacing:.035em}
a {color:#08665f}
img {width:100%;height:auto;max-width:100%;margin:1.15rem 0 .4rem}
div:has(>img) {font-size:.92em;line-height:1.4}
table {display:block;max-width:100%;overflow-x:auto;border-collapse:collapse;margin-top:1rem;font-variant-numeric:tabular-nums}
th,td {border-bottom:1px solid #cad7d5;padding:.36rem .5rem;text-align:right;white-space:nowrap}
th:first-child,td:first-child {text-align:left}
math[display=block] {padding:.6rem 0;max-width:100%;overflow-x:auto}
@media(max-width:600px){body{margin:1.4rem auto;font-size:1.04rem} th,td{font-size:.82rem;padding:.3rem} }
</style>
"""
if __name__ == "__main__":
    typst.compile(SOURCE, output=str(HERE / "paper.pdf"))
    html = typst.compile(SOURCE, format="html", sys_inputs={"edition": "web"}).decode()
    (HERE / "paper.html").write_text(html.replace("</head>", STYLE + "</head>", 1))
