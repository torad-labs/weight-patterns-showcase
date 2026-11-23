"""Add centered page numbers to draft.pdf -> the final PDF."""
import io, sys
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
src, dst = sys.argv[1], sys.argv[2]
r = PdfReader(src); w = PdfWriter(clone_from=src)
for i, page in enumerate(w.pages, 1):
    pw, ph = float(page.mediabox.width), float(page.mediabox.height)
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=(pw, ph))
    c.setFont("Times-Roman", 9); c.setFillGray(0.3)
    c.drawCentredString(pw / 2, 0.45 * 72, str(i)); c.save()
    page.merge_page(PdfReader(buf).pages[0])
    page.compress_content_streams()
w.add_metadata({"/Title": "Editing the Weights from the Overlap of Many Answers", "/Author": "Marcos Damasceno"})
w.write(dst)
print(len(r.pages), "pages")
