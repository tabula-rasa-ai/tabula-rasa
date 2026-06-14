"""Convert paper_draft.md to PDF using weasyprint."""

from pathlib import Path

import markdown
from weasyprint import HTML

md_path = Path(r"C:\Users\Admin\tabula-rasa\paper_draft.md")
pdf_path = md_path.with_suffix(".pdf")

# Read markdown
md_content = md_path.read_text(encoding="utf-8")

# Add CSS for publication-quality PDF
css = """
@page {
  size: letter;
  margin: 1in;
}
body {
  font-family: 'Times New Roman', Times, serif;
  font-size: 12pt;
  line-height: 1.6;
  color: #000;
}
h1 { font-size: 18pt; text-align: center; margin-top: 0.5in; }
h2 { font-size: 14pt; margin-top: 24pt; border-bottom: 1px solid #ccc; }
h3 { font-size: 12pt; margin-top: 18pt; font-style: italic; }
h4 { font-size: 12pt; margin-top: 12pt; }
p { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 12pt 0; font-size: 10pt; }
table, th, td { border: 1px solid #555; padding: 4pt 8pt; }
th { background: #eee; }
code {
  font-family: 'Courier New', monospace;
  font-size: 9pt;
  background: #f5f5f5;
  padding: 1pt 3pt;
}
pre {
  background: #f5f5f5;
  border: 1px solid #ddd;
  padding: 8pt;
  font-size: 9pt;
  line-height: 1.2;
  overflow-x: auto;
}
blockquote {
  border-left: 3px solid #ccc;
  margin-left: 0;
  padding-left: 12pt;
  color: #555;
  font-style: italic;
}
"""

# Convert to HTML with extensions
html_body = markdown.markdown(
    md_content, extensions=["extra", "codehilite", "tables", "fenced_code", "footnotes"]
)

html_full = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{css}</style></head>
<body>
{html_body}
</body>
</html>
"""

# Generate PDF
HTML(string=html_full).write_pdf(str(pdf_path))
print(f"PDF generated: {pdf_path}")
print(f"Size: {pdf_path.stat().st_size / 1024:.0f} KB")
