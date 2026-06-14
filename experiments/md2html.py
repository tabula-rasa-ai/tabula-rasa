"""Convert paper_draft.md to publication-ready HTML with embedded CSS."""

from pathlib import Path

import markdown

md_path = Path(r"C:\Users\Admin\tabula-rasa\paper_draft.md")
html_path = Path(r"C:\Users\Admin\tabula-rasa\paper.html")

md_content = md_path.read_text(encoding="utf-8")

# Publication-quality CSS (self-contained, no CDN)
css = """
* { box-sizing: border-box; }
body {
  font-family: 'Times New Roman', Times, serif;
  font-size: 12pt;
  line-height: 1.7;
  max-width: 7.5in;
  margin: 0 auto;
  padding: 0.8in 0.5in;
  color: #000;
  background: #fff;
}
h1 { font-size: 20pt; text-align: center; margin-top: 0; margin-bottom: 6pt; }
h2 { font-size: 14pt; margin-top: 28pt; margin-bottom: 8pt; border-bottom: 1px solid #999; padding-bottom: 4pt; }
h3 { font-size: 12pt; margin-top: 20pt; margin-bottom: 6pt; font-style: italic; }
h4 { font-size: 12pt; margin-top: 14pt; margin-bottom: 4pt; }
p { text-align: justify; margin: 6pt 0; }
table { border-collapse: collapse; width: 100%; margin: 12pt 0; font-size: 10pt; page-break-inside: avoid; }
th, td { border: 1px solid #333; padding: 4pt 6pt; text-align: center; }
th { background: #eee; font-weight: bold; }
code {
  font-family: 'Courier New', Courier, monospace;
  font-size: 9pt;
  background: #f4f4f4;
  padding: 1pt 3pt;
  border-radius: 2pt;
}
pre {
  background: #f8f8f8;
  border: 1px solid #ddd;
  border-left: 3px solid #2c7;
  padding: 8pt 12pt;
  font-size: 9pt;
  line-height: 1.3;
  overflow-x: auto;
  margin: 10pt 0;
  page-break-inside: avoid;
}
blockquote {
  border-left: 3px solid #bbb;
  margin: 10pt 0;
  padding: 4pt 12pt;
  color: #444;
  font-style: italic;
  background: #fafafa;
}
img { max-width: 100%; display: block; margin: 12pt auto; }
hr { border: none; border-top: 1px solid #ccc; margin: 24pt 0; }
strong { color: #000; }
em { color: #333; }
.author { text-align: center; font-size: 12pt; margin-bottom: 24pt; color: #555; }
.abstract {
  margin: 18pt 0; padding: 12pt 18pt;
  border-left: 2px solid #2c7;
  font-size: 11pt;
  background: #f9fcf9;
}
.abstract strong { font-style: normal; }
@media print {
  body { padding: 0; max-width: none; }
  h2 { page-break-after: avoid; }
  h3 { page-break-after: avoid; }
  pre, table { page-break-inside: avoid; }
}
"""

# Process abstract
html_body = markdown.markdown(
    md_content, extensions=["extra", "tables", "fenced_code", "footnotes", "sane_lists"]
)

# Wrap abstract if it exists
if "## Abstract" in html_body:
    html_body = html_body.replace("<h2>Abstract</h2>", '<h2>Abstract</h2>\n<div class="abstract">')
    # Find the next h2 to close the div
    parts = html_body.split("</h2>\n<p>", 1)
    if len(parts) > 1:
        # Close abstract div before next section
        html_body = html_body.replace("<h2>1.", "</div>\n\n<h2>1.", 1)

html_full = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Online Elastic Weight Consolidation: Empirical Validation and Scalability Limits</title>
<style>{css}</style>
</head>
<body>
<p class="author">Tabula Rasa AI &middot; Independent Research &middot; <code>github.com/tabula-rasa-ai/tabula-rasa</code></p>

{html_body}

</body>
</html>
"""

html_path.write_text(html_full, encoding="utf-8")
print(f"Publication-ready HTML generated: {html_path}")
print(f"Size: {html_path.stat().st_size / 1024:.0f} KB")
print()
print("Open in browser and use Ctrl+P → Save as PDF to get the paper.")
print("Browser print settings:")
print("  - Paper size: Letter")
print("  - Margins: None (handled by CSS)")
print("  - Background graphics: ✅")
