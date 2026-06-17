"""Convert paper.tex to paper.html with proper rendering."""
import re

with open('paper.tex', encoding='utf-8') as f:
    tex = f.read()

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tabula Rasa — Capacity Boundary in Arithmetic Continual Learning</title>
<style>
body{font-family:Georgia,'Times New Roman',serif;max-width:800px;margin:40px auto;padding:0 20px;line-height:1.6;color:#222;font-size:16px}
h1{font-size:28px;text-align:center;margin-bottom:5px;margin-top:20px}
.author{text-align:center;color:#666;font-size:14px;margin-bottom:30px}
h2{font-size:20px;margin-top:30px;border-bottom:1px solid #ddd;padding-bottom:5px}
h3{font-size:17px;margin-top:25px;margin-bottom:8px}
p{margin:10px 0;text-align:justify}
figure{margin:20px 0;text-align:center}
img{max-width:100%;border:1px solid #ddd;border-radius:4px}
figcaption{font-size:13px;color:#555;margin-top:5px;text-align:center}
ul{padding-left:25px;margin:10px 0}
li{margin:5px 0}
strong{font-weight:bold}
em{font-style:italic}
code{background:#f4f4f4;padding:2px 5px;border-radius:3px;font-size:14px}
hr{border:none;border-top:1px solid #ddd;margin:30px 0}
.abstract{background:#f9f9f9;padding:15px 20px;border-left:3px solid #4a7fb5;margin:20px 0;font-size:15px;border-radius:0 4px 4px 0}
.abstract p{margin:3px 0}
.blockquote{border-left:3px solid #ddd;padding-left:15px;color:#555;margin:15px 0}
table{border-collapse:collapse;margin:15px auto;font-size:14px}
td,th{border:1px solid #ddd;padding:6px 10px;text-align:center}
th{background:#f5f5f5;font-weight:bold}
</style>
</head>
<body>
"""

def tex_to_html(text):
    """Convert LaTeX text to HTML."""
    text = text.replace(r'\%', '%')
    text = text.replace(r'\_', '_')
    text = text.replace('---', '&mdash;')
    text = text.replace('--', '&ndash;')
    text = text.replace(r'$\boxtimes$', '&#9745;')
    text = text.replace(r'$\square$', '&#9744;')
    text = re.sub(r'\\textbf\{([^}]*)\}', r'<strong>\1</strong>', text)
    text = re.sub(r'\\emph\{([^}]*)\}', r'<em>\1</em>', text)
    text = re.sub(r'\\texttt\{([^}]*)\}', r'<code>\1</code>', text)
    text = re.sub(r'\\textit\{([^}]*)\}', r'<i>\1</i>', text)
    text = text.replace(r'\{', '{').replace(r'\}', '}')
    text = re.sub(r'\$\\sigma\$', '&sigma;', text)
    text = re.sub(r'\$\\lambda\$', '&lambda;', text)
    text = re.sub(r'\$\\to\$', '&rarr;', text)
    text = re.sub(r'\$\\pm\$', '&plusmn;', text)
    text = re.sub(r'\\textasciitilde\{\}', '~', text)
    text = re.sub(r'\$([0-9.+\-]+)\$', r'\1', text)  # simple number math
    return text


# Title
title_match = re.search(r'\\section\{([^}]+)\}', tex)
if title_match:
    html += '<h1>' + tex_to_html(title_match.group(1)) + '</h1>\n'

# Abstract
abstract_match = re.search(r'\\subsection\{Abstract\}(.*?)\\begin\{center\}', tex, re.DOTALL)
if abstract_match:
    abstract_text = tex_to_html(abstract_match.group(1).strip())
    html += '<div class="abstract">' + abstract_text.replace('\n\n', '</p><p>') + '</div>\n'

# Process the body: split into sections
sections = re.split(r'\\(?:section|subsection)\{([^}]+)\}', tex)[1:]
# sections is [title1, content1, title2, content2, ...]
for i in range(0, len(sections), 2):
    title = tex_to_html(sections[i].strip())
    content = sections[i+1] if i+1 < len(sections) else ''

    if title == 'Abstract':
        continue

    is_sub = bool(re.search(r'\\subsection\{' + re.escape(sections[i]) + r'\}', tex))
    html += f'<h{3 if is_sub else 2}>{title}</h{3 if is_sub else 2}>\n'

    # Process figures first (remove from content)
    def process_figure(m):
        inner = m.group(1)
        caption = tex_to_html(m.group(2))
        imgs = re.findall(r'\\includegraphics\[([^\]]*)\]\{([^}]+)\}', inner)
        img_html = ''
        for opts, src in imgs:
            width = re.search(r'width=([0-9.]+\\?textwidth)', opts)
            style = 'width:45%' if width and '0.45' in width.group(1) else 'width:60%'
            img_html += f'<img src="{src}" alt="figure" style="{style}" />\n'
        return f'<figure>\n{img_html}<figcaption>{caption}</figcaption>\n</figure>\n'

    content = re.sub(
        r'\\begin\{figure\}\[h\].*?\\centering\s*(.*?)\\caption\{([^}]*)\}.*?\\end\{figure\}',
        process_figure, content, flags=re.DOTALL
    )

    # Remove other LaTeX environments
    content = re.sub(r'\\begin\{center\}.*?\\end\{center\}(.*?)\\rule', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{center\}|\\end\{center\}|\\rule\{[^}]*\}\{[^}]*\}', '', content)
    content = re.sub(r'\\label\{[^}]*\}', '', content)

    # Process itemize lists
    def process_list(m):
        items_raw = m.group(1)
        items = re.split(r'\\item\s*', items_raw)
        items = [tex_to_html(i.strip()) for i in items if i.strip()]
        lis = '\n'.join(f'  <li>{i}</li>' for i in items)
        return f'<ul>\n{lis}\n</ul>\n'

    content = re.sub(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', process_list, content, flags=re.DOTALL)

    # Convert remaining LaTeX math
    content = tex_to_html(content)

    # Clean up
    content = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', '', content)  # remove leftover commands
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Split into paragraphs
    paragraphs = []
    for block in content.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        if block.startswith('<') and not block.startswith('<p>'):
            paragraphs.append(block)
        else:
            paragraphs.append(f'<p>{block}</p>')
    html += '\n'.join(paragraphs) + '\n'

html += '\n</body>\n</html>'

with open('paper.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f'paper.html written ({len(html)} bytes)')
print(f'Sections: {len(sections)//2}')
