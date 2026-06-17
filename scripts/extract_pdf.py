import glob
import os

import fitz

research_dir = r"C:\Users\Admin\tabula-rasa\Research"
print(f"Research dir exists: {os.path.isdir(research_dir)}")
print(f"Contents: {os.listdir(research_dir)}")

pdfs = glob.glob(os.path.join(research_dir, "*.pdf"))
print(f"PDFs: {pdfs}")

if pdfs:
    doc = fitz.open(pdfs[0])
    text = ""
    for page in doc:
        text += page.get_text()
    print(text)
else:
    print("No PDFs found")
