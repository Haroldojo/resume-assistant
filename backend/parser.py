import fitz  # PyMuPDF
import re
import json
import os

# Paths
PDF_PATH = "data/Parul_resume.pdf"
TXT_PATH = "data/resume.txt"
JSON_PATH = "data/resume_sections.json"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# 1️⃣ Extract text from PDF
doc = fitz.open(PDF_PATH)
all_text = ""
for page in doc:
    text = page.get_text()
    all_text += text + "\n"

# 2️⃣ Light cleaning
clean_text = re.sub(r"\n\s*\n", "\n", all_text)  # remove multiple blank lines
clean_text = re.sub(r"-\n", "", clean_text)      # fix hyphenated words
clean_text = clean_text.strip()

# Save raw text
with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write(clean_text)

print(f"[INFO] Saved cleaned text to {TXT_PATH}")
print(f"[INFO] Total characters extracted: {len(clean_text)}")

# 3️⃣ Enhanced section detection
# Detect headings, subheadings, bullets
section_headings = ["Experience", "Projects", "Skills", "Education", "Certifications", "Internship", "Achievements"]
sections = {}

# Normalize text for regex matching
norm_text = clean_text.replace("\r\n", "\n")

# Add fuzzy matching pattern for headings
heading_pattern = r"(?i)^\s*({})\s*$".format("|".join(section_headings))
lines = norm_text.split("\n")
current_section = None
buffer = []

for line in lines:
    if re.match(heading_pattern, line):
        if current_section and buffer:
            sections[current_section] = "\n".join(buffer).strip()
        current_section = line.strip()
        buffer = []
    elif current_section:
        buffer.append(line)

# Add last section
if current_section and buffer:
    sections[current_section] = "\n".join(buffer).strip()

# Optional: clean bullets (remove excess spaces, keep dash/•)
for sec in sections:
    sections[sec] = re.sub(r"\s*•\s*", "• ", sections[sec])
    sections[sec] = re.sub(r"\s*-\s*", "- ", sections[sec])

# Save JSON sections
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(sections, f, indent=4, ensure_ascii=False)

print(f"[INFO] Saved detected sections to {JSON_PATH}")
