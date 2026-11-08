# -*- coding: utf-8 -*-
from docx import Document

path = r"D:\HuaweiMoveData\Users\zheng\Desktop\shiyougan\刘日浠 24072108 北京市优秀学生干部登记表.docx"
out = r"D:\人工智能+\registration_dump.txt"

doc = Document(path)
lines = []
lines.append("=== PARAGRAPHS ===")
for i, p in enumerate(doc.paragraphs):
    if p.text.strip():
        lines.append(f"{i} | {p.text}")
lines.append("=== TABLES ===")
for ti, t in enumerate(doc.tables):
    lines.append(f"--- table {ti}: {len(t.rows)}x{len(t.columns)} ---")
    for ri, row in enumerate(t.rows):
        cells = []
        seen = set()
        for c in row.cells:
            if id(c._tc) in seen:
                continue
            seen.add(id(c._tc))
            cells.append(c.text.replace("\n", " ⏎ ").strip())
        lines.append(f"{ri} " + " || ".join(cells))

with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", out, len(lines))
