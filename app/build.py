"""Embed data/app_data.json into app/template.html -> app/index.html"""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
data = (root / "data" / "app_data.json").read_text()
html = (root / "app" / "template.html").read_text().replace("/*DATA*/", "window.__DATA__=" + data.replace("</", "<\\/") + ";")
(root / "app" / "index.html").write_text(html)
print("wrote app/index.html", len(html) // 1024, "KB")
