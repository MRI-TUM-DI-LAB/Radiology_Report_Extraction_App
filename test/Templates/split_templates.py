import os
from pathlib import Path
from bs4 import BeautifulSoup
from copy import copy

TEMPLATES_DIR = Path(__file__).parent
INPUT_FILE    = TEMPLATES_DIR / "KenoTemplates.htm"
OUTPUT_DIR    = TEMPLATES_DIR / "individual"   # make sure this matches your actual folder

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf8") as f:
        soup = BeautifulSoup(f, "html.parser")

    for header in soup.find_all("h1"):
        title = header.get_text().strip()
        slug  = title.lower().replace(" ", "_").replace("/", "_")
        out_fp = OUTPUT_DIR / f"{slug}.html"

        # Gather everything until the next <h1>
        content_nodes = []
        for sib in header.next_siblings:
            if getattr(sib, "name", None) == "h1":
                break
            content_nodes.append(str(sib))

        # Build a minimal HTML document
        out_soup = BeautifulSoup(
            "<!doctype html><html><head>"
            "<meta charset='utf-8'><title></title></head><body></body></html>",
            "html.parser"
        )
        out_soup.title.string = title
        out_soup.body.append(copy(header))  # ← shallow‐copy the <h1> node
        out_soup.body.append(BeautifulSoup("".join(content_nodes), "html.parser"))

        with open(out_fp, "w", encoding="utf8") as outf:
            outf.write(str(out_soup))

        print(f"Wrote: {out_fp}")

if __name__ == "__main__":
    main()
