import os
from pathlib import Path
from bs4 import BeautifulSoup

# 1) Directory containing your split‐out HTML templates
INPUT_DIR  = Path(__file__).parent / "individual"
# 2) Directory where to write cleared templates
OUTPUT_DIR = Path(__file__).parent / "cleared"

def clear_one(input_path: Path, output_path: Path):
    soup = BeautifulSoup(input_path.read_text(encoding="utf8"), "html.parser")

    for td in soup.find_all("td"):
        # skip nested tables entirely
        if td.find("table"):
            continue
        # skip label cells (your section headers, e.g. bgcolor="#ababab")
        if td.get("bgcolor", "").lower() == "#ababab":
            continue
        # otherwise replace whatever was there with the placeholder
        td.string = "__FILL__"

    # write out to the cleared folder
    output_path.write_text(str(soup), encoding="utf8")
    print(f"Wrote cleared template  {output_path.name}")

def main():
    # make sure output folder exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    html_files = sorted(INPUT_DIR.glob("*.html"))

    if not html_files:
        print("No HTML files found in", INPUT_DIR)
        return

    for inp in html_files:
        outp = OUTPUT_DIR / inp.name
        clear_one(inp, outp)

if __name__ == "__main__":
    main()
