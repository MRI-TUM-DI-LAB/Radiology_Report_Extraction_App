import os
from pathlib import Path
from bs4 import BeautifulSoup

# adjust these paths as needed
INPUT_DIR  = Path(__file__).parent / "cleared"
OUTPUT_DIR = Path(__file__).parent / "first_example"

OUTPUT_DIR.mkdir(exist_ok=True)

def keep_first_example(in_path: Path, out_path: Path):
    # parse the HTML
    soup = BeautifulSoup(in_path.read_text(encoding="utf8"), "html.parser")

    # locate the table body
    tbody = soup.find("tbody")
    if not tbody:
        return

    # get only the top–level <tr> children
    rows = [tr for tr in tbody.find_all("tr", recursive=False)]

    # find the indices where the first <td> is “Free Text”
    ft_indices = [
        i for i, tr in enumerate(rows)
        if tr.find("td") and tr.find("td").get_text(strip=True) == "Free Text"
    ]

    # if there are at least two Free Text sections, slice at the second one
    if len(ft_indices) >= 2:
        keep_up_to = ft_indices[1]
        rows = rows[:keep_up_to]

    # clear out all existing rows and re-append only the ones we want
    tbody.clear()
    for tr in rows:
        tbody.append(tr)

    # write the result
    out_path.write_text(str(soup), encoding="utf8")
    print("Wrote:", out_path.name)

def main():
    for html_file in INPUT_DIR.glob("*.html"):
        keep_first_example(html_file, OUTPUT_DIR / html_file.name)

if __name__ == "__main__":
    main()
