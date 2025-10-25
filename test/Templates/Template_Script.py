
import os
from pathlib import Path
from bs4 import BeautifulSoup
from copy import deepcopy

# ── CONFIG ────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MASTER_FILE   = BASE_DIR / "KenoTemplates.htm"
INDIVIDUAL    = BASE_DIR / "individual"
CLEARED       = BASE_DIR / "cleared"
FIRST_EXAMPLE = BASE_DIR / "first_example"

for d in (INDIVIDUAL, CLEARED, FIRST_EXAMPLE):
    d.mkdir(exist_ok=True)

def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()

# read & parse once
master_html = MASTER_FILE.read_text(encoding="utf8")
master_soup = BeautifulSoup(master_html, "html.parser")
headers     = master_soup.find_all("h1")

for header in headers:
    title = header.get_text(strip=True)
    fname = slugify(title) + ".html"

    # ── 1) SPLIT ─────────────────────────────────────────────────────────
    split_soup = BeautifulSoup(master_html, "html.parser")
    body = split_soup.body
    body.clear()

    # copy this <h1> node exactly
    body.append(deepcopy(header))

    # collect everything until the next <h1>
    fragment = []
    for sib in header.next_siblings:
        if getattr(sib, "name", None) == "h1":
            break
        fragment.append(str(sib))
    body.append(BeautifulSoup("".join(fragment), "html.parser"))

    path_ind = INDIVIDUAL / fname
    path_ind.write_text(str(split_soup), encoding="utf8")

    # ── 2) CLEAR ─────────────────────────────────────────────────────────
    clear_soup = BeautifulSoup(str(split_soup), "html.parser")
    for td in clear_soup.find_all("td"):
        # keep your section-title cells
        if td.get("bgcolor", "").lower() == "#ababab":
            continue
        # keep any cell that *contains* a nested table (we'll clear inside it)
        if td.find("table"):
            continue
        td.clear()
        td.string = "__FILL__"

    path_clr = CLEARED / fname
    path_clr.write_text(str(clear_soup), "utf8")

    # ── 3) FIRST EXAMPLE ─────────────────────────────────────────────────
    first_soup = BeautifulSoup(str(clear_soup), "html.parser")
    tbody = first_soup.find("tbody")
    if tbody:
        # top-level rows
        rows = [tr for tr in tbody.find_all("tr", recursive=False)]
        # indices of Free Text rows
        ft_idxs = [
            i for i, tr in enumerate(rows)
            if tr.find("td", {"bgcolor":"#ababab"}) 
               and tr.find("td", {"bgcolor":"#ababab"}).get_text(strip=True) == "Free Text"
        ]
        if len(ft_idxs) >= 2:
            cutoff = ft_idxs[1]
            rows = rows[:cutoff]
        tbody.clear()
        for tr in rows:
            tbody.append(tr)

    path_first = FIRST_EXAMPLE / fname
    path_first.write_text(str(first_soup), "utf8")

    print("Processed -->", fname)
