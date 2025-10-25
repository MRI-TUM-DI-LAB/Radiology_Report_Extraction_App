import os
import json
from bs4 import BeautifulSoup

def parse_table_recursive(table_tag):
    result = {}
    for row in table_tag.find_all("tr", recursive=False):
        headers = row.find_all("th", recursive=False)
        cells = row.find_all("td", recursive=False)

        if len(headers) == 1 and len(cells) == 1:
            key = headers[0].get_text(strip=True)
            cell = cells[0]

            nested_table = cell.find("table")
            if nested_table:
                result[key] = parse_table_recursive(nested_table)
            else:
                result[key] = cell.get_text(strip=True)
    return result

def parse_structured_table(structured_html):
    structured_table = structured_html.find("table")
    if not structured_table:
        return {}
    return parse_table_recursive(structured_table)

def extract_all_structured_reports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    reports = []
    for tr in soup.find_all('tr'):
        tds = tr.find_all('td')
        if len(tds) >= 2 and 'structured' in tds[0].get_text(strip=True).lower():
            structured_td = tds[1]
            report = parse_structured_table(structured_td)
            if report:
                reports.append(report)
    return reports

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_counter = {}

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.html'):
            continue

        file_path = os.path.join(input_dir, fname)
        all_reports = extract_all_structured_reports(file_path)

        if not all_reports:
            print(f"⚠️  Skipped: {fname} (no structured reports found)")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        title_tag = soup.find('title')
        base_name = title_tag.get_text(strip=True) if title_tag else os.path.splitext(fname)[0]

        for i, report_json in enumerate(all_reports, 1):
            count = report_counter.get(base_name, 0) + 1
            report_counter[base_name] = count

            numbered_name = f"{base_name}_{count}.json"
            json_path = os.path.join(output_dir, numbered_name)

            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(report_json, jf, indent=2)
            print(f"✅ Saved: {json_path}")

if __name__ == "__main__":
    input_folder = "./split_html"     
    output_folder = "./ground-truth" 
    process_directory(input_folder, output_folder)
