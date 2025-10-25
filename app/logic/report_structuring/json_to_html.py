from json2html import Json2Html

def json_to_html_table(json_data, title="Structured Report") -> str:

    if isinstance(json_data, dict):
        json_list = [json_data]
    elif isinstance(json_data, list):
        json_list = json_data
    else:
        raise ValueError("json_data must be a dict or list")

    html_start = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border: 1px solid black;
      padding: 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background-color: #f2f2f2;
      width: 25%;
    }}
    tr:nth-child(even) {{
      background-color: #f9f9f9;
    }}
  </style>
</head>
<body>
<h1>{title}</h1>
"""

    html_end = "</body></html>"
    json2html = Json2Html()
    rows = []

    record = json_list[0]
    for key in record:
        td_style = "white-space: pre-wrap; word-wrap: break-word;"
        key_td = f'<td style="{td_style}" bgcolor="#ededed"><b>{key}</b></td>'
        value_td = f'<td style="{td_style}" bgcolor="#ffffff">{json2html.convert(record[key])}</td>'
        rows.append(f"<tr>{key_td}{value_td}</tr>")

    html_table = f"<table>\n<tbody>\n{''.join(rows)}\n</tbody>\n</table>"
    full_html = html_start + html_table + html_end
    return full_html
