import re
import logging
from bs4 import BeautifulSoup
logger = logging.getLogger('root')


def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["style", "meta", "head", "script"]):
        tag.decompose()

    for tag in soup(True):
        tag.attrs = {k: v for k, v in tag.attrs.items() if k not in ["style", "bgcolor", "border"]}

    clean_html = soup.prettify()
    return clean_html.strip()

def clean_json(node):
    # If it's a list with one element, just return the cleaned single element
    if isinstance(node, list):
        if len(node) == 1:
            return clean_json(node[0])
        return [clean_json(n) for n in node]

    # If it's a dictionary
    if isinstance(node, dict):
        # If it only has a _value key, return that
        if list(node.keys()) == ['_value']:
            return node['_value']
        
        cleaned = {}
        for key, value in node.items():
            if key == '_attributes':
                continue  # skip attributes
            cleaned[key] = clean_json(value)

        # Flatten if single key and the value is primitive or dict
        if len(cleaned) == 1:
            sole_key = next(iter(cleaned))
            return {sole_key: cleaned[sole_key]} if isinstance(cleaned[sole_key], dict) else cleaned[sole_key]
        
        return cleaned

    return node  # primitive (str, int, etc.)

def find_by_name_in_json(obj, name_to_find):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == name_to_find:
                return v
            result = find_by_name_in_json(v, name_to_find)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_by_name_in_json(item, name_to_find)
            if result:
                return result
    return None

def extract_table(table_data):
    result = {}

    if isinstance(table_data, list):
        for entry in table_data:
            if isinstance(entry, dict) and 'th' in entry and 'td' in entry:
                key = entry['th']
                value = entry['td']
                if isinstance(value, list):
                    # Nested table, recurse
                    result[key] = extract_table(value)
                else:
                    result[key] = value
            elif isinstance(entry, list):
                if len(entry) == 2:
                    result[entry[0]] = extract_table(entry[1])

        return result

    elif isinstance(table_data, dict):
        if 'th' in table_data and 'td' in table_data:
            # Handle single row table
            key = table_data['th']
            value = table_data['td']
            if isinstance(value, list):
                # Nested table, recurse
                result[key] = extract_table(value)
            else:
                result[key] = value
        return result
    
    return table_data


def extract_all_tables(node):
    tables_list = find_by_name_in_json(node, "table")
    if not tables_list:
        raise ValueError("No table found in the JSON structure")

    logger.info(f"Found {len(tables_list)} tables in the JSON structure")
    result = {}
    for table in tables_list:
        if len(table) != 2:
            raise Exception(f"Table format is invalid: {table}")
        key = table[0]
        if key in result.keys():
            key = f"{key}_{len(result)}"
        result[key] = extract_table(table[1])

    return result

