import json
import os
import re
import logging
from bs4 import BeautifulSoup
from pathlib import Path
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, GenerationConfig
from html_to_json import convert as html_to_json_convert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SOURCE = Path("./befundvorlagen/v3_v2_mix")
DEST_DIR_DE = Path.cwd() / "./befundvorlagen/final"
os.makedirs(DEST_DIR_DE, exist_ok=True)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './vertexai_credentials.json'

GOOGLE_CLOUD_PROJECT = 'llm-structuring'
GOOGLE_CLOUD_REGION = 'us-central1'
VERTEX_MODEL_NAME = "gemini-2.0-flash"
aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)
vertex_model = GenerativeModel(VERTEX_MODEL_NAME)
generation_config = GenerationConfig(
    temperature=0.2,
    top_p=1.0,
    top_k=30,
#     max_output_tokens=8192,
)



def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["style", "meta", "head", "script"]):
        tag.decompose()

    for tag in soup(True):
#        tag.attrs = {k: v for k, v in tag.attrs.items() if k not in ["style", "bgcolor", "border"]}
        tag.attrs = {}

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

def decode_unicode_json(json_input):
    json_text = json.dumps(json_input, indent=2, ensure_ascii=False)
    parsed = json.loads(json_text)
    return parsed

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


def transform_th_td(obj):
    if isinstance(obj, list):
        # Recurse into each item in a list
        return [transform_th_td(item) for item in obj]

    elif isinstance(obj, dict):
        # If 'th' and 'td' are both keys, transform the object
        if 'th' in obj and 'td' in obj:
            key = obj['th']
            if isinstance(key, str):
                value = transform_th_td(obj['td'])
                return {key: value}
            else:
                return {k: transform_th_td(v) for k, v in obj.items() if k != 'td' and k != 'th'}

        # Otherwise recurse through the dict
        new_obj = {}
        for k, v in obj.items():
            new_obj[k] = transform_th_td(v)
        return new_obj

    # For primitive values, return as-is
    return obj


def clean_json_new(data):
    if isinstance(data, dict):
        # Remove empty structures
        cleaned = {
            k: clean_json(v) for k, v in data.items()
            if v not in ({}, [], None)
        }

        # Remove keys that are now empty after recursion
        cleaned = {k: v for k, v in cleaned.items() if v != {} and v != []}

        # Special cleanup: remove _value if it’s the only key
        if "_value" in cleaned and len(cleaned) == 1:
            return cleaned["_value"]

        return cleaned if cleaned else None

    elif isinstance(data, list):
        # Recursively clean list items
        cleaned_list = [clean_json(item) for item in data]
        # Remove None values
        cleaned_list = [item for item in cleaned_list if item is not None]
        return cleaned_list if cleaned_list else None

    else:
        return data

def remove_suffix(data, suffix):
    if isinstance(data, dict):
        keys_to_rename = [key for key in data if key.endswith(suffix)]
        for key in keys_to_rename:
            value = data.pop(key)
            # Recursively update the value in case it's a dict/list
            data[key[:-len(suffix)]] = value
        # Process remaining items
        for key in list(data.keys()):
            data[key] = remove_suffix(data[key], suffix)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = remove_suffix(data[i], suffix)
    return data


def rename_keys_with_suffix_and_move_to_end(data, suffix, new_key_name, default_notes):
    if isinstance(data, dict):
        keys_to_rename = [key for key in data if suffix in key]
        for key in keys_to_rename:
            value = data.pop(key)
            # Recursively update the value in case it's a dict/list
            if default_notes not in data.keys():
                data[new_key_name] = "__FILL__"
        # Process remaining items
        for key in list(data.keys()):
            data[key] = rename_keys_with_suffix_and_move_to_end(data[key], suffix, new_key_name, default_notes)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = rename_keys_with_suffix_and_move_to_end(data[i], suffix, new_key_name, default_notes)
    return data


def remove_keys_with_suffix(data, suffix):
    if isinstance(data, dict):
        keys_to_remove = [key for key in data if key.endswith(suffix)]
        for key in keys_to_remove:
            del data[key]
        # Recursively check nested structures
        for key, value in data.items():
            remove_keys_with_suffix(value, suffix)
    elif isinstance(data, list):
        for item in data:
            remove_keys_with_suffix(item, suffix)
    return data

def collapse_single_key_dicts(data, remove_keys, parent_key=None):
    if isinstance(data, dict):
        # Recursively apply to all child values first
        for key in list(data.keys()):
            data[key] = collapse_single_key_dicts(data[key], remove_keys, key)

        # Collapse if only one key and it's the same as parent
        if len(data) == 0:
            return "__FILL__"
        elif len(data) == 1:
    #        key = next(iter(data))
    #        if key == parent_key or key in remove_keys:
    #            return data[key]
            return "__FILL__"
        elif len(data) == 2 and remove_keys[0] in data.keys() and remove_keys[1] in data.keys():
            return "__FILL__"

    elif isinstance(data, list):
        return "__FILL__"

    return data

def remove_duplicate_keys(data, replace_with, parent_key=None):
    if isinstance(data, dict):
        # Recursively apply to all child values first
        for key in list(data.keys()):
            data[key] = remove_duplicate_keys(data[key], replace_with, key)

        # Collapse if only one key and it's the same as parent
        for key in list(data.keys()):
            if key == parent_key:
                val = data[key]
                del data[key]
                data[replace_with] = val

    return data


def clean_with_llm(content: str) -> str:
    prompt_old = (
        f"""Wandle die folgende HTML-Vorlage in eine sehr prägnante und saubere JSON-Vorlage um.
        Werte im verschachtelten JSON sollen immer entweder eine Liste, ein Objekt oder ein String mit dem Inhalt „__FILL__“ sein.
        Lasse Metadaten, Kommentare und andere unnötige Informationen weg.
        Verwende keine Schlüssel, die für die Vorlage nicht notwendig sind.
        JEDES WORT MUSS DEUTSCH SEIN. Gib keine Beispielwerte an.
        Für jeden auszufüllenden Wert soll es nur ein einzelnes Schlüssel-Wert-Paar geben, wobei der Wert der Platzhalter „__FILL__“ ist.
        Achte darauf in der JSON-Vorlage die Hierarchie der HTML-Vorlage korrekt wiederzuspiegeln! Der Schlüsseltext soll unverändert bleiben und nicht formattiert werden.
        Die Ausgabe soll ein gültiges JSON-Objekt sein, ohne zusätzlichen Text oder Kommentare. Achte auf die korrekte JSON-Syntax!

        Hier ist der Inhalt:"
        {content}"""
    )

    prompt_v2 = (
        f"""Wandle die folgende HTML-Vorlage in eine sehr prägnante und saubere JSON-Vorlage um.
        Werte im verschachtelten JSON sollen immer entweder eine Liste, ein Objekt oder ein String mit dem Inhalt „__FILL__“ sein.
        Lasse Metadaten, Kommentare und andere unnötige Informationen weg.
        Verwende keine Schlüssel, die für die Vorlage nicht notwendig sind.
        JEDES WORT MUSS DEUTSCH SEIN. Gib keine Beispielwerte an.
        Für jeden auszufüllenden Wert soll es nur ein einzelnes Schlüssel-Wert-Paar geben, wobei der Wert der Platzhalter „__FILL__“ ist.
        Achte darauf in der JSON-Vorlage die Hierarchie der HTML-Vorlage korrekt wiederzuspiegeln! Der Schlüsseltext soll unverändert bleiben und nicht formattiert werden.
        Bei mehreren Auswahlmöglichkeiten für den Wert eines Schlüssels sollen diese zusätzlich in mit dem Schlüssel „<schlüssel>_möglichkeiten" auf der selben Ebene gespeichert werden.
        Die Ausgabe soll ein gültiges JSON-Objekt sein, ohne zusätzlichen Text oder Kommentare. Achte auf die korrekte JSON-Syntax!

        Hier ist der Inhalt:"
        {content}"""
    )

    prompt_v3 = (
        f"""Wandle die folgende HTML-Vorlage in eine sehr prägnante und saubere JSON-Vorlage um.
        Werte im verschachtelten JSON sollen immer entweder eine Liste, ein Objekt oder ein String mit dem Inhalt „__FILL__“ sein.
        Lasse Metadaten, Kommentare und andere unnötige Informationen weg.
        Verwende keine Schlüssel, die für die Vorlage nicht notwendig sind.
        JEDES WORT MUSS DEUTSCH SEIN. Gib keine Beispielwerte an.
        Für jeden auszufüllenden Wert soll es nur ein einzelnes Schlüssel-Wert-Paar geben, wobei der Wert der Platzhalter „__FILL__“ ist.
        Achte darauf in der JSON-Vorlage die Hierarchie der HTML-Vorlage korrekt wiederzuspiegeln! Der Schlüsseltext soll unverändert bleiben und nicht formattiert werden.
        Bei mehreren Auswahlmöglichkeiten für den Wert eines Schlüssels sollen diese Werte ignoriert werden. Fokussiere dich auf die Struktur und die Schlüssel.
        Die Ausgabe soll ein gültiges JSON-Objekt sein, ohne zusätzlichen Text oder Kommentare. Achte auf die korrekte JSON-Syntax!

        Hier ist der Inhalt:"
        {content}"""
    )

    response = vertex_model.generate_content(prompt_v3, generation_config=generation_config)
    print(response.text.strip())
    matches = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
    if matches:
        return json.loads(matches.group(0))

for file in SOURCE.glob("*.json"):
    print(f"Processing JSON file: {file.name}")
    try:
        orig_path = DEST_DIR_DE / file.name.lower()
        content = file.read_text(encoding="utf-8")
        content = json.loads(content)

        default_notes = "Sonstiges"

        content = remove_suffix(content, ":")
        content = remove_suffix(content, " =")
        content = remove_keys_with_suffix(content, "_möglichkeiten")
        content = rename_keys_with_suffix_and_move_to_end(content, "textarea", default_notes, default_notes)
        content = rename_keys_with_suffix_and_move_to_end(content, "select", "Details", default_notes)
        content = rename_keys_with_suffix_and_move_to_end(content, "input", default_notes, default_notes)
        content = remove_duplicate_keys(content, "Details", default_notes)
        content = collapse_single_key_dicts(content, ["Details", default_notes])

        content = json.dumps(content, indent=2, ensure_ascii=False)

        with orig_path.open("w", encoding="utf-8") as out:
            out.write(content)

        print(f"Wrote {orig_path}")
    except Exception as e:
        logger.error(f"Error processing {file.name}: {e}")


for file in SOURCE.glob("*.html"):
    print(f"Processing HTML file: {file.name}")
    try:
        orig_path = DEST_DIR_DE / file.name.lower().replace('.html', '.json')
        content = file.read_text(encoding="utf-8")

        if file.name.endswith('.html') or file.name.endswith('.htm'):
            content = clean_html(content)
        #    content = html_to_json_convert(content)
        #    content = decode_unicode_json(content)
        #    content = clean_json(content)
        #    content = transform_th_td(content)
        #    content = clean_json_new(content)
    #     content = extract_all_tables(content)
            content = clean_with_llm(content)
            content = json.dumps(content, indent=2, ensure_ascii=False)

            with orig_path.open("w", encoding="utf-8") as out:
                out.write(content)

        print(f"Wrote {orig_path}")
    except Exception as e:
        logger.error(f"Error processing {file.name}: {e}")