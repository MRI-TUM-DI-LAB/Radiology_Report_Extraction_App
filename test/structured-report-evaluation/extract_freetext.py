from bs4 import BeautifulSoup
import os

def extract_all_free_texts(html_content, output_dir='free_text'):
    soup = BeautifulSoup(html_content, 'html.parser')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    category_counter = {}

    # Get all <h1> headers which define each examination category
    headers = soup.find_all('h1')

    for header in headers:
        category = header.get_text(strip=True).replace(" ", "_").upper()

        if category not in category_counter:
            category_counter[category] = 1

        # Go through the following sibling elements (e.g., <table>, <tr>)
        next_elem = header.find_next_sibling()
        while next_elem:
            if next_elem.name == 'h1':
                break

            # Find all "Free Text" rows in the current <table>
            if next_elem.name == 'table':
                free_text_tds = next_elem.find_all('td', string=lambda x: x and 'Free Text' in x)
                for free_text_td in free_text_tds:
                    # Get the adjacent td with the actual content
                    content_td = free_text_td.find_next_sibling('td')
                    if content_td:
                        free_text = content_td.get_text(separator='\n', strip=True)

                        # Generate filename with count
                        index = category_counter[category]
                        filename = f"{category}_{index}.txt"
                        filepath = os.path.join(output_dir, filename)

                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(free_text)
                        print(f"Saved: {filepath}")

                        category_counter[category] += 1
            next_elem = next_elem.find_next_sibling()

    print("\n=== Summary Statistics ===")
    total_categories = len(category_counter)
    total_free_texts = sum(category_counter.values()) - total_categories  # subtract initial 1 from each

    print(f"Total categories: {total_categories}")
    print(f"Total free texts: {total_free_texts}")
    print("\nFree texts per category:")
    for category, count in category_counter.items():
        print(f"· {category}: {count - 1} Free Text(s)")

if __name__ == '__main__':
    with open("KenoTemplates.html", "r", encoding="utf-8") as f:
        html_data = f.read()

    # Extract and save Free Texts
    extract_all_free_texts(html_data)

