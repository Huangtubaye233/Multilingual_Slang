
import time
import requests
from bs4 import BeautifulSoup
import csv

alphabet = ["90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "9A", "9B", "9C", "9D", "9E", "9F", "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "AA", "AB", "AC", "AD", "AE", "AF"]

def get_all_words(url):
    word_set = []
    for letter in alphabet:
        print(f"Scraping page: {letter}")

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }

            response = requests.get(url + f"{letter}", headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html5lib')

            word_sections = soup.find_all('div', itemtype="http://schema.org/ScholarlyArticle")

            if not word_sections:
                print("No words found on page:", letter)
                continue

            for element in word_sections:
                if hasattr(element, 'find_all'):
                    word_item = element.find('p', class_="art")
                    if word_item:
                        word = word_item.find('span', class_='lem').get_text(strip=True) if word_item.find('span', class_='lem') else None
                        pos = word_item.find('span', class_='grm').get_text(strip=True) if word_item.find('span', class_='grm') else None

                    definitions = element.find_all('div', itemprop="articleBody")

                    for definition in definitions:
                        if hasattr(definition, "find_all"):
                            eng_definitions = [d.get_text(strip=True) for d in definition.find_all('p', class_="art2")]
                            examples = [e.get_text(strip=True) for e in definition.find_all('p', class_="cit")]
                            translations = [t.get_text(strip=True) for t in definition.find_all('p', class_="ctr")]

                            for i in range(len(eng_definitions)):
                                word_set.append({
                                    "Word": word,
                                    "Translation": None,
                                    "POS": pos,
                                    "Eng_Definition": eng_definitions[i] if i < len(eng_definitions) else None,
                                    "Rus_Definition": None,
                                    "Rus_Example": examples[i] if i < len(examples) else None,
                                    "Eng_Example": translations[i] if i < len(translations) else None
                                })

        except Exception as e:
            print(f"Exception: {e}")
            continue

    print("Scraping complete")
    return word_set

def main():
    start_time = time.time()

    dict_url = 'https://www.russki-mat.net/page.php?l=RuEn&a=%D0%'
    words = get_all_words(dict_url)

    csv_file_path = "russian_words_list.csv"

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Word", "Translation", "POS", "Eng_Definition", "Rus_Definition", "Rus_Example", "Eng_Example"])
        writer.writeheader()
        writer.writerows(words)

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total duration: {total_duration} seconds")

if __name__ == '__main__':
    main()
