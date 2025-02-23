
import time
import requests
from bs4 import BeautifulSoup
import csv

def get_data(file):
    with open(file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        data = []

        word_set = set()

        for line in lines[1:]:
            Word, url = line.split(",")
            Word = Word.strip()
            url = url.strip()

            print(f"Scraping page: {Word}")

            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
                }

                response = requests.get(url, headers=headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html5lib')

                word_area = soup.find_all('div', id='tdEntry')

                if not word_area:
                    print("No words found on page:", Word)
                    continue

                for element in word_area:
                    if hasattr(element, 'find_all'):
                        definitions = element.find_all('div', class_='definition')
                        for definition in definitions:
                            if hasattr(definition, 'find_all'):
                                defin = definition.find('div', class_="def").get_text(strip=True)
                                examples = definition.find_all('div', class_="example")
                                
                                # If no examples are found, create an entry with just the definition
                                if(Word in word_set):
                                    Word = None
                                else:
                                    word_set.add(Word)
                                if not examples:
                                    data.append({
                                        "Word": Word,
                                        "Definition": defin,
                                        "Direct Translation": None,
                                        "Definition Translation": None,
                                        "Example Sentence": None,
                                        "Example Sentence Translation": None
                                    })
                                
                                chinese_list = []
                                english_list = []

                                for example in examples:
                                    
                                    if hasattr(example, 'find_all'):
                                        chinese_list.append(example.find('div', class_="cn").get_text(strip=True))
                                        english_list.append(example.find('div', class_="trad").get_text(strip=True))

                                data.append({
                                    "Word": Word,
                                    "Definition": defin,
                                    "Direct Translation": None,
                                    "Definition Translation": None,
                                    "Example Sentence": chinese_list,
                                    "Example Sentence Translation": english_list
                                })

            except Exception as e:
                print(f"Exception: {e}")
                continue

        return data

def main():
    start_time = time.time()

    word_file = "chinese_words_list.csv"

    data = get_data(word_file)

    csv_file_path = "chinese_db_v3.csv"

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Word", "Definition", "Direct Translation", "Definition Translation", "Example Sentence", "Example Sentence Translation"])
        writer.writeheader()  # Write column headers
        writer.writerows(data)  # Write data rows

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total duration: {total_duration} seconds")

if __name__ == '__main__':
    main()

