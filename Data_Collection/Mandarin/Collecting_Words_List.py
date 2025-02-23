import time
import requests
from bs4 import BeautifulSoup
import csv

alphabet = "ABCDEFGHJKLMNOPQRSTWXYZ"

def get_all_words(url):
    word_set = []
    for letter in alphabet:
        print(f"Scraping page: {letter}")

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }

            response = requests.get(url + f"?abc={letter}", headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html5lib')

            # Find all the brand cards
            word_area = soup.find_all('div', id='tdEntries')

            if not word_area:
                print("No words found on page:", letter)
                break

            for element in word_area:
                if hasattr(element, 'find_all'):
                    word_item = element.find_all('ul')
                    for words in word_item:
                        if hasattr(words, 'find_all'):
                            word = words.find_all('li')
                            for w in word:
                                if hasattr(w, 'find_all'):
                                   link = w.find_all('a')
                                   for a in link:
                                        addy = a.get('href')
                                        text = a.get_text(strip=True)
                                        word_set.append({'Word': text, 'Link': addy})
        


        except Exception as e:
            print(f"Exception: {e}")
            break

    print("done")
    print(word_set)
    return word_set

def main():
    start_time = time.time()

    dict_url = 'https://www.chinese-tools.com/chinese/slang/'
    words = get_all_words(dict_url)

    csv_file_path = "chinese_words_list.csv"

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Word', 'Link'])
        for data in words:
            writer.writerow([data['Word'], data['Link']])

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"Total duration: {total_duration} seconds")

if __name__ == '__main__':
    main()
