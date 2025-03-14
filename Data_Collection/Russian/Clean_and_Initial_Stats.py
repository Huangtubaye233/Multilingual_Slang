'''
1) Total number of entries
2) Number of unique terms (i.e., words/phrases)
3) Average length of definition sentences (# of words in each sentence)
4) (For Chinese) Average number of characters in each Chinese term


Cyrillic: \u0400-\u04FF
Chinese: \u4E00-\u9FFF
Devanagari (e.g., Hindi): \u0900-\u097F
Arabic: \u0600-\u06FF
non_latin_pattern = r'[^\u0000-\u007F]'

cyrillic_pattern = r'[\u0400-\u04FF]'
if re.search(cyrillic_pattern, field_value):
'''
import csv
import re

def is_proper_format(line):
    cyrillic_pattern = r'[\u0400-\u04FF]'
    try:
        if re.search(cyrillic_pattern, line[0]) and not re.search(cyrillic_pattern, line[1]) and re.search(cyrillic_pattern, line[-2]) and not re.search(cyrillic_pattern, line[-1]):
            return True
    except Exception as e:
        print(f"Exception: {e}")
        print(line)
    return False


def get_stats(file):

    total_entries = 0
    terms = set()
    definition_length = 0
    characters_length = 0

    with open('cleaned_russian_db.csv', 'w', encoding='utf-8') as output_file:
        with open(file, mode='r', encoding='utf-8') as input_file:

            lines = input_file.readlines()
            for line in lines:
                line_data = line.split(',')
                if(is_proper_format(line_data)):
                    total_entries += 1
                    terms.add(line_data[0])
                    definition_length += len(line_data[-1].split(" "))
                    characters_length += len(list(line_data[0]))

                    output_file.write(line)
                else:
                    print("Bad data found", line_data)
    
    print("Total number of entries: " + str(total_entries))
    print("Total number of terms: " + str(len(terms)))
    print("Average definition length: " + str(float(definition_length)/float(total_entries)))
    print("Average term length: " + str(float(characters_length)/float(total_entries)))

    return




def main():

    file = "russian_words_list.csv"

    get_stats(file)


if __name__ == '__main__':
    main()
