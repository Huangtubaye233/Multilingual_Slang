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

re.compile(r'[0-9]+/.+ +(.*)[0-9]+\.+.*')


'''
import csv
import re
import json
def process_as_list(value):
    """
    Converts a comma-separated string into a list of items.
    Removes brackets and trims whitespace around items.
    """
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]  # Remove surrounding brackets
    return [item.strip() for item in value.split(",") if item.strip()]  # Split into a list


def is_proper_format(line):
    chinese_pattern = r'[\u4E00-\u9FFF]'  # Regex for Chinese characters
    try:
        # Check if the first field exists and contains Chinese characters
        if not line[0] or re.search(chinese_pattern, line[0]):
            # Check if the second field does not contain Chinese characters
            if not re.search(chinese_pattern, line[1]):
                # Convert the last two fields into lists
                for idx in [-2, -1]:
                    line[idx] = process_as_list(line[idx])

                # Verify that the second-to-last field contains Chinese characters
                for i in line[-2]:
                    if not re.search(chinese_pattern, i):
                        return False

                # Verify that the last field does not contain Chinese characters
                for i in line[-1]:
                    if re.search(chinese_pattern, i):
                        return False

                return True
    except Exception as e:
        print(f"Exception: {e}")
        print(f"Line causing error: {line}")
    return False


# Function to calculate and print statistics
def get_stats(file):
    total_entries = 0
    terms = set()
    definition_length = 0
    characters_length = 0

    with open('cleaned_chinese_db_v3.csv', 'w', encoding='utf-8') as output_file:
        with open(file, mode='r', encoding='utf-8') as input_file:
            # Read the CSV file and process each line
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)
            
            header = next(reader)  # Read header row
            writer.writerow(header)  # Write header to the output file

            for line in reader:
                if is_proper_format(line):
                    total_entries += 1
                    if(line[0]):
                        terms.add(line[0])  # Add term to the set
                    definition_length += len(line[1].split())  # Count words in the definition
                    characters_length += len(line[0])  # Count characters in the term
                    print(line[-1][-1])
                    
                    writer.writerow(line)  # Write the valid line to the output file
                else:
                    print(f"Invalid data found: {line}")
    
    # Calculate averages
    avg_definition_length = definition_length / total_entries if terms else 0
    avg_term_length = characters_length / len(terms) if terms else 0

    # Print statistics
    print(f"Total number of entries: {total_entries}")
    print(f"Total number of unique terms: {len(terms)}")
    print(f"Average length of definition sentences: {avg_definition_length:.2f} words")
    print(f"Average number of characters in each term: {avg_term_length:.2f}")

def delete_lines(file):
    with open('cleaned_chinese_db_v4.csv', 'w', encoding='utf-8') as output_file:
        with open(file, mode='r', encoding='utf-8') as input_file:
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)
            
            header = next(reader)  # Read header row
            writer.writerow(header)  # Write header to the output file

            for line in reader:
                if(line):
                    writer.writerow(line)
                    



# Main function to execute the script
def main():
    file = "chinese_db_v3.csv"  # Input CSV file
    get_stats(file)
    # delete_lines("cleaned_chinese_db_v3.csv")

if __name__ == '__main__':
    main()
