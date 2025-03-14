import csv

# def clean():
#     cleaned_rows = []
#     seen_entries = set()
    
#     with open('russian_words_list.csv', mode='r', encoding='utf-8') as input_file:
#         reader = csv.DictReader(input_file)
        
#         for row in reader:
#             word = row['Word']
#             pos = row['POS']
#             eng_def = row['Eng_Definition']
#             rus_def = row['Rus_Definition']
#             Rus_Example = row['Rus_Example']
#             eng_example = row['Eng_Example']
                                                          
#             # Ensure the essential fields are not null
#             if not eng_def or not rus_example or not eng_example:
#                 continue
            
#             # Split definitions if they have leading numbers "1." or "2."
#             definitions = eng_def.split(' 1.') if ' 1.' in eng_def else [eng_def]
#             cleaned_definitions = []
            
#             for definition in definitions:
#                 parts = definition.split(' 2.') if ' 2.' in definition else [definition]
#                 cleaned_definitions.extend(parts)
            
#             for cleaned_def in cleaned_definitions:
#                 cleaned_def = cleaned_def.strip()
                
#                 # Remove leading numbering
#                 if cleaned_def.startswith(('1.', '2.')):
#                     cleaned_def = cleaned_def[2:].strip()
                
#                 entry = (word, pos, cleaned_def, rus_def, rus_example, eng_example)
                
#                 # Remove duplicate values in any position
#                 if entry not in seen_entries:
#                     seen_entries.add(entry)
#                 cleaned_rows.append(entry)
    
#     # Write to output file
#     with open('cleaned_russian_db_v2.csv', 'w', encoding='utf-8', newline='') as output_file:
#         fieldnames = ['ord', 'POS', 'Eng_Definition', 'Rus_Definition', 'Rus_Example', 'Eng_Example']
#         writer = csv.writer(output_file)
#         writer.writerow(fieldnames)  # Write header
#         writer.writerows(cleaned_rows)
import pandas as pd
import json


def to_dataframe():
    
    # Read the CSV file
    df = pd.read_csv('cleaned_russian_db_v2.csv')
    df['Rus_Example'] = df['Rus_Example'].astype('str')
    df['Eng_Example'] = df['Eng_Example'].astype('str')

    # Initialize a dictionary to structure data
    structured_data = {}

    # Iterate through the dataframe to build the nested structure
    for _, row in df.iterrows():
        word = row['Word']
        definition = row['Eng_Definition']
        example_sentence = row['Rus_Example']
        example_translation = row['Eng_Example']
        
        if word not in structured_data:
            structured_data[word] = {'definitions': {}}
        
        if definition not in structured_data[word]['definitions']:
            structured_data[word]['definitions'][definition] = {'examples': []}
        
        structured_data[word]['definitions'][definition]['examples'].append({
            'sentence': example_sentence,
            'translation': example_translation
        })

    # Convert structured data to a DataFrame for easier export
    entries = []
    for word, word_data in structured_data.items():
        for definition, def_data in word_data['definitions'].items():
            entries.append({
                'Word': word,
                'Definition': definition,
                'Examples': json.dumps(def_data['examples'], ensure_ascii=False)
            })

    structured_df = pd.DataFrame(entries)

    # Save to a new CSV file
    structured_df.to_csv('structured_russian_db.csv', index=False, encoding='utf-8')


def main():
    # clean()
    to_dataframe()

if __name__ == '__main__':
    main()
