import pandas as pd
import json

# Read the CSV file
df = pd.read_csv('cleaned_chinese_db_v5.csv')
df.fillna("", inplace=True)  # Replace NaN with empty strings
df['Example Sentence'] = df['Example Sentence'].astype('str')
df['Example Sentence Translation'] = df['Example Sentence Translation'].astype('str')

# Initialize a dictionary to structure data
structured_data = {}
current_word = None  # Keep track of the last valid word

# Iterate through the dataframe to build the nested structure
for _, row in df.iterrows():
    word = row['Word'].strip()
    definition = row['Definition'].strip()
    example_sentence = row['Example Sentence']
    example_translation = row['Example Sentence Translation']
    
    if word.lower() != "nan":  # New word entry
        current_word = word
        if current_word not in structured_data:
            structured_data[current_word] = {'definitions': {}}
    
    if current_word and definition:  # Ensure there is a valid definition
        if definition not in structured_data[current_word]['definitions']:
            structured_data[current_word]['definitions'][definition] = {'examples': []}
    
        structured_data[current_word]['definitions'][definition]['examples'].append({
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

# Save to a new CSV file
structured_df = pd.DataFrame(entries)
structured_df.to_csv('structured_chinese_db2.csv', index=False, encoding='utf-8')
