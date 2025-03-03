import csv
import json

def collect_stats(file_path):
    stats = {
        "total_terms": 0,
        "total_defs": 0,
        "total_example_sentences": 0,
        "total_translation_sentences": 0,
        "total_term_length": 0,
        "total_def_length": 0,
        "total_example_sentence_length": 0,
        "total_translation_sentence_length": 0
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        
        for row in reader:
            if len(row) < 3:
                print(row)
                continue  # Skip malformed rows
            
            term, definition, examples = row[0], row[1], row[2]
            stats["total_terms"] += 1
            stats["total_defs"] += 1
            stats["total_term_length"] += len(term) # len(term.split()) for russian
            stats["total_def_length"] += len(definition.split())
            
            try:
                examples_list = json.loads(examples)
                for example in examples_list:
                    if "sentence" in example:
                        stats["total_example_sentences"] += 1
                        stats["total_example_sentence_length"] += len(example["sentence"]) # len(example["sentence"].split()) for russian
                    if "translation" in example:
                        stats["total_translation_sentences"] += 1
                        stats["total_translation_sentence_length"] += len(example["translation"].split())
            except json.JSONDecodeError:
                print("error")
                continue  # Skip malformed JSON data
    
    # Compute averages
    stats["avg_term_length"] = stats["total_term_length"] / stats["total_terms"] if stats["total_terms"] else 0
    stats["avg_def_length"] = stats["total_def_length"] / stats["total_defs"] if stats["total_defs"] else 0
    stats["avg_example_sentence_length"] = stats["total_example_sentence_length"] / stats["total_example_sentences"] if stats["total_example_sentences"] else 0
    stats["avg_translation_sentence_length"] = stats["total_translation_sentence_length"] / stats["total_translation_sentences"] if stats["total_translation_sentences"] else 0
    
    return stats

# Example usage
file_path = "Mandarin/structured_mandarin_dataset.csv"
stats = collect_stats(file_path)
print(stats)
