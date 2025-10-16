import numpy as np
import pandas as pd

# Define Word class to handle loading of npy files containing Word objects
class Word:
    def __init__(self, word):
        self.word = word
        self.pos_tags = set()
        self.definitions = []

    def attach_def(self, word_def, pos, sentences):
        new_def = {'def': word_def, 'pos': pos, 'sents': sentences}
        self.pos_tags.add(pos)
        self.definitions.append(new_def)

def check_npy_file(file_path):
    """Check the structure and content of an npy file"""
    print(f"\n=== Checking {file_path} ===")
    
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"File shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        
        if len(data) > 0:
            print(f"First element type: {type(data[0])}")
            print(f"First element: {data[0]}")
            
            if len(data) > 1:
                print(f"Second element: {data[1]}")
            
            # Check if it contains Word objects
            if hasattr(data[0], 'word'):
                print("Contains Word objects")
                print(f"Sample Word object - word: '{data[0].word}'")
                print(f"Sample Word object - definitions: {data[0].definitions}")
            else:
                print("Contains regular data (not Word objects)")
                
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

if __name__ == "__main__":
    # Check both files
    check_npy_file('RU_ru_conv_data.npy')
