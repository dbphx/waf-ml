import pandas as pd
import numpy as np
import re
import urllib.parse
from sklearn.model_selection import train_test_split
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase all text
    text = text.lower()
    
    # 2. URL decode (2 passes to handle double encoding)
    try:
        text = urllib.parse.unquote(text)
        text = urllib.parse.unquote(text)
    except:
        pass
    
    # 3. Normalize whitespace but keep everything else
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_data(input_path, output_dir):
    """Legacy preprocessing function - replaced by standardize_data.py but kept for compatibility."""
    df = pd.read_csv(input_path)
    df['cleaned_text'] = df.apply(lambda x: clean_text(str(x)), axis=1)
    # This is a stub for the old pipeline
    pass

if __name__ == "__main__":
    pass
