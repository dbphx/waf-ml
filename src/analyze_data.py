import pandas as pd
import os

data_path = "/Users/dmac/Desktop/ml/data/attack.csv"
df = pd.read_csv(data_path, sep=';')

print(f"Total rows: {len(df)}")
# Check rule_ids column
if 'rule_ids' in df.columns:
    print("\nRule ID distribution (Top 10):")
    print(df['rule_ids'].value_counts().head(10))
    
    # Rows with empty rules
    empty_rules = len(df[df['rule_ids'] == '[]'])
    print(f"\nRows with rule_ids == '[]': {empty_rules} ({empty_rules/len(df)*100:.2f}%)")

if 'rule_names' in df.columns:
    print("\nRule Names distribution (Top 10):")
    print(df['rule_names'].value_counts().head(10))

# Check some samples of [] rules
print("\nSamples with [] rules:")
print(df[df['rule_ids'] == '[]'][['http_method', 'http_path', 'http_query']].head())
