import pandas as pd
import json

with open('final_candidate_data.json', 'r') as file:
    data = json.load(file)
comments = [item[1][0] for item in data]
labels = [item[1][1] for item in data]
df = pd.DataFrame({'comment':comments, 'label':labels})
df['label'] = df['label'].astype(int)
df.to_csv('final_candidate_data.csv', index=False)