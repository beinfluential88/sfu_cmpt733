import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string

# Read the downloaded dataset and load it as a dataframe
spam_df = pd.read_csv('spam.csv', encoding="ISO-8859-1")

# Take the 2 columns we are interested in and give them meaningful names
spam_df = spam_df[['v1', 'v2']]
spam_df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

# Convert "spam/ham" to binary values (0 or 1)
spam_df['label'] = spam_df['label'].apply(lambda a: True if a=='spam' else False)

# lowercase everything and remove punctuation
table = str.maketrans('', '', string.punctuation)   # using maketrans() to construct a translate table
spam_df['text'] = spam_df['text'].apply(lambda t: t.lower().translate(table))

# shuffle records randomly
spam_df = spam_df.sample(frac=1)