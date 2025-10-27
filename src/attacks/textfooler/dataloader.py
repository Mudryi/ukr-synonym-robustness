import pandas as pd
import random
import re

def tokenize_ukrainian(text):
    pattern = r"(\s+|[^\w\s']+|[\w']+)"
    tokens = re.findall(pattern, text)
    
    final_tokens = []
    i = 0
    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and tokens[i].isalpha()
            and tokens[i+1] == "'"
            and tokens[i+2].isalpha()
        ):
            final_tokens.append(tokens[i] + tokens[i+1] + tokens[i+2])
            i += 3
        else:
            final_tokens.append(tokens[i])
            i += 1
    return final_tokens


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def read_corpus(path, dataset_name, data_size=10000, encoding='utf8'):
    df = pd.read_csv(path, encoding=encoding)
    if len(df)>data_size:
        df = df.sample(data_size, random_state=1914)
    
    data = []
    labels = []

    if dataset_name == 'reviews':
        for _, row in df.iterrows():
            text = row['text']
            label = int(row['label'])-1

            labels.append(label)
            data.append(tokenize_ukrainian(text))

    elif dataset_name == 'news':
        label_list = ['бізнес', 'новини', 'політика', 'спорт', 'технології']
        label2id = {label: i for i, label in enumerate(label_list)}
        df['target'] = df['target'].map(label2id)
        
        for _, row in df.iterrows():
            text = row['title']
            label = int(row['target'])

            labels.append(label)
            data.append(tokenize_ukrainian(text))

    elif dataset_name == 'unlp':
        for _, row in df.iterrows():
            text = row['text']
            label = int(row['label'])

            labels.append(label)
            data.append(tokenize_ukrainian(text))

    return data, labels