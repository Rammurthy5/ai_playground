# example code to demonstrate text embeddings (semantic search or clustering)

from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

text = "BERT is powerful for NLP tasks."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# Get the embeddings (last hidden state)
outputs = model(**inputs)
embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
print(embeddings.shape)

