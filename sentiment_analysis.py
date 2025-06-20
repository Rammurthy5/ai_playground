# approach 1 for sentiment analysis using pre-trained model on sentiment analysis. high accuracy, prod-ready
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love using BERT! It's incredibly powerful.")
print(result)

#approach 2 for sentiment analysis using RAW model which needs training, customization if req. 
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification
text = "I love this movie, it's amazing!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(predictions, dim=1).item()
sentiment = "Positive" if predicted_class == 1 else "Negative"
print(f"Text: {text}")
print(f"Sentiment: {sentiment} (Confidence: {predictions[0][predicted_class]:.2f})")
