import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load PhoBERT model for sentiment analysis
model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels: pos, neg, neu

# Sentiment Mapping
sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return sentiment_labels[predicted_class]
