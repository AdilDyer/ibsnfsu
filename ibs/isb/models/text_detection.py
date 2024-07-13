import torch
from transformers import pipeline

def is_generated_by_ai(paragraph):
    # Load the text classification pipeline
    text_classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

    # Classify the input paragraph
    result = text_classifier(paragraph)

    # You can adjust this threshold based on experimentation
    confidence_threshold = 0.5
    print(paragraph)
    # Check if the label is consistent with AI-generated text
    label = result[0]['label']
    confidence = result[0]['score']
    print(confidence)
    print(label)
    if confidence >= confidence_threshold:
        return True
    else:
        return False