from transformers import pipeline
classifier = pipeline('sentiment-analysis')
print(classifier(['We are very happy to show you the 🤗 Transformers library.']))