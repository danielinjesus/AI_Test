from transformers import pipeline
summarizer = pipeline("summarization")
summary = summarizer("""

                     """, min_length=5, max_length=300)
print(summary)
