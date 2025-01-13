import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 모델과 토크나이저를 로드합니다.
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def summarize(text, max_new_tokens=150, num_beams=4):
    inputs = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {key: val.to(model.device) for key, val in inputs.items()}  # 입력 텐서를 모델과 동일한 장치로 이동
    summary_ids = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens, 
        num_beams=num_beams, 
        length_penalty=2.0, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 요약할 텍스트
text = """
tetsetsetsetse
"""

# 요약 결과 출력
summary = summarize(text)
print("Summary:", summary)