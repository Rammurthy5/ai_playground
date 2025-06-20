from transformers import pipeline
# Question answering

qa = pipeline("question-answering")

result = qa({
    'question': "Where is Hugging Face based?",
    'context': "Hugging Face Inc. is based in New York City."
})
print(result)

# Named- Entity recognition
ner = pipeline("ner", grouped_entities=True)
text = "Hugging Face Inc. is based in New York City."

result = ner(text)
print(result)

