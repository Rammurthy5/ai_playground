# approach 1 using fb's bart-large-cnn model which is excellent
from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Long text input
text = """
The Amazon rainforest, also known as Amazonia, is a vast tropical rainforest in South America. It is the largest rainforest in the world and home to an unparalleled diversity of flora and fauna. Spanning over nine countries, the majority of the forest is contained within Brazil. The Amazon plays a critical role in regulating the Earth's oxygen and carbon cycles. However, it is under severe threat due to deforestation, illegal logging, and climate change. Efforts are underway globally to protect and preserve this vital ecosystem for future generations.
"""

# Summarize. here we can tweak max_length, do_sample, min_length
summary = summarizer(text, max_length=60, min_length=25, do_sample=False)
print("Summary:", summary[0]['summary_text'])


# approach 2 replacing it with "t5-base" or "t5-large" – general purpose text-to-text transformer
# "google/pegasus-cnn_dailymail" – very strong abstractive summarizer
# Just replace the model in the pipeline() like so:

summarizer = pipeline("summarization", model="google/pegasus-cnn_dailymail")


# approach 3 - summarising multi-page docs
from transformers import pipeline, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

# Download required tokenizer for sentence splitting
nltk.download("punkt")

# Load summarization model and tokenizer
model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example long text
long_text = """
The Amazon rainforest is one of the most biodiverse regions on Earth. It spans across multiple countries and plays a crucial role in the global climate system. 
However, deforestation caused by illegal logging and agricultural expansion poses a significant threat to this ecosystem. 
Scientists warn that continued destruction could lead to irreversible changes and massive biodiversity loss. 
Efforts to protect the rainforest have been increasing, including policies by governments and activism by NGOs. 
Technology is also being used to monitor deforestation in real-time, enabling faster intervention.
"""

# ======================
# METHOD 1: CHARACTER-BASED CHUNKING
# ======================
def summarize_with_char_chunks(text, max_chars=500):
    print("\n📝 Method 1: Character-Based Chunking")
    sentences = sent_tokenize(text)
    chunks, chunk = [], ""

    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_chars:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())

    summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarizer(chunk, max_length=100, min_length=25, do_sample=False)[0]['summary_text']
        print(f"\nChunk {i+1} Summary:\n{summary}")
        summaries.append(summary)

    if len(summaries) > 1:
        final_summary = summarizer(" ".join(summaries), max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        print("\n🧠 Final Summary:\n", final_summary)


# ======================
# METHOD 2: TOKEN-BASED CHUNKING
# ======================
def summarize_with_token_chunks(text, max_tokens=1024):
    print("\n📝 Method 2: Token-Based Chunking")
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_len = [], [], 0

    for sentence in sentences:
        tokenized = tokenizer.encode(sentence, add_special_tokens=False)
        if current_len + len(tokenized) <= max_tokens:
            current_chunk.append(sentence)
            current_len += len(tokenized)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = len(tokenized)
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarizer(chunk, max_length=100, min_length=25, do_sample=False)[0]['summary_text']
        print(f"\nChunk {i+1} Summary:\n{summary}")
        summaries.append(summary)

    if len(summaries) > 1:
        final_summary = summarizer(" ".join(summaries), max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        print("\n🧠 Final Summary:\n", final_summary)


# ======================
# Run Both Methods
# ======================
summarize_with_char_chunks(long_text)
summarize_with_token_chunks(long_text)

