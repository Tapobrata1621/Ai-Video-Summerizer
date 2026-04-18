import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

def summarize_text(text, num_sentences=5):
    if not text or len(text.strip()) == 0:
        return "No text available for summarization."

    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())

    word_frequencies = defaultdict(int)

    for word in words:
        if word not in stop_words and word not in string.punctuation:
            word_frequencies[word] += 1

    if not word_frequencies:
        return "Could not generate summary."

    sentence_scores = defaultdict(int)

    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        for word in sentence_words:
            if word in word_frequencies:
                sentence_scores[sentence] += word_frequencies[word]

    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    selected_sentences = ranked_sentences[:num_sentences]

    ordered_summary = [sentence for sentence in sentences if sentence in selected_sentences]

    return " ".join(ordered_summary)