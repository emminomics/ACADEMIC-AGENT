"""
Tool 2: Text summarization and keyword extraction tool
"""

import re


def extract_key_sentences(text: str, num_sentences: int = 3) -> str:
    """Extract key sentences from text (extractive summarization)"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s) > 10]

    if len(sentences) <= num_sentences:
        return text

    selected = [sentences[0]]  # The first sentence usually contains the core idea
    if num_sentences >= 2:
        selected.append(sentences[len(sentences) // 2])
    if num_sentences >= 3:
        selected.append(sentences[-1])  # The last sentence usually contains the conclusion

    return " ".join(selected)


def extract_keywords(text: str, top_n: int = 10) -> list:
    """Extract keywords from text (frequency-based, with stopword filtering)"""
    stop_words = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
        "but", "is", "are", "was", "were", "be", "been", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "this", "that", "these", "those", "with", "by", "from", "as", "we",
        "our", "their", "it", "its", "which", "who", "also", "can", "such",
        "show", "using", "use", "used", "results", "paper", "propose",
        "proposed", "model", "method", "approach", "based", "than", "more"
    }
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    word_count = {}
    for word in words:
        if word not in stop_words:
            word_count[word] = word_count.get(word, 0) + 1

    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def summarize_text(text: str) -> dict:
    """Perform a full analysis of the text (summary + keywords)"""
    return {
        "summary": extract_key_sentences(text, num_sentences=3),
        "keywords": extract_keywords(text, top_n=8),
        "word_count": len(text.split())
    }
