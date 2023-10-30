import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from gensim.models import Word2Vec
import warnings

# Download NLTK resources if not already downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Sample corpus to train Word2Vec model (you can use a larger corpus for better results)
corpus = [
    "king is a strong ruler",
    "queen is a wise leader",
    "man is strong",
    "woman is wise",
    "boy is young",
    "girl is intelligent"
]

# Tokenize the corpus
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, sg=0, min_count=1)

# Function for preprocessing, context derivation, and ambiguity resolution
def preprocess_resolve_ambiguity(text, word):
    # Lowercasing and tokenization
    lowercased_text = text.lower()
    tokens = word_tokenize(lowercased_text)
    filtered_tokens = [token for token in tokens if token.isalnum()]
    
    # Ambiguity Resolution using WordNet
    resolved_tokens = []
    for token in filtered_tokens:
        synsets = wordnet.synsets(token)
        if synsets:
            # Choose the synset with the most frequent sense (first one in the list)
            resolved_tokens.append(synsets[0].lemmas()[0].name())
        else:
            resolved_tokens.append(token)  # Keep the token unchanged if no synsets found
    
    # Context Derivation using Word2Vec
    context_derived_tokens = []
    for i, token in enumerate(resolved_tokens):
        if token == word and token in model.wv:
            # Consider the surrounding words for context derivation
            context_words = resolved_tokens[max(0, i - 2): i] + resolved_tokens[i + 1: min(i + 3, len(resolved_tokens))]
            context_similar_words = model.wv.most_similar(context_words, topn=5)
            context_derived_tokens.extend([word for word, _ in context_similar_words])
        else:
            context_derived_tokens.append(token)
    
    return context_derived_tokens

# Read input from input.txt
with open("input.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

# Specify the word for which you want to find synonyms
word_to_resolve_ambiguity = "ruler"  # Change this word to resolve ambiguity for a different word

# Preprocess, resolve ambiguity, and derive context
processed_text = preprocess_resolve_ambiguity(input_text, word_to_resolve_ambiguity)

# Output the results
print(f"Original Text: {input_text}")
print(f"Processed Text: {' '.join(processed_text)}")
