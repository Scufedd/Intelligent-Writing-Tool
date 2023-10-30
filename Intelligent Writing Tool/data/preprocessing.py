import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources if not already downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm")

def NER(text_list):
    processed_text_list = []
    for text in text_list:
        doc = nlp(text)
        processed_text = []
        for token in doc:
            if token.ent_type_ not in ["PERSON", "ORG", "GPE"]:  # Remove PERSON, ORG, and GPE entities
                processed_text.append(token.text)
        processed_text_list.append(" ".join(processed_text))
    return processed_text_list


# Read input from input.txt
with open("data/pride_and_prejudice.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

lowercased_text = input_text.lower()
tokens = word_tokenize(lowercased_text)
tokens = [token for token in tokens if token.isalnum()]
stop_words = set(stopwords.words("english"))
filtered_tokens = [token for token in tokens if token not in stop_words]
filtered_tokens = NER(filtered_tokens)
#stemmer = PorterStemmer()
#stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
#lemmatizer = WordNetLemmatizer()
#lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

# Write preprocessed output to output.txt
with open("data/output_pride_and_prejudice.txt", "w", encoding="utf-8") as file:
    file.write("Filtered Tokens (Without Punctuation and Stopwords): " + str(filtered_tokens) + "\n")

print("Preprocessed data has been written to output.txt.")
