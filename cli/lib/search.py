import string

from nltk.stem.porter import PorterStemmer

from .utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

_stop_words = load_stopwords()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    stemmer = PorterStemmer()
    query_tokens = tokenize_text(query, stemmer)

    for movie in movies:
        if has_matching_token(query_tokens, tokenize_text(movie["title"], stemmer)):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token == title_token:
                return True
    return False


def tokenize_text(text: str, stemmer: PorterStemmer | None = None) -> list[str]:
    if not stemmer:
        stemmer = PorterStemmer()

    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []

    for token in tokens:
        if token not in _stop_words:
            stemmed = stemmer.stem(token)
            valid_tokens.append(stemmed)

    return valid_tokens


def preprocess_text(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))
