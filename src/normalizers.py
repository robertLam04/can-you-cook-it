import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def lowercasing(tokens):
    lower_tokens = []
    for token in tokens:
        lower_tokens.append(token.lower())
    return lower_tokens

def lemmatize(tokens):

    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        elif treebank_tag.startswith('s'):
            return wordnet.ADJ_SAT
        else:
            return wordnet.NOUN

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []

    tagged_tokens = nltk.tag.pos_tag(tokens)

    for token, tag in tagged_tokens:
        wordnet_pos = get_wordnet_pos(tag) 
        lemmatized_tokens.append(lemmatizer.lemmatize(token, wordnet_pos))

    return lemmatized_tokens

def remove_stopwords(tokens):
    new_tokens = []
    custom_stop_words = set(["youre", "youll", "ive", "tell", "let", "see", "take", "ill", "wanna", "right", "one", "bill", "since", "tit", "like", "well", "could", "might", "way", "gonna", "thing", "make", "also", "u", "hey", "come", "know", "get", "say", "im", "yeah", "ooh", "dont", "cant", "oh", "cause", "ah", "im"])
    stop_words = set(stopwords.words('english')) | custom_stop_words
    for token in tokens:
        if token not in stop_words:
            new_tokens.append(token)
    return new_tokens

def remove_proper_nouns(tokens):
    tagged_tokens = nltk.tag.pos_tag(tokens)
    new_tokens = []
    for (token, tag) in tagged_tokens:
        if tag != "NNP":
            new_tokens.append(token)
    return new_tokens

def remove_numbers(tokens):
    tokens = [token for token in tokens if not token.isdigit()]
    return tokens