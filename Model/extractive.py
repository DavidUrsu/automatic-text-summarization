import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
import pandas as pd
from heapq import nlargest

def generate_title(text):

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    tokens = [token.text.lower() for token  in doc
          if not token.is_stop and
          not token.is_punct and
          token.text != '\n']

    tokens1 = []
    stopwords = list(STOP_WORDS)
    allowed_pos = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    for token in doc:
        if token.text in stopwords or token.text in punctuation:
            continue
        if token.pos_ in allowed_pos:
            tokens1.append(token.text)

    word_freq = Counter(tokens1)

    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = (word_freq[word]/max_freq)
    sent_token = [sent.text for sent in doc.sents]

    sent_score = {}
    for sent in sent_token:
        for word in sent.split():
            if word.lower() in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = word_freq[word.lower()]
                else:
                    sent_score[sent] += word_freq[word.lower()]

    pd.DataFrame(sent_score.items(), columns=['Sentence', 'Score'])
    num_sentences = 1
    n = nlargest(num_sentences, sent_score, key=sent_score.get)
    return " ".join(n)



