# -*- coding: utf-8 -*-
from nltk.stem.snowball import GermanStemmer
import re

class Tokenizer:
    def __init__self():
        pass

    def split_to_words(self, s, delimiter=' '):
        s = re.sub(r'[^\w\s]','',s)
        return s.split(delimiter)
    
    def tokenize(self, sentence):
       words = self.split_to_words(sentence)
       return self.process_words(words)


# Move into own file later
class GermanTokenizer(Tokenizer):
    def __init__(self):
        super()
    
    def process_words(self, words):
        stemmed_words = self.stem_words(words)
        return stemmed_words
        
    def stem_words(self, words):
        stemmer = GermanStemmer()
        stemmed_words = []        
        for word in words:
            stemmed_words.append(stemmer.stem(word))
        return stemmed_words
        
        
        

tok = GermanTokenizer()
print(tok.tokenize("WIr gehen nach hause"))
