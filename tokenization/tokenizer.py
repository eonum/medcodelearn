# -*- coding: utf-8 -*-
import re
from nltk.stem.snowball import GermanStemmer
import os
import csv

class Tokenizer:
    def __init__(self):
        pass

    def split_to_words(self, s, delimiter=' '):
        s = re.sub(r'[^\w\s]','',s)
        return s.split(delimiter)
    
    def tokenize(self, sentence):
        words = self.split_to_words(sentence)
        return self.process_words(words)
       
class GermanTokenizer(Tokenizer):
    def __init__(self, split_compound_words=False):
        super()
        self.split_compound_words = split_compound_words
# Hack (Using a Java library). This is only a prototype. Should choose one language later.    
    def split_compound_words(self, words, basepath=''): 
        try:
            os.remove('/tmp/compound_words.tmp')
            os.remove('/tmp/split_compound_words.tmp')
        except OSError:
            pass        
        with open('/tmp/compound_words.tmp', 'w') as file:        
            for word in words:
                file.write("%s\n" % word)        
        os.system("java -jar "+basepath+"java/lib/jwordsplitter-4.1.jar /tmp/compound_words.tmp > /tmp/split_compound_words.tmp")
        split_compound_words = []        
        with open('/tmp/split_compound_words.tmp', 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=',')
            for row in reader:
                split_compound_words += row
        try:
            os.remove('/tmp/compound_words.tmp')
            os.remove('/tmp/split_compound_words.tmp')
        except OSError:
            pass  
        return split_compound_words
        
        
    def process_words(self, words):
        if self.split_compound_words:
            words  = self.split_compound_words(words) 
    
        stemmed_words = self.stem_words(words)
        
        return stemmed_words
        
    def stem_words(self, words):
        stemmer = GermanStemmer()
        stemmed_words = []        
        for word in words:
            stemmed_words.append(stemmer.stem(word))
        return stemmed_words
        
    


