# -*- coding: utf-8 -*-
from reader.csvreader import CSVReader
from tokenization.tokenizer import SimpleGermanTokenizer
from tokenization.tokenizer import TextBlobDeTokenizer
from nltk.corpus import stopwords

import re

def tokenize_code(code, prefix):
    code = re.sub(r'[^\w\s]','',code)
    return [prefix + '_' + code[0:x] for x in reversed(range(1,len(code)+1))]

def tokenize_and_output(csv_filename, tokenizer, output_filename, key_of_code,
                         key_of_description, vocab, delimiter, code_prefix,
                          use_description=True, stop_words=None):
    reader = CSVReader(csv_filename, delimiter)
    dataset = reader.read_from_file()
    try:
        #os.remove(output_filename)
        pass
    except OSError:
        pass  
    

    with open(output_filename, 'w+') as out_file:
        for record in dataset:
            tokenized_record = tokenizer.tokenize(record[key_of_description]) if use_description else []
            if stop_words != None:
                tokenized_record = [w for w in tokenized_record if w.lower() not in stop_words]
                
            tokenized_record = tokenize_code(record[key_of_code], code_prefix) + tokenized_record 
            output_line = " ".join(tokenized_record)
            vocab.update(tokenized_record)
            print(output_line, file=out_file) 
            #print(output_line)
    
    
def combine_files(files, big_file):
    with open(big_file, 'w+') as big_file:
        for file_name in files:
            big_file.write(open(file_name).read())
            
def output_vocab(vocab_filename, vocab):
    with open(vocab_filename, 'w+') as out_file:
        for word in vocab:
            print(word, file=out_file)
    
def tokenize_catalogs(config):
    if config['use-textblob-de']:
        tokenizer = TextBlobDeTokenizer()
    else:
        tokenizer = SimpleGermanTokenizer(config['tokenizer-german-split-compound-words'])

    vocab_de = set()
    # You have to install the stopwords corpus by executing nltk.download()
    # and install Corpora -> stopwords
    stop_words = stopwords.words('german')
    tokenize_and_output(config['drg-catalog'], tokenizer, config['drg-tokenizations'],
                         'code', 'text_de', vocab_de, ',', 'DRG',
                          config['use-descriptions'], stop_words)
    tokenize_and_output(config['chop-catalog'], tokenizer, config['chop-tokenizations'], 
                        'code', 'text_de', vocab_de, ',', 'CHOP',
                         config['use-descriptions'], stop_words)
    tokenize_and_output(config['icd-catalog'], tokenizer, config['icd-tokenizations'], 
                        'code', 'text_de', vocab_de, ',', 
                        'ICD', config['use-descriptions'], stop_words)
    combine_files([config['drg-tokenizations'], config['chop-tokenizations'], config['icd-tokenizations']],  config['all-tokens'])
    output_vocab(config['all-vocab'], vocab_de)
