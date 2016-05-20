# -*- coding: utf-8 -*-
from reader.csvreader import CSVReader
from tokenization.tokenizer import SimpleGermanTokenizer
from tokenization.tokenizer import TextBlobDeTokenizer
from nltk.corpus import stopwords

import re

def tokenize_code(code, prefix):
    code = re.sub(r'[^\w\s]','',code)
    return [prefix + '_' + code[0:x] for x in reversed(range(1,len(code)+1))]

def tokenize_and_output(csv_filename, tokenizers_by_key_of_description, output_filename, key_of_code,
                         keys_of_descriptions, vocab, delimiter, code_prefix,
                          use_descriptions=True, stop_words=None):
    reader = CSVReader(csv_filename, delimiter)
    dataset = reader.read_from_file()
    try:
        #os.remove(output_filename)
        pass
    except OSError:
        pass  
    
    with open(output_filename, 'w+') as out_file:
        for record in dataset:
            tokenized_record = []
            if use_descriptions:
                for key_of_description in keys_of_descriptions:
                    tokenizer = tokenizers_by_key_of_description[key_of_description]    
                    tokenized_record.extend(tokenizer.tokenize(record[key_of_description])) 
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
    keys_of_descriptions = []
    tokenizers_by_key_of_description = {'text_de': TextBlobDeTokenizer() if config['use-textblob-de'] else SimpleGermanTokenizer(config['tokenizer-german-split-compound-words']), 'text_fr': None, 'text_it': None}
    if config["only-fr-descriptions"]:
        keys_of_descriptions = ['text_fr']
    elif config["only-it-descriptions"]:
        keys_of_descriptions = ['text_it']
    elif config["only-de-fr-descriptions"]:
        keys_of_descriptions = ['text_de', 'text_fr']
    elif config["only-de-it-descriptions"]:
        keys_of_descriptions = ['text_de', 'text_it']
    elif config["only-fr-it-descriptions"]:
        keys_of_descriptions = ['text_fr', 'text_it']
    elif config["only-de-fr-it-descriptions"]:
        keys_of_descriptions = ['text_de', 'text_fr', 'text_it']
    else:
        keys_of_descriptions = ['text_de']
        
    vocab = set()
    # You have to install the stopwords corpus by executing nltk.download()
    # and install Corpora -> stopwords
    stop_words = stopwords.words('german')
    tokenize_and_output(config['drg-catalog'], tokenizers_by_key_of_description, config['drg-tokenizations'],
                         'code', keys_of_descriptions, vocab, ',', 'DRG',
                          config['use-descriptions'], stop_words)
    tokenize_and_output(config['chop-catalog'], tokenizers_by_key_of_description, config['chop-tokenizations'], 
                        'code', keys_of_descriptions, vocab, ',', 'CHOP',
                         config['use-descriptions'], stop_words)
    tokenize_and_output(config['icd-catalog'], tokenizers_by_key_of_description, config['icd-tokenizations'], 
                        'code', keys_of_descriptions, vocab, ',', 
                        'ICD', config['use-descriptions'], stop_words)
    combine_files([config['drg-tokenizations'], config['chop-tokenizations'], config['icd-tokenizations']],  config['all-tokens'])
    output_vocab(config['all-vocab'], vocab)
