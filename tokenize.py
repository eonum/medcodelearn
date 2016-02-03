# -*- coding: utf-8 -*-
from reader.csvreader import CSVReader
from tokenization.tokenizer import GermanTokenizer
import re
from pprint import pprint

def tokenize_code(code):
    code = re.sub(r'[^\w\s]','',code)
    return ['CODEPREFIX'+code[0:x] for x in reversed(range(1,len(code)+1))]

def tokenize_and_output(csv_filename, tokenizer, output_filename, key_of_code, key_of_description, vocab):
    reader = CSVReader(csv_filename)
    dataset = reader.read_from_file()
    try:
        #os.remove(output_filename)
        pass
    except OSError:
        pass  
    

    with open(output_filename, 'w+') as out_file:
        for record in dataset:
            tokenized_record = tokenizer.tokenize(record[key_of_description])
            tokenized_record = tokenize_code(record[key_of_code]) + tokenized_record 
            output_line = " ".join(tokenized_record)
            vocab.update(tokenized_record)
            print(output_line, file=out_file) 
            print(output_line)
    
    
def combine_files(files, big_file):
    with open(big_file, 'w+') as big_file:
        for file_name in files:
            big_file.write(open(file_name).read())
            
def output_vocab(vocab_filename, vocab):
    with open(vocab_filename, 'w+') as out_file:
        for word in vocab:
            print(word, file=out_file)
    
if __name__ == '__main__':
    tokenizer = GermanTokenizer()
   
    vocab_de = set()
    tokenize_and_output('data/2015/drgs.csv', tokenizer, 'data/tokenization/drgs_tokenized.csv', 'code', 'text_de', vocab_de)
    #tokenize_and_output('data/2015/chop_codes.csv', tokenizer, 'data/tokenization/chop_codes_tokenized.csv', 'code', 'text_de', vocab_de)
    #tokenize_and_output('data/2015/icd_codes.csv', tokenizer, 'data/tokenization/icd_codes_tokenized.csv', 'code', 'text_de', vocab_de)
    #combine_files(['data/tokenization/drgs_tokenized.csv', 'data/tokenization/chop_codes_tokenized.csv', 'data/tokenization/icd_codes_tokenized.csv'],  'data/tokenization/tokens.csv')
    output_vocab('data/tokenization/vocab_only_drgs.csv', vocab_de)
    pprint(vocab_de)
