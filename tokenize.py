# -*- coding: utf-8 -*-
from reader.csvreader import CSVReader
from tokenization.tokenizer import GermanTokenizer

def tokenize_and_output(csv_filename, tokenizer, output_filename, key_of_code, key_of_description):
    reader = CSVReader(csv_filename)
    dataset = reader.read_from_file()
    try:
        #os.remove(output_filename)
        pass
    except OSError:
        pass  

    with open(output_filename, 'w+') as file:
        for record in dataset:
            tokenized_record = tokenizer.tokenize(record[key_of_description])
            output_line = record[key_of_code] + ", " + ", ".join(tokenized_record)
            print(output_line, file=file)
            print(output_line)
            
if __name__ == '__main__':
   #drgs_reader = CSVReader('data/2015/drgs.csv')
   #chop_codes_reader = CSVReader('data/2015/chop_codes.csv')
   #icd_codes_reader = CSVReader('data/2015/icd_codes.csv')
   tokenizer = GermanTokenizer()
   
   tokenize_and_output('data/2015/drgs.csv', tokenizer, 'data/drgs_tokenized.csv', 'code', 'text_de')
   tokenize_and_output('data/2015/chop_codes.csv', tokenizer, 'data/chop_codes_tokenized.csv', 'code', 'text_de')
   tokenize_and_output('data/2015/icd_codes.csv', tokenizer, 'data/icd_codes_tokenized.csv', 'code', 'text_de')

