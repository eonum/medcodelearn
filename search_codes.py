import sys
import numpy as np

from vectorize import read_code_vectors, read_vectors, unitvec
from tokenization.tokenizer import GermanTokenizer
from scipy.spatial.distance import cosine
from reader.csvreader import CSVReader

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print("Usage: python search_codes.py token_file vector_file code_type catalog search_phrase")
        print("   code_type is one of DRG|ICD|CHOP")
        exit(-1)
    
    token_file = sys.argv[1]
    vector_file = sys.argv[2]
    code_type = sys.argv[3]
    catalog = sys.argv[4]
    phrase = sys.argv[5]
    
    if code_type not in ['ICD', 'CHOP', 'DRG']:
        print("Code type has to be one of ICD|DRG|CHOP")
        exit(-2)
        
    print("Reading catalog")
    reader = CSVReader(catalog, ',')
    descriptions_de = {}
    dataset = reader.read_from_file()
    for record in dataset:
        descriptions_de[code_type + '_' + record['code'].replace('.', '').upper()] = record['text_de']
        
    print("Reading vectors and tokens..")
    
    vector_by_token = read_vectors(vector_file)
    res = read_code_vectors(vector_by_token, token_file)
    vectors_by_codes = res['vectors']
    tokens_by_codes = res['tokens']
    
    code_vocab = []
    for code in vectors_by_codes.keys():
        if(code.startswith(code_type)):
            code_vocab.append(code)
    
    vector_size = vectors_by_codes[code_vocab[0]][0].shape[0]
    
    print("Vector size is " + str(vector_size))
    
    average_vector_by_code = np.zeros((len(code_vocab), vector_size), dtype=np.float32)
    
    for i, code in enumerate(code_vocab):
        vectors = vectors_by_codes[code]
        data = np.zeros(vector_size, dtype=np.float32)
        # sum over all vectors (first vector is the code token)
        for v in vectors:
            data += v
        data = unitvec(data)
        average_vector_by_code[i] = data
        
    tokenizer = GermanTokenizer()
    
    print("Search..")
    
    tokens = tokenizer.tokenize(phrase)
    print(tokens)
    average_phrase = np.zeros(vector_size, dtype=np.float32)
    for token in tokens:
        if token in vector_by_token.keys():
            print("Found " + token)
            average_phrase += vector_by_token[token]
        elif token.upper() in vector_by_token.keys():
            print("Found upper case " + token)
            average_phrase += vector_by_token[token.upper()]
    average_phrase = unitvec(average_phrase)
    
    distances = np.ones(len(code_vocab), dtype=np.float32)
    # TODO remove this for loop with more efficient numpy computation
    for i, code in enumerate(code_vocab):
        distances[i] = cosine(average_phrase, average_vector_by_code[i])
    
    most_similar_codes = distances.argsort()[:5]
    
    print("\nSearch Results")
    for rank, i in enumerate(most_similar_codes):
        desc = descriptions_de[code_vocab[i]] if code_vocab[i] in descriptions_de.keys() else ''
        print(str(rank) + '. ' + code_vocab[i] + ' ' + desc)
    
    