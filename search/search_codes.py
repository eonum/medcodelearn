import sys

from vectorize import read_code_vectors, read_vectors

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python search_codes.py token_file vector_file code_type search_phrase")
        print("   code_type is one of DRG|ICD|CHOP")
        exit(-1)
    
    token_file = sys.argv[1]
    vector_file = sys.argv[2]
    code_type = sys.argv[3]
    phrase = sys.argv[4]
    
    if code_type not in ['ICD', 'CHOP', 'DRG']:
        print("Code type has to be one of ICD|DRG|CHOP")
        exit(-2)
        
    print("Reading vectors and tokens..")
    
    vector_by_token = read_vectors(vector_file)
    # several vectors for each code. The first vector is from the code token.
    res = read_code_vectors(vector_by_token, token_file)
    vectors_by_codes = res['vectors']
    tokens_by_codes = res['tokens']
    
    code_vocab = []
    for code in vectors_by_codes.keys():
        if(code.startswith(code_type)):
            code_vocab.append(code)
    
    vector_size = vectors_by_codes[code_vocab[0]][0].shape[0]
    
    print("Vector size is " + str(vector_size))
    
    #average_vector_by_code = 