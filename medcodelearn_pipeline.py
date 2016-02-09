from tokenize_codes import tokenize_catalogs
import os

def run (config):
    base_folder = config['base_folder']
    if not os.path.exists(base_folder + 'tokenization'):
        os.makedirs(base_folder + 'tokenization')
    tokenize_catalogs(config)



if __name__ == '__main__':
    base_folder = 'data/pipelinetest/'
    config = {
        'base_folder' : base_folder,
        'drg-catalog' : 'data/2015/drgs.csv',
        'chop-catalog' : 'data/2015/chop_codes.csv',
        'icd-catalog' : 'data/2015/icd_codes.csv',
        'drg-tokenizations' : base_folder + 'tokenization/drgs_tokenized.csv',
        'icd-tokenizations' : base_folder + 'tokenization/chop_codes_tokenized.csv',
        'chop-tokenizations' : base_folder + 'tokenization/chop_codes_tokenized.csv',
        'all-tokens' : base_folder + 'tokenization/all_tokens.csv',
        'all-vocab' : base_folder + 'tokenization/vocab_all.csv' }
    
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    run(config)
    