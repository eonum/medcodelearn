from reader.flatvectors.pcreaderflatvectorized import FlatVectorizedPCReader

# Sequence to Flat classification
class SequencePCReader(FlatVectorizedPCReader):
    tokens_by_code = None
    
    def init(self):
        pass
    
    def finalize(self):
        # replace codes with indices
        input_set = set()
        for sample in self.data:
            for code in sample:
                input_set.add(code)
        self.vocab = ['mask'] + list(input_set)
        for i, codes in enumerate(self.data):
            code_indices = []
            for code in codes:
                code_indices.append(self.vocab.index(code))
            self.data[i] = code_indices
     
    def empty_input(self, dataset):
        # Use this if padding is done in the reader
        # return np.empty((len(dataset), 15, self.vector_size), dtype=np.float32)         
        return [None] * len(dataset)
    
    def instance(self, row, diags, procs, gt):
        sequence = []
        #demographic = np.zeros(len(self.demo_variables_to_use), dtype=np.float32)
        #for i, var in enumerate(self.demo_variables_to_use):
        #    demographic[i] = self.convert_demographic_variable(row, var)
        #sequence.append(demographic)
        #self.demo_vars.append(demographic)
        
        excludes = []
        for diag in diags:
            if self.code_type in ['pdx', 'sdx']:
                excludes.append(diag)
            if self.use_all_tokens:
                for t in self.tokens_by_code['ICD_' + diag]:
                    sequence.append(t)
            else:
                sequence.append(self.tokens_by_code['ICD_' + diag][0])
        
        for proc in procs:
            if self.code_type == 'srg':
                excludes.append(proc)
            if self.use_all_tokens:
                for t in self.tokens_by_code['CHOP_' + proc]:
                    sequence.append(t)
            else:
                sequence.append(self.tokens_by_code['CHOP_' + proc][0])
        
        return [sequence, gt, excludes]
    
