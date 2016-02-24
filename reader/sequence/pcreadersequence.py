from reader.flatvectors.pcreaderflatvectorized import FlatVectorizedPCReader

# Sequence to Flat classification
class SequencePCReader(FlatVectorizedPCReader):
    tokens_by_code = None
    
    def init(self):
        pass
    
    def finalize(self):
        pass
     
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
            for t in self.tokens_by_code['ICD_' + diag]:
                sequence.append(t)
        
        for proc in procs:
            if self.code_type == 'srg':
                excludes.append(proc)
            for t in self.tokens_by_code['CHOP_' + proc]:
                sequence.append(t)
        
        return [sequence, gt, excludes]
    