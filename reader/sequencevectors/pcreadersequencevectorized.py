import numpy as np      
from vectorize import unitvec
from reader.flatvectors.pcreaderflatvectorized import FlatVectorizedPCReader
from sklearn import preprocessing

# Sequence to Flat classification
class SequenceVectorizedPCReader(FlatVectorizedPCReader):
    def init(self):
        self.vector_size = len(self.vectors_by_code[list(self.vectors_by_code.keys())[0]][0]) + len(self.demo_variables_to_use)
        self.word2vec_dims = self.vector_size - len(self.demo_variables_to_use)
        self.demo_vars = []
    
    def finalize(self):
        demos = np.empty((len(self.demo_vars), self.vector_size), dtype=np.float32)
        for i, demo in enumerate(self.demo_vars):
            demos[i] = demo
        scaler = preprocessing.MaxAbsScaler().fit(demos)
        demos = scaler.transform(demos) 
        for i, demo in enumerate(demos):
            self.data[i][0] = demo
     
    def empty_input(self, dataset):
        # Use this if padding is done in the reader
        # return np.empty((len(dataset), 15, self.vector_size), dtype=np.float32)         
        return [None] * len(dataset)
    
    def instance(self, row, diags, procs, gt):
        sequence = []
        demographic = np.zeros(self.vector_size, dtype=np.float32)
        for i, var in enumerate(self.demo_variables_to_use):
            demographic[self.word2vec_dims + i] = self.convert_demographic_variable(row, var)
        sequence.append(demographic)
        self.demo_vars.append(demographic)
        
        excludes = []
        for diag in diags:
            data = np.zeros(self.word2vec_dims, dtype=np.float32)
            if self.code_type in ['pdx', 'sdx']:
                excludes.append(diag)
            for t in self.vectors_by_code['ICD_' + diag]:
                data += t
            data = unitvec(data)
            data.resize(self.vector_size)
            sequence.append(data)
        
        for proc in procs:
            data = np.zeros(self.word2vec_dims, dtype=np.float32)
            if self.code_type == 'srg':
                excludes.append(proc)
            for t in self.vectors_by_code['CHOP_' + proc]:
                data += t
            data = unitvec(data)
            data.resize(self.vector_size)
            sequence.append(data)  
        
        return [sequence, gt, excludes]
    
