from reader.sparsehierarchical.drgreader import DRGReader
import csv   
import numpy as np      
from vectorize import unitvec
from reader.flatvectors.pcreaderflatvectorized import FlatVectorizedPCReader

# Sequence to Flat classification
class SequenceVectorizePCReader(FlatVectorizedPCReader):
    def calculate_input_size(self):
        return 2 * len(self.vectors_by_code[list(self.vectors_by_code.keys())[0]][0]) + len(self.demo_variables_to_use)
    
    
    def empty_input(self, dataset):
        return np.empty((len(dataset), self.vector_size), dtype=np.float32)         
    
    def instance(self, row, diags, procs, gt):
        sequence = []
        demographic = np.zeros(self.vector_size, dtype=np.float32)
        for i, var in enumerate(self.demo_variables_to_use):
            demographic[self.word2vec_dims + i] = self.convert_demographic_variable(row, var)
        sequence.append(demographic)
        
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
        
        data_procedures = np.zeros(self.word2vec_dims / 2, dtype=np.float32)
        for proc in procs:
            data = np.zeros(self.word2vec_dims, dtype=np.float32)
            if self.code_type == 'srg':
                excludes.append(proc)
            for t in self.vectors_by_code['CHOP_' + proc]:
                data_procedures += t
            data = unitvec(data)
            data.resize(self.vector_size)
            sequence.append(data)  
        
        
        return [sequence, gt, excludes]
    
