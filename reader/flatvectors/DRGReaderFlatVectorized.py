from reader.sparsehierarchical.drgreader import DRGReader
import csv   
import numpy as np      
from vectorize import unitvec


class DRGReaderFlatVectorized(DRGReader):
    def read_from_file(self, vectors_by_code):
        self.vectors_by_code = vectors_by_code
        self.vector_size = len(vectors_by_code[vectors_by_code.keys()[0]])
        dataset = []
        with open(self.filename, 'r') as csvFile:
            reader = csv.DictReader(csvFile, fieldnames=self.FIELDNAMES, restkey=self.RESTKEY, delimiter=';')
            for row in reader:
                for instance in self.get_drg_instances_from_row(row):
                    dataset.add(instance)
                    
        self.data = np.empty((len(dataset), self.vector_size), dtype=np.float)
        self.targets = np.empty(len(dataset), dtype=np.string_)
        
        for i, instance in enumerate(dataset):
            self.data[i] = instance[0]
            self.targets[i] = instance[1]
        
        return {'data' : self.data, 'targets' : self.targets}          

    def get_drg_instances_from_row(self, row):
        diagproc = row[self.RESTKEY]
        diags = diagproc[0:self.MAX_ADDITIONAL_DIAGNOSES] + row['pdx']
        procs = diagproc[self.MAX_ADDITIONAL_DIAGNOSES:self.MAX_ADDITIONAL_DIAGNOSES+self.MAX_PROCEDURES]
        infos = [row[fieldname] for fieldname in self.FIELDNAMES]        
        
        instance_per_diag = [self.flat_instance(infos, [diag for diag in diags if diag != gt], procs, gt) for gt in diags]
        instance_per_proc = [self.flat_instance(infos, diags, [proc for proc in procs if proc != gt], gt) for gt in procs]        
        return instance_per_diag + instance_per_proc  
    
    def flat_instance(self, infos, diags, procs, gt):
        data = np.empty(self.vector_size, dtype=np.float)
        for diag in diags:
            data.add(self.vectors_by_code['ICD_' + diag])
        for proc in procs:
            data.add(self.vectors_by_code['CHOP_' + proc])
        data = unitvec(data)
        return [data, gt]