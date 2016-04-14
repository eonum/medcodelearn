from reader.sparsehierarchical.drgreader import DRGReader
import csv   
import numpy as np      
from vectorize import unitvec


class FlatVectorizedPCReader(DRGReader):
    def empty_input(self, dataset):
        return np.empty((len(dataset), self.vector_size), dtype=np.float32)
    
    def init(self):
        self.vector_size = 2 * len(self.vectors_by_code[list(self.vectors_by_code.keys())[0]][0]) + len(self.demo_variables_to_use)
        self.word2vec_dims = self.vector_size - len(self.demo_variables_to_use)
    
    def finalize(self):
        pass
    
    
    def read_from_file(self, vectors_by_code, 
                       code_type = 'pdx', 
                       drg_out_file = None,
                       demo_variables_to_use= ['admWeight', 'hmv', 'sex', 'los', 'ageYears', 
                                               'ageDays', 'adm-normal', 'adm-transfer', 
                                               'adm-transfer-short', 'adm-unknown',
                                               'sep-normal', 'sep-dead', 'sep-doctor',
                                               'sep-unknown', 'sep-transfer']):
        # available demographic variables:
        # 'id', 'ageYears', 'ageDays', 'admWeight', 'sex', 'adm', 'sep', 'los', 'sdf', 'hmv'
        self.demo_variables_to_use = demo_variables_to_use
        self.code_type = code_type
        self.vectors_by_code = vectors_by_code
        self.invalid_pdx = 0
        self.drg_out_file = drg_out_file
        
        self.init();
        
        if self.code_type == 'drg':
            if self.drg_out_file == None:
                raise ValueError('You must specify a corresponding DRG output file for the "drg" classification task')
            self.drg_by_id = self.read_drg_output()
        
        dataset = []
        with open(self.filename, 'r') as csvFile:
            reader = csv.DictReader(csvFile, fieldnames=self.FIELDNAMES, restkey=self.RESTKEY, delimiter=';')
            for row in reader:
                for instance in self.get_instances_from_row(row):
                    dataset.append(instance)
                    
        self.data = self.empty_input(dataset)
        self.targets = []
        self.excludes = []
        
        for i, instance in enumerate(dataset):
            self.data[i] = instance[0]
            self.targets.append(instance[1])
            self.excludes.append(instance[2])
        
        if self.invalid_pdx > 0:
            print('Skipped patient cases due to invalid PDX: ' + str(self.invalid_pdx))
            
        self.finalize()
            
        return {'data' : self.data, 'targets' : self.targets}          
    
    def get_instances_from_row(self, row):
        diagproc = row[self.RESTKEY]
        diags = diagproc[0:self.MAX_ADDITIONAL_DIAGNOSES]
        procs = map(lambda x: x.split(':')[0], diagproc[self.MAX_ADDITIONAL_DIAGNOSES:self.MAX_ADDITIONAL_DIAGNOSES+self.MAX_PROCEDURES])
        diags = [d for d in diags if d != '']
        procs = [p for p in procs if p != '']
        diags = map(lambda c: c.replace('.', '').upper(), diags)
        procs = map(lambda c: c.replace('.', '').upper(), procs)
        diags = [d for d in diags if 'ICD_' + d in self.vectors_by_code]
        procs = [p for p in procs if 'CHOP_' + p in self.vectors_by_code]
        pdx = row['pdx'].replace('.', '')
        # do not use this patient case if the PDX is non existent or invalid
        if pdx == '' or 'ICD_' + pdx not in self.vectors_by_code:
            self.invalid_pdx += 1
            return []
             
        
        if self.code_type == 'pdx':
            return [self.instance(row, diags, procs, pdx)]
        elif self.code_type == 'sdx':
            return [self.instance(row, [diag for diag in diags if diag != gt] + [pdx], procs, gt) for gt in diags]
        elif self.code_type == 'srg':
            return [self.instance(row, diags + [pdx], [proc for proc in procs if proc != gt], gt) for gt in procs]
        elif self.code_type == 'drg':
            return [self.instance(row, diags + [pdx], procs, self.drg_by_id[row['id']])]

        raise ValueError('code_type should be one of "drg", "pdx", "sdx" or "srg" but was ' + self.code_type)
    
    def instance(self, row, diags, procs, gt):
        data = np.zeros(int(self.word2vec_dims / 2), dtype=np.float32)
        excludes = []
        # sum over all vectors (first vector is the code token)
        for diag in diags:
            if self.code_type in ['pdx', 'sdx']:
                excludes.append(diag)
            for t in self.vectors_by_code['ICD_' + diag]:
                data += t
        data = unitvec(data)
        
        data_procedures = np.zeros(int(self.word2vec_dims / 2), dtype=np.float32)
        for proc in procs:
            if self.code_type == 'srg':
                excludes.append(proc)
            for t in self.vectors_by_code['CHOP_' + proc]:
                data_procedures += t
                
        data_procedures = unitvec(data_procedures)
        data = np.append(data, data_procedures)
        
        data.resize(self.vector_size)
        
        for i, var in enumerate(self.demo_variables_to_use):
            data[self.word2vec_dims + i] = self.convert_demographic_variable(row, var)
        
        return [data, gt, excludes]
    
    def convert_demographic_variable(self, row, var):
        value = row[var[0:3]] if var[0:3] in ['adm', 'sep'] else row[var]
        if var == 'sex':
            return 1.0 if value.upper() == 'M' else -1.0
        if var[0:3] == 'adm':
            if var[4:] == 'normal' and value == '01':
                return 1.0
            elif var[4:] == 'transfer' and value == '11':
                return 1.0
            elif var[4:] == 'transfer-short' and value== '06':
                return 1.0
            elif var[4:] == 'unknown' and value == '99':
                return 1.0
            else:
                return 0.0
        if var[0:3] == 'sep':
            if var[4:] == 'normal' and value == '00':
                return 1.0
            elif var[4:] == 'transfer' and value == '06':
                return 1.0
            elif var[4:] == 'dead' and value== '07':
                return 1.0
            elif var[4:] == 'doctor' and value== '04':
                return 1.0
            elif var[4:] == 'unknown' and value == '99':
                return 1.0
            else:
                return 0.0
        return float(value)
    
    def read_drg_output(self):
        drg_by_id = {}
        with open(self.drg_out_file, 'r') as csvFile:
            reader = csv.DictReader(csvFile, self.DRG_OUT_FIELDNAMES, delimiter=';')
            for row in reader:
                drg_by_id[row['id']] = row['drg']
        return drg_by_id
    
