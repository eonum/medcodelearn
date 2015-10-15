# -*- coding: utf-8 -*-
import csv
from drgtraininginstance import DRGTrainingInstance
class DRGReader:
    def __init__(self, filename, drg_filename):
        self.FIELDNAMES = ['id', 'ageYears', 'ageDays', 'admWeight', 'sex', 'adm', 'sep', 'los', 'sdf', 'hmv', 'pdx']
        self.RESTKEY = 'diagproc'
        self.MAX_ADDITIONAL_DIAGNOSES = 99
        self.MAX_PROCEDURES = 100 
        self.filename = filename
        self.drg_filename = drg_filename
        self.drgs_by_id = self.get_drgs_by_id()
    
    def get_drgs_by_id(self):
        drgs_by_id = {}               
        with open(self.drg_filename, 'r') as csvFile:
            reader = csv.reader(csvFile, delimiter=';')
            for row in reader:
                drgs_by_id[row[0]] = row[1]       
        return drgs_by_id

    def read_from_file(self):
        with open(self.filename, 'r') as csvFile:
            reader = csv.DictReader(csvFile, fieldnames=self.FIELDNAMES, restkey=self.RESTKEY, delimiter=';')
            for row in reader:
                drg_training_instance = self.get_drg_training_instance_from_row(row)
                
    def get_drg_training_instance_from_row(self, row):
        codes = {}
        diagproc = row[self.RESTKEY]
        diags = diagproc[0:self.MAX_ADDITIONAL_DIAGNOSES]
        procs = diagproc[self.MAX_ADDITIONAL_DIAGNOSES:self.MAX_ADDITIONAL_DIAGNOSES+self.MAX_PROCEDURES]
        print(' '.join([row['id'],row['sex'],diags[0],procs[0]]))
        
        for diag in diags:
            self.add_code(diag, codes, 1)
            
        for proc in procs:
            self.add_code(proc, codes, 1)
        
        return DRGTrainingInstance(codes, self.drgs_by_id[row['id']])
    
    def add_code(self, code, codes, weight):
        code = code.upper().replace(' ', '')
        for i in range(1,len(code)):
            subcode = code[:i]
            if len(subcode) == 0:
                continue
            if subcode in codes:
                codes[subcode] += weight
            else:
                codes[subcode] = weight
            
            
        
if __name__ == '__main__':
    r = DRGReader('../data/2015/trainingData2015_20151001.csv', '../data/2015/drg_file.csv')
    r.read_from_file()