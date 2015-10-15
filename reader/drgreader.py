# -*- coding: utf-8 -*-
import csv
class DRGReader:
    def __init__(self, filename):
        self.fieldnames = ['id', 'ageYears', 'ageDays', 'admWeight', 'sex', 'adm', 'sep', 'los', 'sdf', 'hmv', 'pdx']
        self.MAX_ADDITIONAL_DIAGNOSES = 99
        self.MAX_PROCEDURES = 100       
        self.filename = filename
    
    def readFromFile(self):
        
        with open(self.filename, 'r') as csvFile:
            reader = csv.DictReader(csvFile, fieldnames=self.fieldnames, restkey='diagproc', delimiter=';')
            for row in reader:
                diagproc = row['diagproc']
                diags = diagproc[0:self.MAX_ADDITIONAL_DIAGNOSES]
                procs = diagproc[self.MAX_ADDITIONAL_DIAGNOSES:self.MAX_ADDITIONAL_DIAGNOSES+self.MAX_PROCEDURES]
                print(' '.join([row['id'],row['sex'],diags[0],procs[0]]))
                
    
        
if __name__ == '__main__':
    r = DRGReader('../data/2015/trainingData2015_20151001.csv')
    r.readFromFile()