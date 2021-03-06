# -*- coding: utf-8 -*-
import csv
from reader.sparsehierarchical.drgtraininginstance import DRGTrainingInstance
from reader.sparsehierarchical.drgtrainingset import DRGTrainingSet

class DRGReader:
    FIELDNAMES = ['id', 'ageYears', 'ageDays', 'admWeight', 'sex', 'adm', 'sep', 'los', 'sdf', 'hmv', 'pdx']
    DRG_OUT_FIELDNAMES = ['id', 'drg', 'mdc', 'gage', 'gsex', 'gst', 'pccl', 'ecw', 'cflag']
    RESTKEY = 'diagproc'
    MAX_ADDITIONAL_DIAGNOSES = 99
    MAX_PROCEDURES = 100 
        
    def __init__(self, filename):
        self.filename = filename
        self.feature_names = []  
        self.drg_trainingset = DRGTrainingSet()

    def read_from_file(self):
        with open(self.filename, 'r') as csvFile:
            reader = csv.DictReader(csvFile, fieldnames=self.FIELDNAMES, restkey=self.RESTKEY, delimiter=';')
            for row in reader:
                for instance in self.get_drg_instances_from_row(row):
                    self.drg_trainingset.add_drg_training_instance(instance)
        return self.drg_trainingset
    
          

class DRGCodeProposalReader(DRGReader):
    def get_drg_instances_from_row(self, row):
        diagproc = row[self.RESTKEY]
        diags = diagproc[0:self.MAX_ADDITIONAL_DIAGNOSES] + row['pdx']
        procs = diagproc[self.MAX_ADDITIONAL_DIAGNOSES:self.MAX_ADDITIONAL_DIAGNOSES+self.MAX_PROCEDURES]
        infos = [row[fieldname] for fieldname in self.FIELDNAMES]        
        #print(' '.join([row['id'],row['sex'],diags[0],procs[0]])) # TODO: Delete this line.
        
        instance_per_diag = [DRGTrainingInstance(infos, [diag for diag in diags if diag != gt], procs, gt) for gt in diags]
        instance_per_proc = [DRGTrainingInstance(infos, diags, [proc for proc in procs if proc != gt], gt) for gt in procs]        
        return instance_per_diag + instance_per_proc    


if __name__ == '__main__':
    r = DRGCodeProposalReader('../data/2015/trainingData2015_20151001.csv')
    r.read_from_file()