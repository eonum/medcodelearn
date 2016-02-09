# -*- coding: utf-8 -*-
import csv
from reader.drgtraininginstance import DRGTrainingInstance
from reader.drgtrainingset import DRGTrainingSet

class DRGReader:
    def __init__(self, filename):
        self.FIELDNAMES = ['id', 'ageYears', 'ageDays', 'admWeight', 'sex', 'adm', 'sep', 'los', 'sdf', 'hmv', 'pdx']
        self.RESTKEY = 'diagproc'
        self.MAX_ADDITIONAL_DIAGNOSES = 99
        self.MAX_PROCEDURES = 100 
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