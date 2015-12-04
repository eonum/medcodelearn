# -*- coding: utf-8 -*-
import csv

class DRGCodesReader:
    def __init__(self, filename):
        self.filename = filename
        self.codes = []

    def read_from_file(self):
        with open(self.filename, 'r') as csvFile:
            reader = csv.DictReader(csvFile, delimiter=',')
            print(reader)
            for row in reader:
                codes.append(row)
        return self.codes
    
