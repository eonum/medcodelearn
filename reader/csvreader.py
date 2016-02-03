# -*- coding: utf-8 -*-
import csv

class CSVReader:
    def __init__(self, filename, delimiter=','):
        self.filename = filename
        self.codes = []
        self.delimiter = delimiter

    def read_from_file(self):
        with open(self.filename, 'r') as csvFile:
            reader = csv.DictReader(csvFile, delimiter=self.delimiter)
            print(reader)
            for row in reader:
                self.codes.append(row)
        return self.codes
    
