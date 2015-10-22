# -*- coding: utf-8 -*-
class DRGInstance:
    def __init__(self, infos, diags, procs, drg):
        self.diags = diags
        self.procs = procs
        
        self.sparse_hierarchically_coded_features = None
        self.contained_hierarchically_coded_feature_names = None
        self.full_hierarchically_coded_features = None
                    
    def get_sparse_hierarchically_coded_features(self):
        if self.sparse_hierarchically_coded_features:
            return (self.sparse_hierarchically_coded_features, self.contained_hierarchically_coded_feature_names) 
        
        self.sparse_hierarchically_coded_features = {}
        self.contained_hierarchically_coded_feature_names =  []
        for diag in self.diags:
            self.add_code(diag, 1)
            
        for proc in self.procs:
            self.add_code(proc, 1)
        
        return (self.sparse_hierarchically_coded_features, self.contained_hierarchically_coded_feature_names)

    def get_full_hierarchically_coded_features(self, all_hierarchically_coded_feature_names):
        pass
    
    def add_code(self, code, weight):
        code = code.upper().replace(' ', '')
        for i in range(1,len(code)):
            subcode = code[:i]
            if len(subcode) == 0:
                continue
            if subcode in self.sparse_hierarchically_coded_features:
                self.sparse_hierarchically_coded_features[subcode] += weight
            else:
                self.sparse_hierarchically_coded_features[subcode] = weight
            if subcode not in self.contained_hierarchically_coded_feature_names:
                self.contained_hierarchically_coded_feature_names.append(subcode)
    
    