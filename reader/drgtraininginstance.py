# -*- coding: utf-8 -*-
class DRGTrainingInstance:
    def __init__(self, infos, diags, procs, gt=None):
        self.diags = diags
        self.procs = procs
        self.gt = gt
        
        self.sparse_hierarchically_coded_features = None
        self.contained_hierarchically_coded_feature_names = None
                    
    def get_sparse_hierarchically_coded_features(self):
        if self.sparse_hierarchically_coded_features:
            return self.sparse_hierarchically_coded_features
        
        self.sparse_hierarchically_coded_features = {}
        self.contained_hierarchically_coded_feature_names =  []
        for diag in self.diags:
            self.add_code(diag, 1)
            
        for proc in self.procs:
            self.add_code(proc, 1)
        
        return self.sparse_hierarchically_coded_features
        
    def get_contained_hierarchically_coded_feature_names(self):
        if not self.contained_hierarchically_coded_feature_names:
            self.get_contained_hierarchically_coded_feature_names
        return self.contained_hierarchically_coded_feature_names

    def get_full_hierarchically_coded_features(self, all_hierarchically_coded_feature_names):
        full_hierarchically_coded_features = dict(self.sparse_hierarchically_coded_features)
        for code in all_hierarchically_coded_feature_names:
            if code not in self.contained_hierarchically_coded_feature_names:
                full_hierarchically_coded_features[code] = 0
        return full_hierarchically_coded_features                    
            
    def add_code(self, code, weight):
        code = code.upper().replace(' ', '')
        for i in range(0,len(code)+1):
            subcode = code[:i]
            if len(subcode) == 0:
                continue
            if subcode in self.sparse_hierarchically_coded_features:
                self.sparse_hierarchically_coded_features[subcode] += weight
            else:
                self.sparse_hierarchically_coded_features[subcode] = weight
            if subcode not in self.contained_hierarchically_coded_feature_names:
                self.contained_hierarchically_coded_feature_names.append(subcode)
    

