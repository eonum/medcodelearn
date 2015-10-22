# -*- coding: utf-8 -*-
class DRGTrainingSet():
    def __init__(self):
        self.instances = []
        
    
    def add_drg_training_instance(self, instance):
        self.instances.append(instance)

    def get_hierarchically_coded_feature_vectors_and_targets(self):
        all_hierarchically_coded_feature_names = self.get_all_hierarchically_coded_feature_names() 
        return [instance.get_full_hierarchically_coded_feature_vectors_and_targets(all_hierarchically_coded_feature_names) for instance in self.instances]
        
    def get_all_hierarachically_coded_feature_names(self):
        all_hierarchically_coded_feature_names = [feature_name for feature_name in [instance.get_contained_hierarchically_coded_feature_names() for instance in self.instances]]
        all_hierarchically_coded_feature_names = list(set(all_hierarchically_coded_feature_names))
        return all_hierarchically_coded_feature_names
        
    