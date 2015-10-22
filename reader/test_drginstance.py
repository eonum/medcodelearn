# -*- coding: utf-8 -*-
import unittest

class TestDRGInstance(unittest.TestCase):        
    def test_sparse_hierarchical_features(self):
        infos = None
        diags = ['AAA', 'AAB', 'ABA', 'BAA']
        procs = ['000', '001', '010', '100']        
        drg_instance = DRGTrainingInstance(infos, diags, procs)
        
        self.assertListEqual(sorted(list(drg_instance.get_sparse_hierarchically_coded_features().keys())), sorted(['A', 'AA', 'AAA', 'AAB', 'AB', 'ABA', 'B', 'BA', 'BAA', '0', '00', '000', '001', '01', '010', '1', '10', '100']))

if __name__ == '__main__':
    unittest.main()