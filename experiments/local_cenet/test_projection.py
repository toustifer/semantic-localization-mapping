import unittest
import numpy as np
from real_cenet import project

class ProjectionTests(unittest.TestCase):
    def test_nearest_surface_and_original_order(self):
        p=np.array([[5,0,0,.2],[10,0,0,.9],[0,5,0,.3]],dtype=np.float32)
        image,mask,px,py=project(p)
        self.assertTrue(mask[py[0],px[0]])
        self.assertEqual(image[0,py[0],px[0]],5)
        self.assertEqual(px[0],px[1])
        self.assertNotEqual(px[0],px[2])
        self.assertAlmostEqual(float(image[4,py[0],px[0]]),.2)
    def test_reject_invalid_input(self):
        for p in [np.zeros((1,4)),np.array([[np.nan,0,0,1]])]:
            with self.assertRaises(ValueError): project(p)

if __name__=='__main__': unittest.main()
