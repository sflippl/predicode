import unittest
import predicode as pc

class TestImageData(unittest.TestCase):
    
    def test_hex_2(self):
        self.assertEqual(pc.ImageData.hex_2(2), '02')
        self.assertEqual(pc.ImageData.hex_2(30), '1e')
        self.assertEqual(pc.ImageData.hex_2(0), '00')
        with self.assertRaises(AssertionError):
            pc.ImageData.hex_2(16**3)
            pc.ImageData.hex_2(-1)
        with self.assertRaises(TypeError):
            pc.ImageData.hex_2('01')
            pc.ImageData.hex_2(1.5)

# if __name__ == '__main__':
#     unittest.main()