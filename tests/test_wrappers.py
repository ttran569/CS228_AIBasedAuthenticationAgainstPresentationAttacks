import unittest

from facenet_auth.models import FaceNetIdentityModel, MiniFASNetV2PAD


class WrapperSmokeTests(unittest.TestCase):
    def test_face_net_wrapper_instantiates(self):
        model = FaceNetIdentityModel(device="cpu")
        self.assertIsNotNone(model)
        self.assertEqual(model.device.type, "cpu")

    def test_pad_wrapper_instantiates(self):
        model = MiniFASNetV2PAD(device="cpu")
        self.assertIsNotNone(model)
        self.assertEqual(model.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
