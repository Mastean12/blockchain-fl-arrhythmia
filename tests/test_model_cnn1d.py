import unittest

import torch

from src.models.cnn1d import ECG1DCNN, count_trainable_parameters


class ECG1DCNNTests(unittest.TestCase):
    def test_initialization_and_parameter_count(self):
        model = ECG1DCNN(in_channels=2, num_classes=15, input_length=216)
        self.assertEqual(model.num_classes, 15)
        self.assertEqual(count_trainable_parameters(model), 10127)

    def test_forward_pass_shape(self):
        model = ECG1DCNN()
        output = model(torch.randn(4, 2, 216))
        self.assertEqual(tuple(output.shape), (4, 15))

    def test_rejects_wrong_input_dimensions(self):
        model = ECG1DCNN()
        with self.assertRaises(ValueError):
            model(torch.randn(4, 216, 2))
        with self.assertRaises(ValueError):
            model(torch.randn(4, 2, 215))


if __name__ == "__main__":
    unittest.main()
