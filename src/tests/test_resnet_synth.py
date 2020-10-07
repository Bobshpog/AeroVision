from unittest import TestCase

from src.models.resnet_synth import CustomInputResnet


class TestCustomInputResnet(TestCase):
    def test_run_model(self):
        CHECKPOINT_PATH = ""
        model = CustomInputResnet.load_from_checkpoint(CHECKPOINT_PATH)
        model.eval()
        # usage is y=model(x)
