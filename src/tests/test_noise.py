from unittest import TestCase
from src.geometry.animations.synth_wing_animations import *

from src.geometry.numpy.wing_models import *
from src.util.image_transforms import *


class MyTestCase(TestCase):

    def test_random_noise(self):
        num_of_data = 10
        for i in range(num_of_data):
            img = SyntheticWingModel.noisy_image_creation(camera_pos["tunnel_upper_cam_middle_focus"],
                                                          0, 0, 0, cam_noise=(0.0005, 0.001, 0.05))
            #   (0.0005, 0.001, 0.05) looks good
            cv2.imshow("frame", img)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
