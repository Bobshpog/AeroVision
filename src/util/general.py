from dataclasses import dataclass

@dataclass
class MinCounter:
    max_count: int

    def __post_init__(self):
        if self.max_count < 0:
            raise ValueError
        self.data = [(-self.max_count + 1, float("inf"))]
        self.count = 0

    def add(self, val: float, epoch: int,eps:float=0 ) -> bool:
        """
        Args:
            val:number to add to min list
        Returns:
            weather we didn't update the list for max_count epochs
        """
        self.data = [x for x in self.data if ((epoch - x[0]) <= self.max_count)]
        if (not bool(self.data)):
            # if list is empty
            return True
        if (val<self.data[0][1] +eps):
            self.data.append((epoch, val))
        self.data.sort(key=lambda x: x[1])
        return False

class Functor:
    def __init__(self,foo):
        self.foo=foo

    def __call__(self, *args, **kwargs):
        return self.foo(*args,**kwargs)

#
# def create_stripes_texture(path, is_long_stripes, num_of_stripes=50, tex_resolution=(640,500,3), color1=(0,0,255),
#                            color2=(0,0,0),plot=False):     # colors in cv2 format (g,b,r)
#     num_pixel_per_stipe_long = int(tex_resolution[0]/num_of_stripes)
#     num_pixel_per_stipe_short = int(tex_resolution[1] / num_of_stripes)
#     zero = np.zeros(tex_resolution)
#     for i in range(num_of_stripes):
#         if i % 2 == 0:
#             if is_long_stripes:
#                 zero[num_pixel_per_stipe_long*i:num_pixel_per_stipe_long*i+num_pixel_per_stipe_long,:, :] = color1
#             else:
#                 zero[:, num_pixel_per_stipe_short * i:num_pixel_per_stipe_short * i + num_pixel_per_stipe_short, :] = color1
#         else:
#             if is_long_stripes:
#                 zero[num_pixel_per_stipe_long*i:num_pixel_per_stipe_long*i+num_pixel_per_stipe_long,:, :] = color2
#             else:
#                 zero[:, num_pixel_per_stipe_short * i:num_pixel_per_stipe_short * i + num_pixel_per_stipe_short, :] = color2
#
#     mesh = Mesh("data/wing_off_files/synth_wing_v5.off")
#     # mesh.plot_faces()
#     if plot:
#         cv2.imshow("frame", zero)
#         mesh.plot_faces(texture=path)
#     cv2.imwrite(path, zero)
