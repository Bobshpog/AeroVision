import glob
import os
import cv2

from src.geometry.numpy.transforms import *
from src.geometry.numpy.wing_models import *
from src.geometry.spod import *
from src.geometry.numpy.lbo import *

camera_pos = {
            'up_middle': [(0.047, -0.053320266561896174, 0.026735639600027315),
                          (0.05, 0.3, 0.02),
                          (0, 0, 1)],

            'down_middle': [(0.04581499400545182, -0.04477050005202985, -0.028567355483893577),
                            (0.05, 0.3, 0.02),
                            (0.001212842435223535, 0.13947688005070646, -1)],

            "up_right": [(0.11460619078012961, -0.04553696541254279, 0.038810512823530784),
                         (0.05, 0.3, 0.02),
                         (0, 0.16643488101070833, 1)],

            'down_right': [(0.11460619078012961, -0.04553696541254279, -0.038810512823530784),
                           (0.05, 0.3, 0.02),
                           (0, 0.16643488101070833, -1)],

            'up_left': [(-0.019770941905445285, -0.06082136750543311, 0.038694507832388224),
                        (0.05, 0.3, 0.02),
                        (0.041, 0.0438, 1)],

            'down_left': [(-0.019770941905445285, -0.06082136750543311, -0.038694507832388224),
                          (0.05, 0.3, 0.02),
                          (0.041, 0.0438, -1)],

            "rotated_down_left": [(-0.019770941905445285, -0.06082136750543311, -0.038694507832388224),
                                    (0.05, 0.3, 0.02),
                                    (0.041, 0.0438, 1)],

            'rotated_down_middle': [(0.04581499400545182, -0.04477050005202985, -0.028567355483893577),
                                    (0.05, 0.3, 0.02),
                                    (0.001212842435223535, 0.13947688005070646, 1)],

            'rotated_down_right': [(0.11460619078012961, -0.04553696541254279, -0.038810512823530784),
                                    (0.05, 0.3, 0.02),
                                     (0, 0.16643488101070833, 1)],
        }

def annimate_six_wings():
    # the movement for each of the meshes
    g1 = []
    f1 = []
    f2 = []
    g2 = []
    f3 = []
    g3 = []
    f4 = []
    g4 = []
    f5 = []
    g5 = []
    f6 = []
    g6 = []
    f7 = []
    g7 = []
    # we need to create 6 different meshes, three of tips and three for wing. Pyvista will not recognise the
    # meshes as different otherwise.
    tip1 = Mesh('data/wing_off_files/fem_tip.off')
    tip2 = Mesh('data/wing_off_files/fem_tip.off')
    tip3 = Mesh('data/wing_off_files/fem_tip.off')
    mesh1 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    mesh2 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    mesh3 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    tip4 = Mesh('data/wing_off_files/fem_tip.off')
    tip5 = Mesh('data/wing_off_files/fem_tip.off')
    tip6 = Mesh('data/wing_off_files/fem_tip.off')
    mesh4 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    mesh5 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    mesh6 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    tip7 = Mesh('data/wing_off_files/fem_tip.off')
    mesh7 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    meshes = [mesh1, tip1, mesh2, tip2, mesh3, tip3, mesh4, tip4, mesh5, tip5, mesh6, tip6, mesh7, tip7]
    # ^ define the order of each mesh
    frames = 40
    # number of frames of the gif, if no gif should be created this number should be around the 4000~ to make it
    # the same as 60~ with gif is created
    for phase in np.linspace(0, 4 * np.pi, frames + 1):
        f1.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh1.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))
        g1.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip1.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))

        f2.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh2.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))
        g2.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip2.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))

        f3.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh3.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))
        g3.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip3.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))

        f4.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh4.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))
        g4.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip4.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))

        f5.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh5.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))
        g5.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip5.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))

        f6.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh6.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))
        g6.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip6.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))
        f7.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh7.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))
        g7.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip7.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))
        # couldnt vectorise
    # the movement list
    fg = [f1, g1, f2, g2, f3, g3, f4, g4, f5, g5, f6, g6, f7, g7]
    cords = [(0, 0), (0, 0), (0, 1), (0, 1), (0, 2), (0, 2), (1, 0), (1, 0), (1, 1), (1, 1), (1, 2), (1, 2), (2, 1),
             (2, 1)]
    # cords of the subplot, both mesh are in the same subplot so both needing to be the same
    plotter = pv.Plotter(shape=(3, 3))
    for i in range(3):
        plotter.subplot(2, i)
        plotter.add_mesh(mesh=pv.Sphere(center=camera_pos["up_right"][0], radius=0.01),
                         color='black')
        plotter.add_mesh(mesh=pv.Sphere(center=camera_pos["up_left"][0], radius=0.01),
                         color='black')
        plotter.add_mesh(mesh=pv.Sphere(center=camera_pos["up_middle"][0], radius=0.01),
                         color='black')
        plotter.add_mesh(mesh=pv.Sphere(center=camera_pos["down_right"][0], radius=0.01),
                         color='black')
        plotter.add_mesh(mesh=pv.Sphere(center=camera_pos["down_left"][0], radius=0.01),
                         color='black')
        plotter.add_mesh(mesh=pv.Sphere(center=camera_pos["down_middle"][0], radius=0.01),
                         color='black')

    mesh1.plot_faces(show=False, plotter=plotter, index_row=2, index_col=0, texture="data/textures/checkers2.png")
    mesh1.plot_faces(show=False, plotter=plotter, index_row=2, index_col=2, texture="data/textures/checkers2.png")
    tip1.plot_faces(show=False, plotter=plotter, index_row=2, index_col=0)
    tip1.plot_faces(show=False, plotter=plotter, index_row=2, index_col=2)

    scalars = [None] * 14
    textures = ["data/textures/checkers2.png", None] * 7
    color_maps = ["jet"] * 14
    titles = ["up left", "", "up middle", "", "up right", "", "down left", "", "down middle", "", "down right", "",
              " camera view", ""]
    font_colors = ["black"] * 14
    font_size = [10] * 14
    cam = [camera_pos["up_left"], camera_pos["up_left"], camera_pos["up_middle"],camera_pos["up_middle"],
           camera_pos["up_right"], camera_pos["up_right"], camera_pos["down_left"], camera_pos["down_left"],
           camera_pos["down_middle"], camera_pos["down_middle"], camera_pos["down_right"], camera_pos["down_right"],
           None, None
           ]
    animate_few_meshes(mesh=meshes, movement=fg, f=scalars, subplot=cords,
                       texture=textures, cmap=color_maps, plotter=plotter,
                       title=titles, font_size=font_size, font_color=font_colors,
                       gif_path="src/tests/temp/camera_positions.gif",
                       camera=cam, depth=False
                       )


def annimate_three_wings():
    # the movement of all meshes
    g1 = []
    f1 = []
    f2 = []
    g2 = []
    f3 = []
    g3 = []
    # we need to create 6 different meshes, three of tips and three for wing. Pyvista will not recognise the
    # meshes as different otherwise.
    tip1 = Mesh('data/wing_off_files/fem_tip.off')
    tip2 = Mesh('data/wing_off_files/fem_tip.off')
    tip3 = Mesh('data/wing_off_files/fem_tip.off')
    mesh1 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    mesh2 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    mesh3 = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    meshes = [mesh1, tip1, mesh2, tip2, mesh3, tip3]
    # ^ define the order of each mesh
    frames = 60
    # number of frames of the gif, if no gif should be created this number should be around the 4000~ to make it
    # the same as 60~ with gif is created
    for phase in np.linspace(0, 4 * np.pi, frames + 1):
        f1.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh1.vertices,
                                     freq_t=1, freq_s=1, amp=0.05, t=phase,decay_rate_s=5))
        g1.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip1.vertices,
                                    freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5))

        f2.append(np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh2.vertices,
                                          freq_t=1, freq_s=25, amp=0.05, t=phase, decay_rate_s=5))
        g2.append(np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=tip2.vertices,
                                          freq_t=1, freq_s=25, amp=0.05, t=phase, decay_rate_s=5))

        f3.append(np.apply_along_axis(fem_wing_normal_sine, axis=1, arr=mesh3.vertices,
                                          freq_t=1, freq_s=25, amp=0.05, t=phase, decay_rate_s=5))

        g3.append(np.apply_along_axis(fem_tip_normal_sine, axis=1, arr=tip3.vertices,
                                          freq_t=1, freq_s=25, amp=0.05, t=phase, decay_rate_s=5))
        # couldnt vectorise

    fg = [f1, g1, f2, g2, f3, g3]
    cords = [(0, 0), (0, 0), (1, 0), (1, 0), (2, 0), (2, 0)]
    # cords of the subplot, both mesh are in the same subplot so both needing to be the same
    plotter = pv.Plotter(shape=(3, 1))

    mesh1.main_cords(plot=True, index_row=0, index_col=0, scale=0.1, plotter=plotter, show=False)
    mesh2.main_cords(plot=True, index_row=1, index_col=0, scale=0.1, plotter=plotter, show=False)
    mesh3.main_cords(plot=True, index_row=2, index_col=0, scale=0.1, plotter=plotter, show=False)
    scalars = [None] * 6
    textures = ["data/textures/checkers2.png", None] * 3
    color_maps = ["jet"] * 6
    titles = ["big wave length", "", "small wave length", "", "non decaying sin", ""]
    font_colors = ["black"] * 6
    font_size = [10, 10, 10, 10, 10, 10]
    cam = [(0.005, -0.2, 0.01), (0.047, 0.3, 0), (0, 0, 1)]
    animate_few_meshes(mesh=meshes, movement=fg, f=scalars, subplot=cords,
                       texture=textures, cmap=color_maps, plotter=plotter,
                       title=titles, font_size=font_size, font_color=font_colors,
                       gif_path="src/tests/temp/three_red_wings2.gif",
                       camera=[cam, cam, cam, cam, cam, cam], depth=False
                       )
    # ^ every argument should be given as a list, the default args for this function is for a single mesh, not more
    # self.mesh.animate(movement=f, texture="data/textures/cat.jpg", gif_path="src/tests/temp/")
    # ^ would animate a single mesh in a single subplot


def depth_video():
    mesh = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    mesh2 = Mesh("data/wing_off_files/fem_tip.off")
    res = [480,480]
    plotter = pv.Plotter(off_screen=True)
    frames = 40
    url = "src/tests/temp/video_frames/"
    i = 0
    im_frames = []
    for phase in np.linspace(0, 4 * np.pi, frames):
        i = i + 1
        f1 = np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh.vertices,
                                     freq_t=1, freq_s=1, amp=0.2, t=phase)
        g1 = np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=mesh2.vertices,
                                     freq_t=1, freq_s=1, amp=0.2, t=phase)

        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                                   texture=["data/textures/checkers2.png", None],
                                   cmap=[None, None], camera=camera_pos["up_left"], resolution=res, title="up left")
        depth = photo[:, :, -1]
        depth_f1 = (((depth - depth.min()) / depth.max()) * 255).astype('uint8')
        cv2.imwrite(url + "depth_frameA" + str(i) + ".jpg", depth_f1)
        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                                   texture=["data/textures/checkers2.png", None],
                                   cmap=[None, None], camera=camera_pos["up_middle"], resolution=res, title="")
        depth = photo[:, :, -1]
        depth_f2 = (((depth - depth.min()) / depth.max()) * 255).astype('uint8')
        cv2.imwrite(url + "depth_frameB" + str(i) + ".jpg", depth_f2)
        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                               texture=["data/textures/checkers2.png", None],
                               cmap=[None, None], camera=camera_pos["up_right"], resolution=res, title="")
        depth = photo[:, :, -1]
        depth_f3 = (((depth - depth.min()) / depth.max()) * 255).astype('uint8')
        cv2.imwrite(url + "depth_frameC" + str(i) + ".jpg", depth_f3)

        img1 = cv2.imread(url + "depth_frameA" + str(i) + ".jpg")
        img2 = cv2.imread(url + "depth_frameB" + str(i) + ".jpg")
        img3 = cv2.imread(url + "depth_frameC" + str(i) + ".jpg")
        img_u = cv2.hconcat([img3, img2, img1])

        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                                texture=["data/textures/checkers2.png", None],
                                cmap=[None, None], camera=camera_pos["down_left"], resolution=res,
                               title="")
        depth = photo[:, :, -1]
        depth_f4 = (((depth - depth.min()) / depth.max()) * 255).astype('uint8')
        cv2.imwrite(url + "depth_frameD" + str(i) + ".jpg", depth_f4)
        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                                   texture=["data/textures/checkers2.png", None],
                                   cmap=[None, None], camera=camera_pos["down_middle"], resolution=res,
                                    title="down middle")
        depth = photo[:, :, -1]
        depth_f5 = (((depth - depth.min()) / depth.max()) * 255).astype('uint8')
        cv2.imwrite(url + "depth_frameE" + str(i) + ".jpg", depth_f5)
        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                               texture=["data/textures/checkers2.png", None],
                               cmap=[None, None], camera=camera_pos["down_right"], resolution=res, title="")
        depth = photo[:, :, -1]
        depth_f6 = (((depth - depth.min()) / depth.max()) * 255).astype('uint8')
        cv2.imwrite(url + "depth_frameF" + str(i) + ".jpg", depth_f6)

        img1 = cv2.imread(url + "depth_frameD" + str(i) + ".jpg")
        img2 = cv2.imread(url + "depth_frameE" + str(i) + ".jpg")
        img3 = cv2.imread(url + "depth_frameF" + str(i) + ".jpg")
        img_d = cv2.hconcat([img3, img2, img1])
        img_f = cv2.vconcat([img_u, img_d])
        cv2.imshow("frame", img_f)
        im_frames.append(img_f)
        # cv2 does not support making video from np array...
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    out = cv2.VideoWriter("src/tests/temp/depth_video2.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 15, (1440, 960))
    for i in range(len(im_frames)):
        out.write(im_frames[i])
    out.release()
    for f in glob.glob(url + '*.jpg'):
       os.remove(f)
    cv2.destroyAllWindows()


def normal_video():
    mesh = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    mesh2 = Mesh("data/wing_off_files/fem_tip.off")
    res = [480, 480]
    plotter = pv.Plotter(off_screen=True)
    frames = 40
    url = "src/tests/temp/video_frames/"
    i = 0
    im_frames = []
    for phase in np.linspace(0, 4 * np.pi, frames):
        i = i + 1
        f1 = np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh.vertices,
                                 freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5)
        g1 = np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=mesh2.vertices,
                                 freq_t=1, freq_s=1, amp=0.05, t=phase)

        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                               texture=["data/textures/checkers2.png", None],
                               cmap=[None, None], camera=camera_pos["up_left"], resolution=res, title="up left")
        depth = photo[:, :,0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameA" + str(i) + ".jpg", depth)
        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                               texture=["data/textures/checkers2.png", None],
                               cmap=[None, None], camera=camera_pos["up_middle"], resolution=res, title="")
        depth = photo[:, :, 0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameB" + str(i) + ".jpg", depth)
        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                               texture=["data/textures/checkers2.png", None],
                               cmap=[None, None], camera=camera_pos["up_right"], resolution=res, title="")
        depth = photo[:, :, 0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameC" + str(i) + ".jpg", depth)

        img1 = cv2.imread(url + "depth_frameA" + str(i) + ".jpg")
        img2 = cv2.imread(url + "depth_frameB" + str(i) + ".jpg")
        img3 = cv2.imread(url + "depth_frameC" + str(i) + ".jpg")
        img_u = cv2.hconcat([img3, img2, img1])

        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                               texture=["data/textures/checkers2.png", None],
                               cmap=[None, None], camera=camera_pos["down_left"], resolution=res,
                               title="")
        depth = photo[:, :, 0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameD" + str(i) + ".jpg", depth)
        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                               texture=["data/textures/checkers2.png", None],
                               cmap=[None, None], camera=camera_pos["down_middle"], resolution=res,
                               title="down middle")
        depth = photo[:, :, 0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameE" + str(i) + ".jpg", depth)
        photo = Mesh.get_photo([mesh, mesh2], [f1, g1], plotter=plotter,
                               texture=["data/textures/checkers2.png", None],
                               cmap=[None, None], camera=camera_pos["down_right"], resolution=res, title="")
        depth = photo[:, :, 0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameF" + str(i) + ".jpg", depth)

        img1 = cv2.imread(url + "depth_frameD" + str(i) + ".jpg")
        img2 = cv2.imread(url + "depth_frameE" + str(i) + ".jpg")
        img3 = cv2.imread(url + "depth_frameF" + str(i) + ".jpg")
        img_d = cv2.hconcat([img3, img2, img1])
        img_f = cv2.vconcat([img_u, img_d])
        cv2.imshow("frame", img_f)
        im_frames.append(img_f)
        # cv2 does not support making video from np array...
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out = cv2.VideoWriter("src/tests/temp/depth_video2.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 15, (1440, 960))
    for i in range(len(im_frames)):
        out.write(im_frames[i])
    out.release()
    for f in glob.glob(url + '*.jpg'):
        os.remove(f)
    cv2.destroyAllWindows()


def ir_video():
    # 753, 9, 23, 120, 108, 38, 169, 1084, 53, 393, 1416, 68, 378, 1748, 83, 143, 2079, 190, 348, 2537, 205,333, 3559,
    # 220, 318, 2648, 235, 303, 2711, 250, 3409, 3424, 3439
    ids = [753, 9, 23, 120, 108, 38, 169, 1084, 53, 393, 1416, 68, 378, 1748, 83, 143, 3416, 3434,
           2079, 190, 348, 2537, 205, 333, 3559, 220, 318, 2648, 235, 303, 2711, 250, 3409, 3424, 3439]
    tip_ids = [906, 727, 637, 457, 368, 142, 7]
    mesh = Mesh('data/wing_off_files/finished_fem_without_tip.off')
    mesh2 = Mesh("data/wing_off_files/fem_tip.off")
    url = "src/tests/temp/video_frames/"
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")
    camera = camera_pos["down_right"]
    plotter.set_position(camera[0])
    plotter.set_focus(camera[1])
    plotter.set_viewup(camera[2])
    resolution = [480, 480]
    im_frames = []
    frames = 40
    i=0
    plotter.add_mesh(mesh.pv_mesh, name="mesh")
    plotter.add_mesh(mesh2.pv_mesh, name="mesh2")
    for phase in np.linspace(0, 4 * np.pi, frames):
        i = i + 1
        f1 = np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh.vertices,
                                 freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5)
        g1 = np.apply_along_axis(fem_tip_sine_decaying_in_space,axis=1, arr=mesh2.vertices,
                                 freq_t=1, freq_s=1, amp=0.05, t=phase)
        for id2, tip_id in enumerate(tip_ids):
            plotter.add_mesh(mesh=pv.Sphere(center=g1[tip_id], radius=0.003), color='black', name=str(id2)+"t")
        for id, v_id in enumerate(ids):
            plotter.add_mesh(mesh=pv.Sphere(center=f1[v_id], radius=0.003), color='black', name=str(id))
        plotter.update_coordinates(f1, mesh.pv_mesh)
        plotter.update_coordinates(g1, mesh2.pv_mesh)
        plotter.show(auto_close=False, window_size=resolution)
        screen = plotter.screenshot(window_size=resolution)
        cv2.imshow("frame",screen)
        cv2.imwrite(url + "depth_frameE" + str(i) + ".jpg", screen)
        img = cv2.imread(url + "depth_frameE" + str(i) + ".jpg")
        im_frames.append(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break

    out = cv2.VideoWriter("src/tests/temp/ir_video.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 15, (480, 480))
    for i in range(len(im_frames)):
        out.write(im_frames[i])
    out.release()
    #for f in glob.glob(url + '*.jpg'):
       #os.remove(f)
    cv2.destroyAllWindows()


def laplasian_wing():
    mesh2 = Mesh('data/wing_off_files/mesh_for_laplace.off')
    mesh = []
    plotter = pv.Plotter(shape=(3,3), off_screen=False)
    threashold = [10,10,10,3,3,3,15,20,20]
    #  didnt work for all marks as 3, no threashold does

    T = []
    plotter.set_background("white")
    # eigvals,eigenfuncs,_,_ = laplacian_spectrum(v,f,k=k,decimals=4)
    _, func,_,_  = laplacian_spectrum(v=mesh2.vertices, f=mesh2.faces, k=9, decimals=4)
    print(mesh2.vertices.shape)
    subplots = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

    for i in range(9):
        mesh.append(Mesh('data/wing_off_files/mesh_for_laplace.off'))
        plotter.subplot(subplots[i][0], subplots[i][1])
        plotter.add_mesh(mesh[i].pv_mesh, name=str(i),scalars=func[:,i], clim=(np.min(func[:,i]), np.max(func[:,i])),
                         cmap="jet")

        if func[:,i][51] == 0:
            print("DANGER")
        T.append(func[:, i][51] > 0)
    bool_arr = func[:,8] <0
    NN = (np.sum(bool_arr))

    plotter.open_movie("src/tests/temp/laplacian_wing.mp4")
    plotter.show(auto_close=False)
    frames = 40
    for phase in np.linspace(0, 4 * np.pi, frames + 1):
        curr = np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1, arr=mesh2.vertices,
                                      freq_t=1, freq_s=1, amp=0.05, t=phase, decay_rate_s=5)
        _,func,_,_ = laplacian_spectrum(curr,f=mesh2.faces,k=9,decimals=4)

        for k in range(9):
            if k == 8:
               bool_arr = func[:,k] < 0
               if np.abs(np.sum(bool_arr) - NN) > threashold[k]:
                   func[:,k] *= -1
            if (func[:, k][51] > 0) == T[k] and k<8:
                func[:,k] *= -1

            plotter.subplot(subplots[k][0], subplots[k][1])
            plotter.update_coordinates(curr, mesh=mesh[k].pv_mesh)
            plotter.update_scalars(scalars=func[:,k], mesh=mesh[k].pv_mesh)

        plotter.write_frame()
    plotter.close()
