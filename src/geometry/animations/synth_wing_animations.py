

import glob
import os
import cv2
from scipy.io import loadmat
from PIL import Image
from src.geometry.numpy.transforms import *
from src.geometry.numpy.wing_models import *
from src.geometry.spod import *
from src.geometry.numpy.lbo import *
from SSIM_PIL import compare_ssim

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


def synth_wing_animation(path):
    mat=loadmat("data/synt_data_mat_files/data.mat")
    x = mat['U1']
    y = mat['U2']
    z = mat['U3']
    tip = Mesh('data/wing_off_files/fem_tip.off')
    mesh = Mesh('data/wing_off_files/synth_wing_v3.off')
    TIP_RADIUS = 0.008
    NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
    tip_vertices_num = 930
    new_tip_position = np.zeros((tip_vertices_num, 3), dtype='float')
    tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE, endpoint=False)
    y_t = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
    z_t = TIP_RADIUS * np.sin(tip_vertex_gain_arr)
    tip_index_arr = tip_arr_creation(mesh.vertices)
    tex = "data/textures/circles_tex.png"
    res = [480,480]
    plotter = pv.Plotter(off_screen=True)
    frames = 1000
    url = "src/tests/temp/video_frames/"
    i = 0
    im_frames = []
    f1 = np.zeros(mesh.vertices.shape)
    for phase in range(frames):
        i = i + 1
        f1[:,0] = x[:, phase] + mesh.vertices[:, 0]
        f1[:,1] = y[:, phase] + mesh.vertices[:, 1]
        f1[:,2] = z[:, phase] + mesh.vertices[:, 2]
        synth_tip_movement(mesh_ver=mesh.vertices, tip_index=tip_index_arr, x=x, y=y, z=z, y_t=y_t, z_t=z_t,
                           tip_table=tip.table, new_tip_position=new_tip_position, t=phase)
        photo = Mesh.get_photo([mesh, tip], [f1, new_tip_position], plotter=plotter, texture=[tex, None],
                               cmap=None, camera=camera_pos["up_right"], resolution=res, title="up right")
        depth = photo[:, :,0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameA" + str(i) + ".jpg", np.asarray(depth* 255,np.uint8))
        photo = Mesh.get_photo([mesh, tip], [f1, new_tip_position], plotter=plotter, texture=[tex,None],
                               cmap=None, camera=camera_pos["up_middle"], resolution=res, title="up middle")
        depth = photo[:, :,0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameB" + str(i) + ".jpg", np.asarray(depth* 255,np.uint8))
        photo = Mesh.get_photo([mesh,tip], [f1, new_tip_position], plotter=plotter, texture=[tex,None],
                               cmap=None, camera=camera_pos["up_left"], resolution=res, title="up left")
        depth = photo[:, :,0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameC" + str(i) + ".jpg", np.asarray(depth* 255,np.uint8))

        img1 = cv2.imread(url + "depth_frameA" + str(i) + ".jpg")
        img2 = cv2.imread(url + "depth_frameB" + str(i) + ".jpg")
        img3 = cv2.imread(url + "depth_frameC" + str(i) + ".jpg")
        img_u = cv2.hconcat([img3, img2, img1])

        photo = Mesh.get_photo([mesh,tip], [f1, new_tip_position], plotter=plotter, texture=[tex,None],
                               cmap=None, camera=camera_pos["rotated_down_right"], resolution=res, title="down right",
                               title_location="lower_edge")
        depth = photo[:, :,0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameD" + str(i) + ".jpg", np.asarray(depth* 255,np.uint8))
        photo = Mesh.get_photo([mesh,tip], [f1, new_tip_position], plotter=plotter, texture=[tex,None], cmap=None,
                               camera=camera_pos["rotated_down_middle"], resolution=res, title="down middle",
                               title_location="lower_edge")
        depth = photo[:, :,0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameE" + str(i) + ".jpg", np.asarray(depth* 255,np.uint8))
        photo = Mesh.get_photo([mesh,tip], [f1, new_tip_position], plotter=plotter, texture=[tex,None], cmap=None,
                               camera=camera_pos["rotated_down_left"], resolution=res, title="down left",
                               title_location="lower_edge")

        depth = photo[:, :,0:3]
        r= np.copy(photo[:,:,2])
        depth[:,:,2] = depth[:,:,0]
        depth[:,:,0] = r
        cv2.imwrite(url + "depth_frameF" + str(i) + ".jpg", np.asarray(depth* 255,np.uint8))

        img1 = cv2.imread(url + "depth_frameD" + str(i) + ".jpg")
        img2 = cv2.imread(url + "depth_frameE" + str(i) + ".jpg")
        img3 = cv2.imread(url + "depth_frameF" + str(i) + ".jpg")
        img_d = cv2.hconcat([img3, img2, img1])
        img_f = cv2.vconcat([img_u, img_d])
        #7,cv2.imshow("frame", img_f)
        im_frames.append(img_f)
        # cv2 does not support making video from np array...
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (1440, 960))
    for i in range(len(im_frames)):
        out.write(im_frames[i])
    out.release()
    for f in glob.glob(url + '*.jpg'):
       os.remove(f)
    cv2.destroyAllWindows()


def synth_ir_video(path):
    TIP_RADIUS = 0.008
    NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
    tip_vertices_num = 930
    ids = [6419, 6756, 7033, 7333, 7635, 7937, 8239, 8541, 8841,  # first line
           6411, 6727, 7025, 7325, 7627, 7929, 8271, 8553, 8854,  # middle
           6361, 6697, 6974, 7315, 7576, 7919, 8199, 8482, 8782]
    mat = loadmat("data/synt_data_mat_files/data.mat")
    x = mat['U1']
    y = mat['U2']
    z = mat['U3']
    mesh = Mesh('data/wing_off_files/synth_wing_v3.off')
    tip = Mesh('data/wing_off_files/fem_tip.off')
    new_tip_position = np.zeros((tip_vertices_num, 3), dtype='float')
    tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE, endpoint=False)
    y_t = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
    z_t = TIP_RADIUS * np.sin(tip_vertex_gain_arr)
    tip_index_arr = tip_arr_creation(mesh.vertices)
    url = "src/tests/temp/video_frames/"
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")
    camera = camera_pos["up_middle"]
    plotter.set_position(camera[0])
    plotter.set_focus(camera[1])
    plotter.set_viewup(camera[2])
    resolution = [480, 480]
    im_frames = []
    frames = 100
    i = 0
    plotter.add_mesh(mesh.pv_mesh, name="mesh", style='wireframe')
    plotter.add_mesh(tip.pv_mesh, name="mesh2", style='wireframe')
    f1 = np.zeros(mesh.vertices.shape)
    for phase in range(frames):
        i = i + 1
        f1[:, 0] = x[:, phase] + mesh.vertices[:, 0]
        f1[:, 1] = y[:, phase] + mesh.vertices[:, 1]
        f1[:, 2] = z[:, phase] + mesh.vertices[:, 2]

        synth_tip_movement(mesh_ver=mesh.vertices, tip_index=tip_index_arr, x=x, y=y, z=z, y_t=y_t, z_t=z_t,
                           tip_table=tip.table, new_tip_position=new_tip_position, t=phase)
        for id, v_id in enumerate(ids):
            plotter.add_mesh(mesh=pv.Sphere(center=f1[v_id], radius=0.003), color='red', name=str(id))

        plotter.update_coordinates(f1, mesh.pv_mesh)
        plotter.update_coordinates(new_tip_position, tip.pv_mesh)
        plotter.show(auto_close=False, window_size=resolution)
        screen = plotter.screenshot(window_size=resolution)
        cv2.imwrite(url + "depth_frameE" + str(i) + ".jpg", screen)
        img = cv2.imread(url + "depth_frameE" + str(i) + ".jpg")
        im_frames.append(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (480, 480))
    for i in range(len(im_frames)):
        out.write(im_frames[i])
    out.release()
    # for f in glob.glob(url + '*.jpg'):
    # os.remove(f)
    cv2.destroyAllWindows()


def modal_animation(path):
    wing_path = "data/wing_off_files/synth_wing_v3.off"
    tip_path = 'data/wing_off_files/fem_tip.off'
    mesh = Mesh(wing_path)
    tip = Mesh('data/wing_off_files/fem_tip.off')
    TIP_RADIUS = 0.008
    NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
    tip_vertices_num = 930
    new_tip_position1 = np.zeros((tip_vertices_num, 3), dtype='float')
    new_tip_position2 = np.zeros((tip_vertices_num, 3), dtype='float')
    new_tip_position3 = np.zeros((tip_vertices_num, 3), dtype='float')
    new_tip_position4 = np.zeros((tip_vertices_num, 3), dtype='float')
    new_tip_position5 = np.zeros((tip_vertices_num, 3), dtype='float')
    new_tip_position6 = np.zeros((tip_vertices_num, 3), dtype='float')
    tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE, endpoint=False)
    y_t = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
    z_t = TIP_RADIUS * np.sin(tip_vertex_gain_arr)
    tip_index_arr = tip_arr_creation(mesh.vertices)
    mode_shapes = loadmat("data/synt_data_mat_files/mode_shapes.mat")["ModeShapes"]
    modes = np.zeros((mesh.vertices.shape[0],6,3))
    modes[:, :, 0] = loadmat("data/synt_data_mat_files/modes.mat")["T1"][:, 0:6]
    modes[:, :, 1] = loadmat("data/synt_data_mat_files/modes.mat")["T2"][:, 0:6]
    modes[:, :, 2] = loadmat("data/synt_data_mat_files/modes.mat")["T3"][:, 0:6]
    frames = 40
    g1 = np.zeros((frames,tip.vertices.shape[0],3))
    f1 = np.zeros((frames,mesh.vertices.shape[0],3))
    f2 = np.zeros((frames,mesh.vertices.shape[0],3))
    g2 = np.zeros((frames,tip.vertices.shape[0],3))
    f3 = np.zeros((frames,mesh.vertices.shape[0],3))
    g3 = np.zeros((frames,tip.vertices.shape[0],3))
    g4 = np.zeros((frames,tip.vertices.shape[0],3))
    f4 = np.zeros((frames,mesh.vertices.shape[0],3))
    g5 = np.zeros((frames,tip.vertices.shape[0],3))
    f5 = np.zeros((frames,mesh.vertices.shape[0],3))
    g6 = np.zeros((frames,tip.vertices.shape[0],3))
    f6 = np.zeros((frames,mesh.vertices.shape[0],3))
    # we need to create 6 different meshes, three of tips and three for wing. Pyvista will not recognise the
    # meshes as different otherwise.
    tip1 = Mesh(tip_path)
    tip2 = Mesh(tip_path)
    tip3 = Mesh(tip_path)
    tip4 = Mesh(tip_path)
    tip5 = Mesh(tip_path)
    tip6 = Mesh(tip_path)
    mesh1 = Mesh(wing_path)
    mesh2 = Mesh(wing_path)
    mesh3 = Mesh(wing_path)
    mesh4 = Mesh(wing_path)
    mesh5 = Mesh(wing_path)
    mesh6 = Mesh(wing_path)
    frames = 40
    # number of frames of the gif, if no gif should be created this number should be around the 4000~ to make it
    # the same as 60~ with gif is created
    for idx, phase in enumerate(np.linspace(0, 0.02, frames)):

        f1[idx, :, :] = phase * modes[:, 0, :] + mesh.vertices
        f2[idx, :, :] = phase * modes[:, 1, :] + mesh.vertices
        f3[idx, :, :] = phase * modes[:, 2, :] + mesh.vertices
        f4[idx, :, :] = phase * modes[:, 3, :] + mesh.vertices
        f5[idx, :, :] = phase * modes[:, 4, :] + mesh.vertices
        f6[idx, :, :] = phase * modes[:, 5, :] + mesh.vertices



        for id in tip_index_arr:
            for i in range(30):
                cord = mesh.vertices[id]
                vec1 = np.array((cord[0], cord[1] + y_t[i], cord[2] + z_t[i]))
                vec1 += phase * modes[id,0,:]
                vec2 = np.array((cord[0], cord[1] + y_t[i], cord[2] + z_t[i]))
                vec2 += phase * modes[id, 1, :]
                vec3 = np.array((cord[0], cord[1] + y_t[i], cord[2] + z_t[i]))
                vec3 += phase * modes[id,2,:]
                vec4 = np.array((cord[0], cord[1] + y_t[i], cord[2] + z_t[i]))
                vec4 += phase * modes[id,3,:]
                vec5 = np.array((cord[0], cord[1] + y_t[i], cord[2] + z_t[i]))
                vec5 += phase * modes[id,4,:]
                vec6 = np.array((cord[0], cord[1] + y_t[i], cord[2] + z_t[i]))
                vec6 += phase * modes[id,5,:]

                new_tip_position1[tip1.table[cord2index(cord + (0, y_t[i], z_t[i]))]] = vec1
                new_tip_position2[tip1.table[cord2index(cord + (0, y_t[i], z_t[i]))]] = vec2
                new_tip_position3[tip1.table[cord2index(cord + (0, y_t[i], z_t[i]))]] = vec3
                new_tip_position4[tip1.table[cord2index(cord + (0, y_t[i], z_t[i]))]] = vec4
                new_tip_position5[tip1.table[cord2index(cord + (0, y_t[i], z_t[i]))]] = vec5
                new_tip_position6[tip1.table[cord2index(cord + (0, y_t[i], z_t[i]))]] = vec6

        g1[idx] = new_tip_position1
        g2[idx] = new_tip_position2
        g3[idx] = new_tip_position3
        g4[idx] = new_tip_position4
        g5[idx] = new_tip_position5
        g6[idx] = new_tip_position6

    fg = [f1, g1, f2, g2, f3, g3, f4, g4, f5, g5, f6, g6]
    cords = [(0, 0), (0, 0), (0, 1), (0, 1), (0, 2), (0, 2), (1, 0), (1, 0), (1, 1), (1, 1), (1, 2), (1, 2), (2, 1),
             (2, 1)]
    scalars = [None] * 12
    textures = ["data/textures/checkers_dark_blue.png", None] * 6
    color_maps = ["jet"] * 12
    titles = ["first mode", "", "second mod", "", "third mode", "", "forth mode", "", "fifth mode", "", "sixth mode",""]
    font_colors = ["black"] * 12
    font_size = [10] * 12
    cam = [camera_pos["up_middle"]] * 12
    plotter = pv.Plotter(shape=(2,3))
    plotter.set_background("white")
    meshes = [mesh1, tip1, mesh2, tip2, mesh3, tip3, mesh4, tip4, mesh5, tip5, mesh6, tip6]

    animate_few_meshes(mesh=meshes, movement=fg, f=scalars, subplot=cords,
                       texture=textures, cmap=color_maps, plotter=plotter,
                       title=titles, font_size=font_size, font_color=font_colors,
                       gif_path=path,
                       camera=cam, depth=False
                       )

    pass


def scale_made_movement(path, amp):
    mat = loadmat("data/synt_data_mat_files/data.mat")
    mode_x = loadmat("data/synt_data_mat_files/modes.mat")["T1"][:,0:5]
    mode_y = loadmat("data/synt_data_mat_files/modes.mat")["T2"][:,0:5]
    mode_z = loadmat("data/synt_data_mat_files/modes.mat")["T3"][:,0:5]
    x = mat['U1']
    y = mat['U2']
    z = mat['U3']
    scale = mat['xi']
    tip = Mesh('data/wing_off_files/fem_tip.off')
    tip2 = Mesh('data/wing_off_files/fem_tip.off')
    mesh = Mesh('data/wing_off_files/synth_wing_v3.off')
    mesh2 = Mesh('data/wing_off_files/synth_wing_v3.off')
    TIP_RADIUS = 0.008
    NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
    tip_vertices_num = 930
    new_tip_position = np.zeros((tip_vertices_num, 3), dtype='float')
    new_tip_position2 = np.zeros((tip_vertices_num, 3), dtype='float')
    tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE, endpoint=False)
    y_t = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
    z_t = TIP_RADIUS * np.sin(tip_vertex_gain_arr)
    tip_index_arr = tip_arr_creation(mesh.vertices)
    tex = "data/textures/circles_tex.png"
    tex2 = "data/textures/rainbow.png"
    tex3 = "data/textures/checkers_dark_blue.png"
    res = [480, 480]
    plotter = pv.Plotter(off_screen=True)
    plotter2 = pv.Plotter(off_screen=True)
    frames = 500
    url = "src/tests/temp/video_frames/"
    i = 0
    im_frames = []
    f1 = np.zeros(mesh.vertices.shape)
    f2 = np.copy(mesh.vertices)

    for phase in range(frames):
        i = i + 1
        f1[:, 0] = x[:, phase] + mesh.vertices[:, 0]
        f1[:, 1] = y[:, phase] + mesh.vertices[:, 1]
        f1[:, 2] = z[:, phase] + mesh.vertices[:, 2]

        #difference = (mode_shapes[:, 0] * scale[0, phase] + mode_shapes[:, 1] * scale[1, phase] +
        #                        mode_shapes[:, 2] * scale[2, phase] + mode_shapes[:, 3] * scale[3, phase] +
        #                        mode_shapes[:, 4] * scale[4, phase]) * amp
        difference_x = (scale[:,phase] * mode_x).sum(axis=1) * amp
        difference_y = (scale[:, phase] * mode_y).sum(axis=1) * amp
        difference_z = (scale[:, phase] * mode_z).sum(axis=1) * amp

        f2[:,0] = mesh.vertices[:,0] + difference_x
        f2[:,1] = mesh.vertices[:,1] + difference_y
        f2[:,2] = mesh.vertices[:, 2] + difference_z

        for id in tip_index_arr:
            for i in range(30):
                cord = mesh2.vertices[id]
                vec = np.array((cord[0] + difference_x[id], cord[1] + y_t[i] + difference_y[id],
                                cord[2] + z_t[i] + difference_z[id]))

                new_tip_position2[tip.table[cord2index(cord + (0, y_t[i], z_t[i]))]] = vec

        synth_tip_movement(mesh_ver=mesh.vertices, tip_index=tip_index_arr, x=x, y=y, z=z, y_t=y_t, z_t=z_t,
                           tip_table=tip.table, new_tip_position=new_tip_position, t=phase)

        photo = Mesh.get_photo([mesh, tip], [f1, new_tip_position], plotter=plotter2, texture=[tex, None],
                               cmap=None, camera=camera_pos["up_right"], resolution=res)
        depth11 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth11[:, :, 2] = depth11[:, :, 0]
        depth11[:, :, 0] = r
        cv2.imwrite(url + "depth_frameA" + str(i) + ".jpg", np.asarray(depth11 * 255, np.uint8))
        photo = Mesh.get_photo([mesh, tip], [f1, new_tip_position], plotter=plotter2, texture=[tex2, None],
                               cmap=None, camera=camera_pos["up_middle"], resolution=res, title=None)
        depth12 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth12[:, :, 2] = depth12[:, :, 0]
        depth12[:, :, 0] = r
        cv2.imwrite(url + "depth_frameB" + str(i) + ".jpg", np.asarray(depth12 * 255, np.uint8))
        photo = Mesh.get_photo([mesh, tip], [f1, new_tip_position], plotter=plotter2, texture=[tex3, None],
                               cmap=None, camera=camera_pos["up_left"], resolution=res)
        depth13 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth13[:, :, 2] = depth13[:, :, 0]
        depth13[:, :, 0] = r
        cv2.imwrite(url + "depth_frameC" + str(i) + ".jpg", np.asarray(depth13 * 255, np.uint8))

        img1 = cv2.imread(url + "depth_frameA" + str(i) + ".jpg")
        img2 = cv2.imread(url + "depth_frameB" + str(i) + ".jpg")
        img3 = cv2.imread(url + "depth_frameC" + str(i) + ".jpg")
        cv2.putText(img2, "created by displacement", (50, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 0),
                    lineType=2)
        img_u = cv2.hconcat([img3, img2, img1])

        photo = Mesh.get_photo([mesh2, tip2], [f2, new_tip_position2], plotter=plotter, texture=[tex, None],
                               cmap=None, camera=camera_pos["up_right"], resolution=res,
                               title_location="lower_edge")
        depth21 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth21[:, :, 2] = depth21[:, :, 0]
        depth21[:, :, 0] = r

        cv2.imwrite(url + "depth_frameD" + str(i) + ".jpg", np.asarray(depth21 * 255, np.uint8))
        photo = Mesh.get_photo([mesh2, tip2], [f2, new_tip_position2], plotter=plotter, texture=[tex2, None], cmap=None,
                               camera=camera_pos["up_middle"], resolution=res, title=None,
                               )
        depth22 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth22[:, :, 2] = depth22[:, :, 0]
        depth22[:, :, 0] = r
        cv2.imwrite(url + "depth_frameE" + str(i) + ".jpg", np.asarray(depth22 * 255, np.uint8))
        photo = Mesh.get_photo([mesh2, tip2], [f2, new_tip_position2], plotter=plotter, texture=[tex3, None], cmap=None,
                               camera=camera_pos["up_left"], resolution=res, title=None,
                               title_location="lower_edge")

        depth23 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth23[:, :, 2] = depth23[:, :, 0]
        depth23[:, :, 0] = r
        cv2.imwrite(url + "depth_frameF" + str(i) + ".jpg", np.asarray(depth23 * 255, np.uint8))

        img12 = cv2.imread(url + "depth_frameD" + str(i) + ".jpg")
        img22 = cv2.imread(url + "depth_frameE" + str(i) + ".jpg")
        img32 = cv2.imread(url + "depth_frameF" + str(i) + ".jpg")

        pil_img1 = Image.fromarray(np.uint8(depth11 * 255))
        pil_img2 = Image.fromarray(np.uint8(depth21 * 255))
        dist1 = compare_ssim(pil_img1, pil_img2)
        pil_img1 = Image.fromarray(np.uint8(depth12 * 255))
        pil_img2 = Image.fromarray(np.uint8(depth22 * 255))
        dist2 = compare_ssim(pil_img1, pil_img2)
        pil_img1 = Image.fromarray(np.uint8(depth13 * 255))
        pil_img2 = Image.fromarray(np.uint8(depth23 * 255))
        dist3 = compare_ssim(pil_img1, pil_img2)


        cv2.putText(img12, "ssim value = " + str(dist1), (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0),
                    lineType=2)
        cv2.putText(img22, "ssim value = " + str(dist2), (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0),
                    lineType=2)
        cv2.putText(img32, "ssim value = " + str(dist3), (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0),
                    lineType=2)
        cv2.putText(img22, "created by the mode shapes", (50, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 0),
                    lineType=2)
        img_d = cv2.hconcat([img32, img22, img12])
        img_f = cv2.vconcat([img_u, img_d])
        # 7,cv2.imshow("frame", img_f)
        im_frames.append(img_f)
        # cv2 does not support making video from np array...
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (1440, 960))

    for i in range(len(im_frames)):
        out.write(im_frames[i])
    out.release()
    for f in glob.glob(url + '*.jpg'):
        os.remove(f)
    cv2.destroyAllWindows()


def create_vid_by_scales(scale1, scale2, vid_path, trash_path, texture_path, mode_shape_path, frames, num_of_scales,
                         show_ssim=True):
    #mat = loadmat("data/synt_data_mat_files/data.mat")
    mode_x = loadmat(mode_shape_path)["T1"][:, 0:num_of_scales]
    mode_y = loadmat(mode_shape_path)["T2"][:, 0:num_of_scales]
    mode_z = loadmat(mode_shape_path)["T3"][:, 0:num_of_scales]
    #x = mat['U1']
    #y = mat['U2']
    #z = mat['U3']
    #scale = mat['xi']
    tip = Mesh('data/wing_off_files/fem_tip.off')
    tip2 = Mesh('data/wing_off_files/fem_tip.off')
    mesh = Mesh('data/wing_off_files/synth_wing_v3.off')
    mesh2 = Mesh('data/wing_off_files/synth_wing_v3.off')
    TIP_RADIUS = 0.008
    NUM_OF_VERTICES_ON_CIRCUMFERENCE = 30
    tip_vertices_num = 930
    new_tip_position = np.zeros((tip_vertices_num, 3), dtype='float')
    new_tip_position2 = np.zeros((tip_vertices_num, 3), dtype='float')
    tip_vertex_gain_arr = np.linspace(0, 2 * np.pi, NUM_OF_VERTICES_ON_CIRCUMFERENCE, endpoint=False)
    y_t = TIP_RADIUS * np.cos(tip_vertex_gain_arr)
    z_t = TIP_RADIUS * np.sin(tip_vertex_gain_arr)
    tip_index_arr = tip_arr_creation(mesh.vertices)
    res = [480, 480]
    plotter = pv.Plotter(off_screen=True)
    plotter2 = pv.Plotter(off_screen=True)
    i = 0
    im_frames = []
    f1 = np.zeros(mesh.vertices.shape)
    f2 = np.copy(mesh.vertices)

    for phase in range(frames):
        i = i + 1
        # difference = (mode_shapes[:, 0] * scale[0, phase] + mode_shapes[:, 1] * scale[1, phase] +
        #                        mode_shapes[:, 2] * scale[2, phase] + mode_shapes[:, 3] * scale[3, phase] +
        #                        mode_shapes[:, 4] * scale[4, phase]) * amp
        difference1_x = (scale1[:, phase] * mode_x).sum(axis=1)
        difference1_y = (scale1[:, phase] * mode_y).sum(axis=1)
        difference1_z = (scale1[:, phase] * mode_z).sum(axis=1)

        difference2_x = (scale2[:, phase] * mode_x).sum(axis=1)
        difference2_y = (scale2[:, phase] * mode_y).sum(axis=1)
        difference2_z = (scale2[:, phase] * mode_z).sum(axis=1)

        f1[:, 0] = mesh.vertices[:, 0] + difference1_x
        f1[:, 1] = mesh.vertices[:, 1] + difference1_y
        f1[:, 2] = mesh.vertices[:, 2] + difference1_z
        f2[:, 0] = mesh.vertices[:, 0] + difference2_x
        f2[:, 1] = mesh.vertices[:, 1] + difference2_y
        f2[:, 2] = mesh.vertices[:, 2] + difference2_z

        for id in tip_index_arr:
            for i in range(30):
                cord = mesh2.vertices[id]
                vec = np.array((cord[0] + difference1_x[id], cord[1] + y_t[i] + difference1_y[id],
                                cord[2] + z_t[i] + difference1_z[id]))

                vec2 = np.array((cord[0] + difference2_x[id], cord[1] + y_t[i] + difference2_y[id],
                                cord[2] + z_t[i] + difference2_z[id]))

                new_tip_position[tip.table[cord2index(cord + (0, y_t[i], z_t[i]))]] = vec
                new_tip_position2[tip.table[cord2index(cord + (0, y_t[i], z_t[i]))]] = vec2



        photo = Mesh.get_photo([mesh, tip], [f1, new_tip_position], plotter=plotter2, texture=[texture_path, None],
                               cmap=None, camera=camera_pos["up_right"], resolution=res)
        depth11 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth11[:, :, 2] = depth11[:, :, 0]
        depth11[:, :, 0] = r
        cv2.imwrite(trash_path + "depth_frameA" + str(i) + ".png", np.asarray(depth11 * 255, np.uint8))
        photo = Mesh.get_photo([mesh, tip], [f1, new_tip_position], plotter=plotter2, texture=[texture_path, None],
                               cmap=None, camera=camera_pos["up_middle"], resolution=res, title=None)
        depth12 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth12[:, :, 2] = depth12[:, :, 0]
        depth12[:, :, 0] = r
        cv2.imwrite(trash_path + "depth_frameB" + str(i) + ".png", np.asarray(depth12 * 255, np.uint8))
        photo = Mesh.get_photo([mesh, tip], [f1, new_tip_position], plotter=plotter2, texture=[texture_path, None],
                               cmap=None, camera=camera_pos["up_left"], resolution=res)
        depth13 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth13[:, :, 2] = depth13[:, :, 0]
        depth13[:, :, 0] = r
        cv2.imwrite(trash_path + "depth_frameC" + str(i) + ".png", np.asarray(depth13 * 255, np.uint8))

        img1 = cv2.imread(trash_path + "depth_frameA" + str(i) + ".png")
        img2 = cv2.imread(trash_path + "depth_frameB" + str(i) + ".png")
        img3 = cv2.imread(trash_path + "depth_frameC" + str(i) + ".png")
        cv2.putText(img2, "first scale", (50, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 0),
                    lineType=2)
        img_u = cv2.hconcat([img3, img2, img1])

        photo = Mesh.get_photo([mesh2, tip2], [f2, new_tip_position2], plotter=plotter, texture=[texture_path, None],
                               cmap=None, camera=camera_pos["up_right"], resolution=res)
        depth21 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth21[:, :, 2] = depth21[:, :, 0]
        depth21[:, :, 0] = r

        cv2.imwrite(trash_path + "depth_frameD" + str(i) + ".png", np.asarray(depth21 * 255, np.uint8))
        photo = Mesh.get_photo([mesh2, tip2], [f2, new_tip_position2], plotter=plotter, texture=[texture_path, None],
                               cmap=None, camera=camera_pos["up_middle"], resolution=res)
        depth22 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth22[:, :, 2] = depth22[:, :, 0]
        depth22[:, :, 0] = r
        cv2.imwrite(trash_path + "depth_frameE" + str(i) + ".png", np.asarray(depth22 * 255, np.uint8))
        photo = Mesh.get_photo([mesh2, tip2], [f2, new_tip_position2], plotter=plotter, texture=[texture_path, None],
                               cmap=None, camera=camera_pos["up_left"], resolution=res)

        depth23 = photo[:, :, 0:3]
        r = np.copy(photo[:, :, 2])
        depth23[:, :, 2] = depth23[:, :, 0]
        depth23[:, :, 0] = r
        cv2.imwrite(trash_path + "depth_frameF" + str(i) + ".png", np.asarray(depth23 * 255, np.uint8))

        img12 = cv2.imread(trash_path + "depth_frameD" + str(i) + ".png")
        img22 = cv2.imread(trash_path + "depth_frameE" + str(i) + ".png")
        img32 = cv2.imread(trash_path + "depth_frameF" + str(i) + ".png")
        if show_ssim:
            pil_img1 = Image.fromarray(np.uint8(depth11 * 255))
            pil_img2 = Image.fromarray(np.uint8(depth21 * 255))
            dist1 = compare_ssim(pil_img1, pil_img2)
            pil_img1 = Image.fromarray(np.uint8(depth12 * 255))
            pil_img2 = Image.fromarray(np.uint8(depth22 * 255))
            dist2 = compare_ssim(pil_img1, pil_img2)
            pil_img1 = Image.fromarray(np.uint8(depth13 * 255))
            pil_img2 = Image.fromarray(np.uint8(depth23 * 255))
            dist3 = compare_ssim(pil_img1, pil_img2)

            cv2.putText(img12, "ssim value =" + f'{dist1: .3e}', (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0),
                        lineType=2)
            cv2.putText(img22, "ssim value =" + f'{dist2: .3e}', (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0),
                        lineType=2)
            cv2.putText(img32, "ssim value =" + f'{dist3: .3e}', (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0),
                        lineType=2)
        cv2.putText(img22, "second scale", (50, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 0),
                    lineType=2)
        img_d = cv2.hconcat([img32, img22, img12])
        img_f = cv2.vconcat([img_u, img_d])
        # 7,cv2.imshow("frame", img_f)
        im_frames.append(img_f)
        # cv2 does not support making video from np array...
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (1440, 960))

    for i in range(len(im_frames)):
        out.write(im_frames[i])
    out.release()
    for f in glob.glob(trash_path + '*.png'):
        os.remove(f)
    cv2.destroyAllWindows()


