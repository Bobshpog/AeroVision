from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Tuple

import h5py
import numpy as np
import pyvista as pv

import src.data.data_generators.data_gen as data_gen
from src.data.matlab_reader import read_data, read_modal_shapes
from src.geometry.numpy.mesh import read_off_size
from src.geometry.numpy.wing_models import SyntheticWingModel
from src.util.memoization.memoization import cached


@dataclass(repr=False)
class SyntheticMatGenerator(data_gen.DataGenerator):
    mat_path: Union[Path, str] = field(repr=False)
    modal_shape_mat_path: Union[Path, str] = field(repr=False)
    mesh_wing_path: Union[Path, str]
    mesh_tip_path: Union[Path, str]
    old_mesh_wing_path: Union[Path, str]
    ir_list: list = field(repr=False)  # list of ids of points in mesh
    resolution: list  # [Width, Height]
    # cameras in pyvista format
    cameras: list = field(repr=False)
    texture_path_wing: Union[str, Path]

    cmap: str = field(repr=False)
    mesh_wing_path: Union[Path, str]
    mesh_tip_path: Union[Path, str]
    texture_path_tip: Union[None, str, Path] = field(repr=False, default=None)

    background_photos: Union[List[str], None] = field(default_factory=list)
    cam_noise_lambda: Tuple[float] = None

    def __post_init__(self):
        if isinstance(self.mat_path, str):
            self.mat_path = Path(self.mat_path)
        if isinstance(self.modal_shape_mat_path, str):
            self.modal_shape_mat_path = Path(self.modal_shape_mat_path)
        if isinstance(self.mesh_wing_path, str):
            self.mesh_wing_path = Path(self.mesh_wing_path)
        if isinstance(self.mesh_tip_path, str):
            self.mesh_tip_path = Path(self.mesh_tip_path)
        if isinstance(self.texture_path_wing, str):
            self.texture_path_wing = Path(self.texture_path_wing)
        if isinstance(self.texture_path_tip, str):
            self.texture_path_tip = Path(self.texture_path_tip)
        self.num_vertices_wing, _ = read_off_size(self.mesh_wing_path)
        self.num_vertices_tip, _ = read_off_size(self.mesh_tip_path)
        if self.background_photos is None:
            plotter = [pv.Plotter(off_screen=True)]
        else:
            plotter = [pv.Plotter(off_screen=True) for _ in self.background_photos]
            for path, plot in zip(self.background_photos, plotter):
                plot.add_background_image(path)
        self.cords, self.disp_arr, self.scales_arr = read_data(str(self.mat_path))
        self.num_frames, self.num_scales = self.scales_arr.shape
        self.wing_model = SyntheticWingModel(self.cords, self.ir_list, self.texture_path_wing, self.texture_path_tip,
                                             self.mesh_wing_path, self.mesh_tip_path, self.old_mesh_wing_path,
                                             self.cameras, self.num_vertices_wing, self.num_vertices_tip, plotter,
                                             self.resolution, self.cmap, background_photos=self.background_photos,
                                             cam_noise_lambda=self.cam_noise_lambda)

    def __len__(self):
        return self.num_frames

    def __repr__(self):
        return f"{self.__class__.__name__}(mesh_wing='{self.mesh_wing_path.name}', mesh_tip='{self.mesh_tip_path.name}'" \
               f", resolution={self.resolution}, texture_path='{self.texture_path_wing.name}'"

    def save_metadata(self, hdf5: h5py.File, group_name: str) -> None:
        group = hdf5.create_group(group_name)
        group.create_dataset('cords', dtype=np.float, data=self.cords)
        # dset_scale_names = group.create_dataset('scale names', shape=(self.num_scales,),
        #                                         dtype=h5py.string_dtype(encoding='ascii'))
        # for idx, name in enumerate(self._get_scale_names()):
        #     dset_scale_names[idx] = name
        dset_displacements = group.create_dataset('displacements', data=self.disp_arr, dtype=np.float)
        dset_mean_images = group.create_dataset('mean images', dtype=np.float32,
                                                data=self.wing_model(self.disp_arr.mean(axis=0))[0])
        group.create_dataset('modal shapes', dtype=np.float64,
                             data=read_modal_shapes(self.modal_shape_mat_path, num_scales=self.num_scales))

        group.attrs['cameras'] = self.cameras
        group.attrs['mesh_wing_path'] = self.mesh_wing_path.name
        group.attrs['mesh_tip_path'] = self.mesh_tip_path.name
        group.attrs['resolution'] = self.resolution
        group.attrs['texture'] = self.texture_path_wing.name
        group.attrs['ir'] = self.ir_list

    @cached(max_size=1)
    def get_data_sizes(self) -> (int, int):
        """
        Returns:
            num_scales,image_shape,num_ir_points
        """
        resolution = tuple(self.resolution[::-1])
        image_shape = (len(self.cameras), *resolution, 4)
        return self.num_scales, image_shape, len(self.ir_list)

    def __iter__(self) -> (str, np.array, np.array, np.array):
        """
           A generator for synthetic data pairs
            yields (video_name, point displacement, scales
        """
        for point_disp, scales in zip(self.disp_arr, self.scales_arr):
            # TODO fix video naming
            image, ir = self.wing_model(point_disp)
            yield "temp name", image, ir, scales
