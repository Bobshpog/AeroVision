import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pyvista as pv
from memoization import cached

import src.data.data_generators.data_gen as data_gen
from src.data.matlab_reader import read_data, read_modal_shapes
from src.geometry.numpy.mesh import read_off_size, Mesh
from src.geometry.numpy.wing_models import FiniteElementWingModel, SyntheticWingModel


@dataclass(repr=False)
class SyntheticMatGenerator(data_gen.DataGenerator):
    mat_path: Union[Path, str] = field(repr=False)
    modal_shape_mat_path: Union[Path, str] = field(repr=False)
    mesh_wing_path: Union[Path, str]
    mesh_tip_path: Union[Path, str]

    ir_list: list = field(repr=False)  # list of ids of points in mesh
    resolution: list  # [Width, Height]
    # cameras in pyvista format
    cameras: list = field(repr=False)
    texture_path: Union[str, Path]
    cmap: str = field(repr=False)
    mesh_wing_path: Union[Path, str]
    mesh_tip_path: Union[Path, str]

    def __post_init__(self):
        if isinstance(self.mat_path, str):
            self.mat_path = Path(self.mat_path)
        if isinstance(self.modal_shape_mat_path, str):
            self.modal_shape_mat_path = Path(self.modal_shape_mat_path)
        if isinstance(self.mesh_wing_path, str):
            self.mesh_wing_path = Path(self.mesh_wing_path)
        if isinstance(self.mesh_tip_path, str):
            self.mesh_tip_path = Path(self.mesh_tip_path)
        if isinstance(self.texture_path, str):
            self.texture_path = Path(self.texture_path)
        self.num_vertices_wing, _ = read_off_size(self.mesh_wing_path)
        self.num_vertices_tip, _ = read_off_size(self.mesh_tip_path)
        plotter = pv.Plotter(off_screen=True)
        self.cords, self.disp_arr, self.scales_arr = read_data(str(self.mat_path))
        self.num_frames, self.num_scales = self.scales_arr.shape
        self.wing_model = SyntheticWingModel(self.cords, self.ir_list, self.texture_path, self.mesh_wing_path,
                                             self.mesh_tip_path, self.cameras, self.num_vertices_wing,
                                             self.num_vertices_tip, plotter, self.resolution, self.cmap)

    def __len__(self):
        return self.num_frames

    def __repr__(self):
        return f"{self.__class__.__name__}(mesh_wing='{self.mesh_wing_path.name}', mesh_tip='{self.mesh_tip_path.name}'" \
               f", resolution={self.resolution}, texture_path='{self.texture_path.name}'"

    def save_metadata(self, hdf5: h5py.File, group_name: str) -> None:
        group = hdf5.create_group(group_name)
        group.create_dataset('cords', dtype=np.float, data=self.cords)
        # dset_scale_names = group.create_dataset('scale names', shape=(self.num_scales,),
        #                                         dtype=h5py.string_dtype(encoding='ascii'))
        # for idx, name in enumerate(self._get_scale_names()):
        #     dset_scale_names[idx] = name
        dset_displacements = group.create_dataset('displacements', data=self.disp_arr, dtype=np.float)
        dset_mean_images = group.create_dataset('mean images', dtype=np.float32,
                                                data=self.wing_model(np.zeros(self.cords.shape, dtype=np.float32))[0])
        group.create_dataset('modal shapes', dtype=np.float64,
                             data=read_modal_shapes(self.modal_shape_mat_path, num_scales=self.num_scales))

        group.attrs['cameras'] = self.cameras
        group.attrs['mesh_wing_path'] = self.mesh_wing_path.name
        group.attrs['mesh_tip_path'] = self.mesh_tip_path.name
        group.attrs['resolution'] = self.resolution
        group.attrs['texture'] = self.texture_path.name
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
