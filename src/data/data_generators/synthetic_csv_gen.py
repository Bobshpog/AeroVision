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
from geometry.numpy.mesh import read_off_size
from geometry.numpy.wing_models import FiniteElementWingModel


@dataclass(repr=False)
class SyntheticCSVGenerator(data_gen.DataGenerator):
    raw_data_folder: Union[Path, str] = field(repr=False)
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

    @classmethod
    @cached(max_size=2)
    def _get_csv_len(cls, csv_path):
        with open(csv_path, 'r') as csv_file:
            return len(csv_file.readlines()) - 1

    @classmethod
    def _get_cords_from_csv(cls, points_path):
        point_num = cls._get_csv_len(points_path)
        with open(points_path, 'r') as point_file:
            point_file.readline()
            points_csv = csv.reader(point_file)
            cords_arr = np.zeros((point_num, 3), 'float')
            for idx, row in enumerate(points_csv):
                cords_arr[idx] = [float(x) for x in row[2:5]]  # indexes of relevant nodes
            return cords_arr

    @classmethod
    def _get_disps_from_csv(cls, points_path, point_num=None):
        if point_num is None:
            point_num = cls._get_csv_len(points_path)
        with open(points_path, 'r') as point_file:
            point_file.readline()
            points_csv = csv.reader(point_file)
            disps_arr = np.zeros((point_num, 3), 'float')
            for idx, row in enumerate(points_csv):
                disps_arr[idx] = [float(x) for x in row[5:8]]  # indexes of relevant nodes
            return disps_arr

    @classmethod
    def _process_csv_pair(cls, csv_pair):
        points_path, scales_path = csv_pair
        scales_csv_parameters_num = cls._get_csv_len(scales_path)
        point_num = cls._get_csv_len(points_path)
        with open(points_path, 'r') as point_file, open(scales_path, 'r') as scales_file:
            point_file.readline()
            scales_file.readline()
            points_csv = csv.reader(point_file)
            scales_csv = csv.reader(scales_file)
            disp_np_arr = np.zeros((point_num, 3), 'float')
            scales_np_arr = np.zeros(scales_csv_parameters_num, 'float')
            for idx, row in enumerate(points_csv):
                disp_np_arr[idx] = [float(x) for x in row[5:8]]  # indexes of relevant nodes
            scales_np_arr[:] = [row[1] for row in scales_csv]
            return disp_np_arr, scales_np_arr

    @cached(max_size=1)
    def _get_scale_names(self):
        _, scales_path = self.get_first_file_pair_path()
        with open(scales_path, 'r') as scale_file:
            scale_file.readline()
            scale_csv = csv.reader(scale_file)
            return [row[0].encode('ascii', 'ignore') for row in scale_csv]

    def __post_init__(self):
        if isinstance(self.raw_data_folder, str):
            self.raw_data_folder = Path(self.raw_data_folder)
        if isinstance(self.mesh_wing_path, str):
            self.mesh_wing_path = Path(self.mesh_wing_path)
        if isinstance(self.mesh_tip_path, str):
            self.mesh_tip_path = Path(self.mesh_tip_path)
        if isinstance(self.texture_path, str):
            self.texture_path = Path(self.texture_path)
        num_vertices_wing, _ = read_off_size(self.mesh_wing_path)
        num_vertices_tip, _ = read_off_size(self.mesh_tip_path)
        plotter = pv.Plotter(off_screen=True)
        self.cords = self._get_base_cords()
        self.wing_model = FiniteElementWingModel(self.cords, self.ir_list, self.texture_path, self.mesh_wing_path,
                                                 self.mesh_tip_path, self.cameras, num_vertices_wing,
                                                 num_vertices_tip, plotter, self.resolution, self.cmap)

    def __repr__(self):
        return f"{self.__class__.__name__}(mesh_wing='{self.mesh_wing_path.name}', mesh_tip='{self.mesh_tip_path.name}'" \
               f", resolution={self.resolution}, texture_path='{self.texture_path.name}'"

    def save_metadata(self, hdf5: h5py.File, group_name: str) -> None:
        group = hdf5.create_group(group_name)
        num_scales, _, _ = self.get_data_sizes()
        num_vertices_input=len(self._get_base_cords())
        group.create_dataset('cords', (num_vertices_input, 3), dtype=np.float, data=self.cords)
        dset_scale_names = group.create_dataset('scale names', shape=(num_scales,),
                                                dtype=h5py.string_dtype(encoding='ascii'))
        for idx, name in enumerate(self._get_scale_names()):
            dset_scale_names[idx] = name
        dset_displacements = group.create_dataset('displacements', (len(self), num_vertices_input, 3), dtype=np.float)
        for idx, files in enumerate(self._filename_iter()):
            point_file, _ = files
            dset_displacements[idx] = self._get_disps_from_csv(point_file, num_vertices_input)

        group.attrs['cameras'] = self.cameras
        group.attrs['mesh_wing_path'] = self.mesh_wing_path.name
        group.attrs['mesh_tip_path'] = self.mesh_tip_path.name
        group.attrs['resolution'] = self.resolution
        group.attrs['texture'] = self.texture_path.name

    @cached(max_size=1)
    def _get_base_cords(self):
        return self._get_cords_from_csv(self.get_first_file_pair_path()[0])

    def _filename_iter(self):
        file_list = sorted(next(os.walk(self.raw_data_folder))[2])
        # folder_list=sorted(next(os.walk(self.raw_data_folder))[1])
        # for folder in folder_list:
        #   file_list = sorted(next(os.walk(folder)[2] #TODO include video lists
        num_files = len(file_list) // 2
        for i in range(num_files):
            scale_file = file_list[i]
            point_file = file_list[num_files + i]
            yield self.raw_data_folder / point_file, self.raw_data_folder / scale_file

    def get_first_file_pair_path(self):
        return next(self._filename_iter())

    @cached(max_size=1)
    def get_data_sizes(self) -> (int, int):
        """
        Returns:
            (num_vertices_input, num_scales)
        """
        point_file, scale_file = self.get_first_file_pair_path()
        resolution = tuple(self.resolution[::-1])
        image_shape = (len(self.cameras), *resolution, 4)
        return self._get_csv_len(point_file), self._get_csv_len(scale_file), image_shape, len(self.ir_list)

    def __iter__(self) -> (str, np.array, np.array, np.array):
        """
           A generator for synthetic data pairs
            yields (video_name, point displacement, scales
        """
        file_list = sorted(next(os.walk(self.raw_data_folder))[2])
        # folder_list=sorted(next(os.walk(self.raw_data_folder))[1])
        # for folder in folder_list:
        #   file_list = sorted(next(os.walk(folder)[2] #TODO include video lists
        for i in self._filename_iter():
            # TODO fix video naming
            point_disp, scales = self._process_csv_pair(i)
            image, ir = self.wing_model(point_disp)
            yield "temp name", image, ir, scales
