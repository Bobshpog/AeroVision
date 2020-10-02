from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pyvista as pv

from src.data.data_generators.data_gen import DataGenerator
from src.geometry.numpy.mesh import Mesh
from src.geometry.numpy.transforms import fem_wing_sine_decaying_in_space, fem_tip_sine_decaying_in_space
from src.util.timing import profile


@dataclass(repr=False)
class SyntheticSineDecayingGen(DataGenerator):
    raw_data_folder: Union[Path, str] = field(repr=False)
    mesh_wing_path: Union[Path, str]
    mesh_tip_path: Union[Path, str]
    num_videos: int
    num_frames_per_video: int
    ir_list: list = field(repr=False)  # list of ids of points in mesh
    resolution: list  # [Width, Height]
    # cameras in pyvista format
    cameras: list = field(repr=False)
    texture_path: Union[str, Path]
    cmap: str = field(repr=False)
    mesh_wing_path: Union[Path, str]
    mesh_tip_path: Union[Path, str]

    def __post_init__(self):
        if isinstance(self.raw_data_folder, str):
            self.raw_data_folder = Path(self.raw_data_folder)
        if isinstance(self.mesh_wing_path, str):
            self.mesh_wing_path = Path(self.mesh_wing_path)
        if isinstance(self.mesh_tip_path, str):
            self.mesh_tip_path = Path(self.mesh_tip_path)
        if isinstance(self.texture_path, str):
            self.texture_path = Path(self.texture_path)
        self.wing_mesh = Mesh(self.mesh_wing_path)
        self.tip_mesh = Mesh(self.mesh_tip_path)
        self.wing_mesh.pv_mesh.texture_map_to_plane(inplace=True)
        self.plotter = pv.Plotter(off_screen=True)

    def __del__(self):
        self.plotter.close()

    def __len__(self):
        return self.num_videos * self.num_frames_per_video

    def get_data_sizes(self) -> (int, int):
        resolution = tuple(self.resolution[::-1])
        image_shape = (len(self.cameras), *resolution, 4)
        return 3, image_shape, len(self.ir_list)

    def save_metadata(self, hdf5: h5py.File, group_name: str) -> None:
        group = hdf5.create_group(group_name)
        num_scales, _, _ = self.get_data_sizes()
        dset_scale_names = group.create_dataset('scale names', shape=(num_scales,),
                                                dtype=h5py.string_dtype(encoding='ascii'))
        for idx, name in enumerate(['amp', 'freq_s', 'decay']):
            dset_scale_names[idx] = name.encode('ascii', 'ignore')
        dset_mean_images = group.create_dataset('mean images', dtype=np.float32,
                                                data=Mesh.get_many_photos([self.wing_mesh, self.tip_mesh],
                                                                          [self.wing_mesh.vertices,
                                                                           self.tip_mesh.vertices], self.resolution,
                                                                          [self.texture_path, None], self.cmap,
                                                                          self.plotter,
                                                                          self.cameras))
        group.attrs['cameras'] = self.cameras
        group.attrs['mesh_wing_path'] = self.mesh_wing_path.name
        group.attrs['mesh_tip_path'] = self.mesh_tip_path.name
        group.attrs['resolution'] = self.resolution
        group.attrs['texture'] = self.texture_path.name

    def __iter__(self):
        for i in range(self.num_videos):
            amp = np.random.uniform(0.01, 0.07)
            decay = np.random.uniform(3, 7)
            freq_s = np.random.uniform(0.5, 3)
            vid_name = f"{i} {amp} {decay} {freq_s}"
            for phase in np.linspace(0, 2 * np.pi, self.num_frames_per_video):
                wing_movement = np.apply_along_axis(fem_wing_sine_decaying_in_space, axis=1,
                                                    arr=self.wing_mesh.vertices,
                                                    freq_t=1, freq_s=freq_s, amp=amp, t=phase, decay_rate_s=decay)
                tip_movement = np.apply_along_axis(fem_tip_sine_decaying_in_space, axis=1, arr=self.tip_mesh.vertices,
                                                   freq_t=1, freq_s=freq_s, amp=amp, t=phase)
                image = Mesh.get_many_photos([self.wing_mesh, self.tip_mesh], [wing_movement, tip_movement],
                                             self.resolution, [self.texture_path, None], 'jet', self.plotter,
                                             self.cameras)
                yield vid_name, image, wing_movement[self.ir_list], np.array([amp, decay, freq_s])

    def __repr__(self):
        string = f"{self.__class__.__name__}(mesh_wing='{self.mesh_wing_path.stem}', mesh_tip='{self.mesh_tip_path.stem}'" \
                 f", resolution={self.resolution}, texture_path='{self.texture_path.name}'"
        return string
