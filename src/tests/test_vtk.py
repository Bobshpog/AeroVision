from unittest import TestCase
import glob
import os
import unittest
import vtk
from src.geometry.animations.normal_wing_animations import *
from src.geometry.spod import *


class Test(TestCase):

    def test_vtk_shananigans(self):
        sphere = vtk.vtkPointSource()
        sphere.SetNumberOfPoints(25)
        mesh = Mesh("data/wing_off_files/synth_wing_v3.off")
        # Triangulate the points with vtkDelaunay3D. This generates a convex hull
        # of tetrahedron.
        delny = vtk.vtkDelaunay3D()
        delny.SetInputConnection(sphere.GetOutputPort())
        delny.SetTolerance(0.01)
        print(dir(mesh.pv_mesh))
        # The triangulation has texture coordinates generated so we can map
        # a texture onto it.
        tmapper = vtk.vtkTextureMapToCylinder()
        tmapper.SetInputConnection(mesh.pv_mesh.GetPointData().GetOutputPort())
        tmapper.PreventSeamOn()

        # We scale the texture coordinate to get some repeat patterns.
        xform = vtk.vtkTransformTextureCoords()
        xform.SetInputConnection(tmapper.GetOutputPort())
        xform.SetScale(4, 4, 1)

        # vtkDataSetMapper internally uses a vtkGeometryFilter to extract the
        # surface from the triangulation. The output (which is vtkPolyData) is
        # then passed to an internal vtkPolyDataMapper which does the
        # rendering.
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(xform.GetOutputPort())

        # A texture is loaded using an image reader. Textures are simply images.
        # The texture is eventually associated with an actor.
        bmpReader = vtk.vtkPNGReader()
        bmpReader.SetFileName("data/textures/checkers.png")
        atext = vtk.vtkTexture()
        atext.SetInputConnection(bmpReader.GetOutputPort())
        atext.InterpolateOn()
        triangulation = vtk.vtkActor()
        triangulation.SetMapper(mapper)
        triangulation.SetTexture(atext)

        # Create the standard rendering stuff.
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        # Add the actors to the renderer, set the background and size
        ren.AddActor(triangulation)
        ren.SetBackground(1, 1, 1)
        renWin.SetSize(300, 300)

        iren.Initialize()
        renWin.Render()
        iren.Start()


    def test_vis(self):
        plotter = pv.Plotter()
        plotter.set_background("black")
        mesh = Mesh("data/wing_off_files/synth_wing_v3.off")
        mesh.plot_faces(plotter=plotter,texture='data/textures/rainbow.png', camera=camera_pos["up_middle"])

    def test_xyz(self):
        mesh = Mesh("data/wing_off_files/synth_wing_v3.off")
        tip = Mesh("data/wing_off_files/fem_tip.off")
        plotter = pv.Plotter()
        plotter.set_background("white")
        tip.plot_faces(show=False, plotter=plotter)
        mesh.plot_faces(show= False, plotter=plotter, texture="data/textures/circles_normal.png",
                        camera=camera_pos["up_middle"])
        vmtx = plotter.camera.GetModelViewTransformMatrix()
        #print(type(vmtx))
        mtx = pv.trans_from_matrix(vmtx)
        dehom = dehomogenize(mtx)

        print(dehom[0:3,:].shape)
        ainv = np.linalg.inv(dehom[0:3,:])
        hom = homogenize(ainv)
        print(hom.shape)
        A = np.array([
            [0.5, 0, 0, 0],
            [0, 0.5, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0, 0, 0.1]
        ])
        #print(mtx.shape)
        #print(dir(plotter.camera))
        plotter.camera.SetUseExplicitProjectionTransformMatrix(True)
        plotter.camera.SetExplicitProjectionTransformMatrix(trans_to_matrix(A))
        plotter.show()

        #synth_ir_video("src/tests/temp/ir_synth.mp4")




def trans_to_matrix(trans):
    """ Convert a numpy.ndarray to a vtk.vtkMatrix4x4 """
    matrix = vtk.vtkMatrix4x4()
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            matrix.SetElement(i, j, trans[i, j])
    return matrix

def dehomogenize(a):
    """ makes homogeneous vectors inhomogenious by dividing by the last element in the last axis
    >>> dehomogenize([1, 2, 4, 2]).tolist()
    [0.5, 1.0, 2.0]
    >>> dehomogenize([[1, 2], [4, 4]]).tolist()
    [[0.5], [1.0]]
    """
    a = np.asfarray(a)
    return a[..., :-1] / a[..., np.newaxis, -1]


def homogenize(v, value=1):
    v = np.asanyarray(v)
    if hasattr(value, '__len__'):
        return np.append(v, np.asanyarray(value).reshape(v.shape[:-1] + (1,)), axis=-1)
    else:
        return np.insert(v, v.shape[-1], np.array(value, v.dtype), axis=-1)