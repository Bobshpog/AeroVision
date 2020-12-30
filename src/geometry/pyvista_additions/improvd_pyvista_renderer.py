import pyvista
from src.geometry.pyvista_additions.My_BackGround_Rendererer import OurBackgroundRenderer


class ImprovedPlotter(pyvista.Plotter):
    def add_background_photo(self, image, scale=1, auto_resize=True,
                             as_global=False):
        """Add a background image to a plot.

        Parameters
        ----------
        image_path : numpy
            numpy img

        scale : float, optional
            Scale the image larger or smaller relative to the size of
            the window.  For example, a scale size of 2 will make the
            largest dimension of the image twice as large as the
            largest dimension of the render window.  Defaults to 1.

        auto_resize : bool, optional
            Resize the background when the render window changes size.

        as_global : bool, optional
            When multiple render windows are present, setting
            ``as_global=False`` will cause the background to only
            appear in one window.

        Examples
        --------

        """
        # verify no render exists
        if self._background_renderers[self._active_renderer_index] is not None:
            raise RuntimeError('A background image already exists.  '
                               'Remove it with remove_background_image '
                               'before adding one')

        # Need to change the number of layers to support an additional
        # background layer
        self.ren_win.SetNumberOfLayers(3)
        if as_global:
            for renderer in self.renderers:
                renderer.SetLayer(2)
            view_port = None
        else:
            self.renderer.SetLayer(2)
            view_port = self.renderer.GetViewport()

        renderer = OurBackgroundRenderer(self, image, scale, view_port)
        renderer.SetLayer(1)
        self.ren_win.AddRenderer(renderer)
        self._background_renderers[self._active_renderer_index] = renderer

        # setup autoscaling of the image
        if auto_resize:  # pragma: no cover
            self._add_observer('ModifiedEvent', renderer.resize)


