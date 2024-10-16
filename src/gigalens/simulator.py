from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from lenstronomy.Data.pixel_grid import PixelGrid

import gigalens.model


@dataclass
class SimulatorConfig:
    """Holds parameters for simulation.

    Attributes:
        delta_pix (float): The pixel scale (i.e., the angular resolution between adjacent pixels)
        num_pix (int): The width of the simulated image in pixels.
        supersample (int): Supersampling factor
        kernel (:obj:`numpy.array`, optional): The point spread function with which to convolve simulated images
        transform_pix2angle (:obj:`numpy.array`, optional): An array mapping indices on the coordinate grid to
            angular units (RA and DEC)
    """

    delta_pix: float
    num_pix: int
    supersample: Optional[int] = 1
    kernel: Optional[Any] = None
    transform_pix2angle: Optional[np.array] = None
    pix_region: Optional[np.array] = None


class LensWCS:
    def __init__(self, n, supersample=1, transform_pix2angle=None, pix_scale=1.):

        if transform_pix2angle is None:
            transform_pix2angle = np.eye(2) * pix_scale
        self.transform_pix2angle = transform_pix2angle / supersample
        self.transform_angle2pix = np.linalg.inv(transform_pix2angle)

        if isinstance(n, int):
            self.n_x, self.n_y = n, n
        else:
            self.n_x, self.n_y = n

        self.supersample = supersample

        low_x = -(self.n_x * self.supersample - 1) / 2
        low_y = -(self.n_y * self.supersample - 1) / 2

        self.radec_at_xy_0 = np.squeeze((self.transform_pix2angle @ ([[low_x], [low_y]])))

    def pix2angle(self, x, y):
        radec = np.einsum('ij,i...->...j', self.transform_pix2angle, np.concatenate([[x], [y]])) + self.radec_at_xy_0
        radec = np.swapaxes(radec, -1, 0).astype(np.float32)
        return radec[0].T, radec[1].T

    def angle2pix(self, ra, dec):
        radec = np.swapaxes(np.concatenate([[ra], [dec]]), -1, 0) - self.radec_at_xy_0
        return np.einsum('ij,...i->j...', self.transform_angle2pix, radec).astype(np.float32)

    def pixel_grid(self):
        x, y = np.arange(self.n_y * self.supersample), np.arange(self.n_y * self.supersample)
        X, Y = np.meshgrid(x, y)
        return self.pix2angle(X, Y)


class LensSimulatorInterface(ABC):
    """
    A class to simulate batches of lenses given a physical model and camera configuration options (i.e., pixel scale,
    number of pixels, point spread function, etc.).

    Attributes:
        phys_model (:obj:`~gigalens.model.PhysicalModel`): The physical model that generated the lensing system. All
            parameters ``z`` that are passed to the simulation methods are expected to correspond to this physical
            model.
        sim_config (:obj:`~gigalens.simulator.SimulatorConfig`): Camera configuration settings
        bs (int): The number of lenses to simulate in parallel
    """

    def __init__(
        self,
        phys_model: gigalens.model.PhysicalModelBase,
        sim_config: SimulatorConfig,
        bs: int,
    ):
        self.phys_model = phys_model
        self.sim_config = sim_config
        self.bs = bs
        self.wcs = LensWCS(n=sim_config.num_pix, supersample=sim_config.supersample,
                           transform_pix2angle=sim_config.transform_pix2angle, pix_scale=sim_config.delta_pix)

    @abstractmethod
    def simulate(
        self,
        params: Dict[str, List[Dict]],
    ):
        """Simulates lenses with physical parameters ``params``.

        Args:
            params (:obj:`tuple` of :obj:`list` of :obj:`dict`): Parameters of the lensing system to simulate, with
                format (``lens_params``, ``lens_light_params``, ``source_light_params``)

        Returns:
            Simulated lenses
        """
        pass

    @abstractmethod
    def lstsq_simulate(
        self,
        params: Dict[str, List[Dict]],
        observed_image,
        err_map,
    ):
        """Simulates lenses and fits for their linear parameters based on the observed data.

        Args:
            params (:obj:`tuple` of :obj:`list` of :obj:`dict`): Parameters of the lensing system to simulate, with
                format (``lens_params``, ``lens_light_params``, ``source_light_params``)
            observed_image: The observed image, needed for solving the linear parameters
            err_map: Noise variance map, needed for solving the linear parameters

        Returns:
            Simulated images that have the linear parameters set to the optimal values, minimizing the chi-squared
            of the residual using ``err_map`` as the variance.
        """
        pass

    @staticmethod
    def get_coords(
        supersample: int, num_pix: int, transform_pix2angle: np.array
    ) -> Tuple[float, float, np.array, np.array]:
        """
        Calculates the coordinate grid for the given simulator settings.  The default coordinate system in this package
        is to ensure the mean coordinate is RA, DEC = 0,0.

        Args:
            supersample (int): Supersampling factor by which to increase the resolution of the coordinate grid
            num_pix (int): Number of pixels in the *downsampled* image (i.e., with `supersample=1`)
            transform_pix2angle (:obj:`numpy.array`): Array specifying the translation from indices to RA and DEC

        Returns:
            A tuple containing the RA and DEC at the (0,0) index of the coordinate grid (i.e., the bottom left corner),
            and the coordinate grids themselves.
        """
        lo = np.arange(0, supersample * num_pix, dtype=np.float32)
        lo = np.min(lo - np.mean(lo))

        ra_at_xy_0, dec_at_xy_0 = np.squeeze((transform_pix2angle @ ([[lo], [lo]])))
        kwargs_pixel_rot = {
            "nx": supersample * num_pix,
            "ny": supersample * num_pix,  # number of pixels per axis
            "ra_at_xy_0": ra_at_xy_0,  # RA at pixel (0,0)
            "dec_at_xy_0": dec_at_xy_0,  # DEC at pixel (0,0)
            "transform_pix2angle": np.array(transform_pix2angle),
        }
        pixel_grid_rot = PixelGrid(**kwargs_pixel_rot)

        img_x, img_y = (
            pixel_grid_rot._x_grid.astype(np.float32),
            pixel_grid_rot._y_grid.astype(np.float32),
        )
        return ra_at_xy_0, dec_at_xy_0, img_x, img_y
