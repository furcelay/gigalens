from abc import ABC, abstractmethod


class PointSolverBase(ABC):

    def __init__(self, phys_model, positions_structure):
        self.phys_model = phys_model
        self.positions_structure = positions_structure
        self.pack_bij = None

    @abstractmethod
    def points_beta_barycentre(self, x, y, params):
        pass

    @abstractmethod
    def points_magnification(self, x, y, params):
        pass
