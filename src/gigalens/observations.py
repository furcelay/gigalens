from dataclasses import dataclass
import numpy as np
from typing import Optional, List


@dataclass
class ExtendedObs:
    image: np.array
    background_rms: Optional[float]
    exp_time: Optional[float]
    error_map: Optional[np.array]


@dataclass
class PointObs:
    x_positions: List[List[float]]
    y_positions: List[List[float]]
    errors: List[List[float]]

    @property
    def positions_structure(self):
        return [len(pos) for pos in self.x_positions]
