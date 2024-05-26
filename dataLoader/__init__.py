from .llff import LLFFDataset
from .blender import BlenderDataset
from .repair import RepairDataset
from .co3d import CO3DDataset
from .co3d_metashape import CO3DMetashapeDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .mip360 import Mip360Dataset


dataset_dict = {
    'blender': BlenderDataset,
    'repair': RepairDataset,
    'co3d': CO3DDataset,
    'co3d_metashape': CO3DMetashapeDataset,
    'llff': LLFFDataset,
    'tankstemple': TanksTempleDataset,
    'mip360': Mip360Dataset,
    'nsvf': NSVF,
    'own_data': YourOwnDataset
}