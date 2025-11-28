try:
    from .model import CannonModel
except ImportError:
    CannonModel = None
from .dataset import Dataset
from . import diagnostics
