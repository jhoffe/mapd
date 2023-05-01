import torch.nn

from mapd import MAPDModule
import lightning as L
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

