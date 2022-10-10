import torch

LOG_DIR = None
ROOT_DIR = "logs"

writer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
