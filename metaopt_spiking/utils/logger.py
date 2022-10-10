"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging
import os
from importlib import reload

import torch
from torch.utils.tensorboard import SummaryWriter

from . import config


def setup_logging(name, dir=""):
    """
    Setup the logging device to log into a uniquely created directory
    """
    # Setup global logging directory
    config.LOG_DIR = os.path.join(config.ROOT_DIR, dir, name)

    # Create the logging folder if it does not exist already
    if not os.path.isdir(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)

    # Need to reload logging as otherwise the logger might be captured by another library
    reload(logging)

    # Setup global logger
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-5.5s %(asctime)s] %(message)s",
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(config.LOG_DIR, name + "_event.log")),
            logging.StreamHandler()
        ])

    # Setup tensorboard writer device
    config.writer = SummaryWriter(os.path.join(config.LOG_DIR, name + "_tensorboard"))

    # Log configured torch device
    if config.device.type == "cuda":
        logging.info("Running on GPU {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
    else:
        logging.info("Running on {}".format(config.device.type))
