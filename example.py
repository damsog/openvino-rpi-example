import cv2
import sys
from openvino.inference_engine import IEPlugin, IENetwork, IECore
import logging as log
import numpy as np

import cv2 as cv
# Load the model.
net = IENetwork.from_ir(model='/home/pi/lpr/vehicle-license-plate-detection-barrier-0106.xml',weights='/home/pi/lpr/vehicle-license-plate-detection-barrier-0106.bin')
