import cv2
import sys
from openvino.inference_engine import IEPlugin, IENetwork, IECore
import logging as log
import numpy as np

import cv2 as cv
# Load the model.
net = IENetwork.from_ir(model='/home/pi/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.xml',weights='/home/pi/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.bin')
