import cv2 as cv
# Load the model.

#ssd tf frozen model
#net = cv.dnn_DetectionModel('/home/pi/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet/frozen_inference_graph.xml',
#                            '/home/pi/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet/frozen_inference_graph.bin')

#openvino face_det
#net = cv.dnn_DetectionModel('/home/pi/ssd_mobilenet_v1_coco_2018_01_28/build/face-detection-adas-0001.xml',
#                            '/home/pi/ssd_mobilenet_v1_coco_2018_01_28/build/face-detection-adas-0001.bin')

#openvino face_det retail
#net = cv.dnn_DetectionModel('/home/pi/ssd_mobilenet_v1_coco_2018_01_28/build/face-detection-retail-0004.xml',
#                            '/home/pi/ssd_mobilenet_v1_coco_2018_01_28/build/face-detection-retail-0004.bin')

#openvino people det retail
net = cv.dnn_DetectionModel('/home/pi/people_counting/person-detection-retail-0013.xml',
                            '/home/pi/people_counting/person-detection-retail-0013.bin')

#net = cv.dnn_DetectionModel('/home/pi/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_deprecated/frozen_inference_graph.xml',
#                            '/home/pi/ssd_mobilenet_v1_coco_2018_01_28/ssd_mobilenet_deprecated/frozen_inference_graph.bin')

# tf ssd custom
#net = cv.dnn_DetectionModel('/home/pi/ssd_mobilenet_v1_coco_2018_01_28/tfssd_custom/frozen_inference_graph.xml',
#                            '/home/pi/ssd_mobilenet_v1_coco_2018_01_28/tfssd_custom/frozen_inference_graph.bin')

# Specify target device.
#net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

video_test = '/home/pi/ssd_mobilenet_v1_coco_2018_01_28/test4.mp4'

# Read an image.
frame = cv.imread('/home/pi/ssd_mobilenet_v1_coco_2018_01_28/build/test.jpg')

#video = cv.VideoCapture("/home/pi/ssd_mobilenet_v1_coco_2018_01_28/dashcam2.mp4")
video = cv.VideoCapture(0)

while video.isOpened():
    ok, frame = video.read()
    img = frame.copy()
    scale_percent = 100# percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    frame = cv.resize(img, dim, interpolation = cv.INTER_AREA) 
    
    if frame is None:
        raise Exception('Image not found!')
    # Perform an inference.
    _, confidences, boxes = net.detect(frame, confThreshold=0.5)
    # Draw detected faces on the frame.
    for confidence, box in zip(list(confidences), boxes):
        cv.rectangle(frame, box, color=(0, 255, 0))
    # Save the frame to an image file.
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break

