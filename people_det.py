import cv2
import sys
from openvino.inference_engine import IEPlugin, IENetwork, IECore
import logging as log
import numpy as np

def main():
    #######################  Device  Initialization  ########################
    #  Plugin initialization for specified device and load extensions library if specified
    #plugin = IEPlugin(device="MYRIAD") 
    #########################################################################

    #########################  Load Neural Network  #########################
    #  Read in Graph file (IR)
    net = IENetwork(model="/home/pi/people_counting/person-detection-retail-0013.xml", weights="/home/pi/people_counting/person-detection-retail-0013.bin") 

    ie = IECore()
    versions = ie.get_versions("MYRIAD")

    #supported_layers = ie.query_network(net, "CPU")
    #not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    #if len(not_supported_layers) != 0:
    #    log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
    #                format(args.device, ', '.join(not_supported_layers)))
    #    log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
    #                "or --cpu_extension command line argument")
    #    sys.exit(1)

    #  Load network to the plugin
    ########################################################################
    log.info("Loading model to the device")
    exec_net = ie.load_network(network=net, device_name="MYRIAD")

    video = cv2.VideoCapture(0)

    while video.isOpened():
        ok, image = video.read()
        if not ok:
            log.info('Finished')
            break
        image_toshow = image.copy()


        #########################  Obtain Input Tensor  ########################
        #  Obtain and preprocess input tensor (image)
        #  Read and pre-process input image  maybe we don't need to show these details
        #image = cv2.imread("input_image.jpg")

        #  Preprocessing is neural network dependent maybe we don't show this
        input_blob = next(iter(net.inputs))
        n, c, h, w = net.inputs[input_blob].shape
        
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        image = image.reshape((n, c, h, w))
        ########################################################################

        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))
        ##########################  Start  Inference  ##########################
        #  Start synchronous inference and get inference result

        log.info("Creating infer request and starting inference")
        res = exec_net.infer(inputs={input_blob: image})

        #req_handle = exec_net.start_async(inputs={input_blob: image})
        ########################################################################

        ######################## Get Inference Result  #########################
        #status = req_handle.wait()
        #res = req_handle.outputs[out_blob]
        res = res[out_blob]
        data = res[0][0]
        ih, iw,_= image_toshow.shape

        # Do something with the results... (like print top 5)
        for number, proposal in enumerate(data):
            if proposal[2] > 0:
                imid = np.int(proposal[0])
                label = np.int(proposal[1])
                confidence = proposal[2]
                xmin = np.int(iw * proposal[3])
                ymin = np.int(ih * proposal[4])
                xmax = np.int(iw * proposal[5])
                ymax = np.int(ih * proposal[6])
                #print("[{},{}] element, prob = {:.6}    ({},{})-({},{}) batch id : {}"\
                #    .format(number, label, confidence, xmin, ymin, xmax, ymax, imid), end="")
                if proposal[2] > 0.5:
                    cv2.rectangle(image_toshow, (xmin, ymin), (xmax, ymax), (232, 35, 244), 2 )

        cv2.imshow('dets',image_toshow)
        cv2.waitKey(10)
        #cv2.destroyAllWindows()
        ###############################  Clean  Up  ############################
        #del exec_net
    ########################################################################
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    sys.exit(main() or 0)
