#!/usr/bin/env python3

import time

import cv2
import pyretinanet as retina


class Detector:
    def __init__(self, weights, threshold):
        self.model = retina.RetinaNet(weights, threshold)

    def detect(self, img):
        detections = self.model.detect(img) 
        it = iter(detections)

        # return object
        # list of objects (label, score, bbox(x1,y1,x2,y2), ...,)
        return list(zip(it, it, zip(it, it, it, it)))


def main():
    weights_path = '/home/nvidia/src/weights/retinanet_rn50person.plan'
    threshold = 0.7

    net = Detector(weights=weights_path, threshold=threshold)
    img = cv2.imread('/home/nvidia/image.jpg')

    # loop detect
    for i in range(10):
        print(f'{i} detections')
        st = time.time()
        dets = net.detect(img)
        fn = time.time() - st

        print(f'inference time: {fn}')

        for label, score, box in dets:
            print(f'label: {label} score: {score} box: {box}')


if __name__ == '__main__':
        main()
