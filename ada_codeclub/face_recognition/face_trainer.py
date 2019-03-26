import os
import sys

import cv2
import numpy as np
from PIL import Image

from ada_codeclub.face_recognition.utility.logger import Logger


class FaceTrainer(object):

    def __init__(self, dataset_path="dataset", training_path="trainer"):
        self.dataset_path = dataset_path
        self.training_path = training_path
        if not os.path.isdir(self.dataset_path):
            pass  # TODO: ABORT HERE DATASET INVALID

        if not os.path.isdir(self.training_path):
            os.makedirs(self.training_path)

        self.cascade_path = "ada_codeclub/cascades/haarcascade_frontalface_default.xml"
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(self.cascade_path)
        self.logger = Logger(sys.stderr)
        self.logger.add_writer(sys.stdout, Logger.ALL)

    def get_images_and_labels(self, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        # should return two lists of numpy_arrays of iamges and the corresponding face ids
        return face_samples, ids

    def run_trainer(self):
        self.logger.info("Training faces. This will take awhile. Wait ...")
        faces, ids = self.get_images_and_labels(self.dataset_path)
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.write(self.training_path + '/trainer.yml')
        self.logger.info("\n {0} faces trained. Exiting Program".format(len(np.unique(ids))))


if __name__ == "__main__":
    trainer = FaceTrainer()
    trainer.run_trainer()
