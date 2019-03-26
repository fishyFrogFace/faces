import os
import sys

import cv2

try:
    from .utility.logger import Logger
except:
    # Script is not running through __main__
    from ada_codeclub.face_recognition.utility.logger import Logger


class DatasetCreator(object):
    WINDOW_WIDTH = 640
    WINDOW_HEIGHT = 480
    PICTURES_PER_SAMPLE = 30

    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        if not os.path.isdir(self.dataset_path):
            os.makedirs(self.dataset_path)
        self.camera = self.initialize_camera()
        self.cascade_path = "ada_codeclub/cascades/haarcascade_frontalface_default.xml"
        self.face_detector = self.inialize_classifiers()
        self.logger = Logger(sys.stderr)
        self.logger.add_writer(sys.stdout, Logger.ALL)
        self.face_id = -1

    def initialize_camera(self):
        camera = cv2.VideoCapture(0)
        camera.set(3, self.WINDOW_WIDTH)
        camera.set(4, self.WINDOW_HEIGHT)
        return camera

    def inialize_classifiers(self):  # TODO: Return multiple detectors
        face_detector = cv2.CascadeClassifier(self.cascade_path)
        return face_detector

    def initialize_recording_session(self):
        print("Press ESC to exit this program.")
        self.face_id = input('\n enter user id end press <return> ==>  ')
        self.logger.info("\n Initializing face capture. Look the camera and wait ...")

    def run_image_recording_session(self):
        count = 0
        while True:
            ret, img = self.camera.read()
            # img = cv2.flip(img, -1)  # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
            #use faces to extract data from img, and store that in a location to be used for training later..
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cv2.imshow("image", roi_gray)
                cv2.imwrite("dataset/" + str(self.face_id) + "/" + str(count) + ".png", roi_gray)
                count += 1

            keycode = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if keycode == 27:
                self.cleanup()
                break
            elif count >= self.PICTURES_PER_SAMPLE:
                break

    def cleanup(self):
        self.logger.info("\n Exiting Program and cleanup stuff")
        self.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pass  # Write 'test' here
