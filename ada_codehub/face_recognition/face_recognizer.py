import cv2
import sys

from ada_codehub.face_recognition.names import names
from ada_codehub.face_recognition.utility.logger import Logger

class FaceRecognizer(object):

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainer/trainer.yml')
        self.cascadePath = "ada_codehub/cascades/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.id = 0
        self.confidence = -1
        
        # Initialize and start realtime video capture
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)  # set video width
        self.cam.set(4, 480)  # set video height

        # Define min window size to be recognized as a face
        self.minW = 0.1 * self.cam.get(3)
        self.minH = 0.1 * self.cam.get(4)

        self.logger = Logger(sys.stderr)
        self.logger.add_writer(sys.stdout, Logger.ALL)

    def recognize_from_file(self, img_path):

        def recognize_and_show():
            faces = self.faceCascade.detectMultiScale(img, 1.3, 5)
            
            #maybe generalize this, since it's almost identical to the live version
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                roi_gray = img[y:y + h, x:x + w]
                self.id, self.confidence = self.recognizer.predict(roi_gray)
                conf = int(100 - self.confidence)
                who = (names[self.id], names[0])[self.confidence > 80]
                cv2.putText(img, who + " " + str(conf), (x+10, y+h-10), self.font, 1, (255, 255, 255), 1)

            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        img = cv2.imread(img_path, 0)
        if img.all():
            recognize_and_show()
        else:
            self.logger.error("Could not open file or it does not exist")

    def run_recognizer(self):
        while True:
            ret, img = self.cam.read()
            # img = cv2.flip(img, -1)  # Flip vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                      minSize=(int(self.minW), int(self.minH)), )

            # should use the faces recognized in faces to draw rectangles and IDs on the image returned to screen.
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                self.id, self.confidence = self.recognizer.predict(roi_gray)
                conf = int(100 - self.confidence)
                who = (names[self.id], names[0])[self.confidence > 80]
                cv2.putText(img, who + " " + str(conf), (x+10, y+h+40), self.font, 1, (255, 255, 255), 1)
            
            cv2.imshow("video", img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                self.cleanup()
                break

    def cleanup(self):
        self.cam.release()
        cv2.destroyAllWindows()
