import cv2


class FaceRecognizer(object):

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainer/trainer.yml')
        self.cascadePath = "ada_codeclub/cascades/haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.id = 0
        self.confidence = -1
        self.names = ['None', 'Peter', 'Bjørn', 'Bård']  # Names of people you want to identify (TRAINED FROM DATASET)

        # Initialize and start realtime video capture
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 640)  # set video widht
        self.cam.set(4, 480)  # set video height

        # Define min window size to be recognized as a face
        self.minW = 0.1 * self.cam.get(3)
        self.minH = 0.1 * self.cam.get(4)

    def run_recognizer(self):
        while True:
            ret, img = self.cam.read()
            # img = cv2.flip(img, -1)  # Flip vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                      minSize=(int(self.minW), int(self.minH)), )

            # should use the faces recognized in faces to draw rectangles and IDs on the image returned to screen.

            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                self.cleanup()
                break

    def cleanup(self):
        self.cam.release()
        cv2.destroyAllWindows()
