import ada_codeclub
from ada_codeclub.face_recognition.dataset_creator import DatasetCreator
from ada_codeclub.face_recognition.face_trainer import FaceTrainer
from ada_codeclub.face_recognition.face_recognizer import FaceRecognizer

if __name__ == '__main__':
    application = DatasetCreator("dataset")
    application.initialize_recording_session()
    application.run_image_recording_session()
    application.cleanup()

    application = FaceTrainer()
    application.run_trainer()

    application = FaceRecognizer()
    application.run_recognizer()
