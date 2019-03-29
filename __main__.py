import ada_codeclub
from ada_codeclub.face_recognition.dataset_creator import DatasetCreator
from ada_codeclub.face_recognition.face_trainer import FaceTrainer
from ada_codeclub.face_recognition.face_recognizer import FaceRecognizer

def display_help_menu():
    help = ("help - displays this menu of available options\n"
          + "quit - exits the program\n"
          + "new - record a new face\n"
          + "train - use previously recorded faces to train the recognizer"
    )
    print(help)
    start()

def input_to_action(choice):
    switcher = {
        "help": display_help_menu,
        "quit": exit,
        "new": new_face,
        "train": train_from_pics
    }
    return switcher.get(choice, not_valid_choice)

def new_face():
    application = DatasetCreator("dataset")
    application.initialize_recording_session()
    application.run_image_recording_session()
    application.cleanup()
    start()

#crashes on empty database
def train_from_pics():
    application = FaceTrainer()
    application.run_trainer()
    start()

def not_valid_choice():
    print("Your choice was not valid\n")
    start()

def start():
        userInput = input("What do you want to do? Type \"help\" for a menu\n")
        input_to_action(userInput.lower())()

        #code currently never reached
        application = FaceRecognizer()
        application.run_recognizer()

if __name__ == '__main__':
    start()