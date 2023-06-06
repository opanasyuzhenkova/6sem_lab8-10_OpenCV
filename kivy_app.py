from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserListView
import argparse
import sys
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--image")

args = parser.parse_args()  # python3 kivy_app.py -- --image faces.jpg


def highlightFace(net, frame, conf_threshold=0.6):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, faceBoxes


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

faceProto = "./files/opencv_face_detector.pbtxt"
faceModel = "./files/opencv_face_detector_uint8.pb"
genderProto = "./files/gender_deploy.prototxt"
genderModel = "./files/gender_net.caffemodel"
ageProto = "./files/age_deploy.prototxt"
ageModel = "./files/age_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)


def detect_age_gender(blob):
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    print(f'Gender: {gender}')

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    print(f'Age: {age[1:-1]} years')
    return [age, gender]


class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        # ret, frame = self.capture.read()
        frame = cv2.flip(frame, 1)
        if ret:
            resultImg, faceBoxes = highlightFace(faceNet, frame)
            if faceBoxes:
                for faceBox in faceBoxes:
                    face = frame[max(0, faceBox[1]):
                                 min(faceBox[3], frame.shape[0] - 1), max(0, faceBox[0])
                                                                      :min(faceBox[2], frame.shape[1] - 1)]
                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    age, gender = detect_age_gender(blob)
                    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 255), 2, cv2.LINE_AA)
            # convert it to texture
            buf1 = cv2.flip(resultImg, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(size=(resultImg.shape[1], resultImg.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


class CamApp(App):
    title = "Detecting age and gender"

    def build(self):
        self.capture = cv2.VideoCapture(args.image if args.image else 0)

        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.my_camera)
        button = Button(text='Выбрать файл', size_hint=(1, None), height=50)
        button.bind(on_release=self.open_file_dialog)
        layout.add_widget(button)
        return layout

    def open_file_dialog(self, instance):
        file_chooser = FileChooserListView()
        file_chooser.bind(on_submit=self.load_image)
        self.root.add_widget(file_chooser)

    def load_image(self, file_chooser, file_path, *args):
        self.root.remove_widget(file_chooser)
        print(file_path)
        image = cv2.imread(file_path[0])
        Clock.unschedule(self.my_camera.update)
        self.capture.release()
        resultImg, faceBoxes = highlightFace(faceNet, image)
        if faceBoxes:
            for faceBox in faceBoxes:
                face = image[max(0, faceBox[1]):
                             min(faceBox[3], image.shape[0] - 1), max(0, faceBox[0])
                                                                  :min(faceBox[2], image.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                age, gender = detect_age_gender(blob)
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2, cv2.LINE_AA)

        self.my_camera.texture = self.create_texture(resultImg)

    def create_texture(self, image):
        # convert image to texture
        buf1 = cv2.flip(image, 0)
        buf = buf1.tobytes()
        texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture

    def on_stop(self):
        # without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == '__main__':
    CamApp().run()
