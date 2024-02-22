from flask import Flask, render_template, Response,session
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import os
from ultralytics import YOLO

# Required to run  the YOLOv8 model
import cv2

#from video import video_detection
app = Flask(__name__)

app.config['SECRET_KEY'] = 'vaishnavi12'
app.config['UPLOAD_FOLDER'] = 'static/files'


class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")



def video_detection(path_x):
    video_capture = path_x
    cap=cv2.VideoCapture(video_capture)
    # text_speech = pyttsx3.init()


    model=YOLO("best.pt")

    classNames = ["Pothole"]


    while True:
        success, img = cap.read()
        # Doing detections using YOLOv8 frame by frame
        results=model(img,stream=True)

        for r in results:
            # boxes: ultralytics.engine.results.Boxes object
            boxes=r.boxes
            for box in boxes:
                #print(box)
                x1,y1,x2,y2=box.xyxy[0]
                # print(x1, y1, x2, y2)
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                print(box.conf[0])
                conf=math.ceil((box.conf[0]*100))/100
                print(conf)
                cls = int(box.cls[0])
                class_name = classNames[cls]
                # if class_name == 'Pothole':
                #     text_speech.say('Pothole Ahead')
                    # text_speech.runAndWait()
                    # text_speech.setProperty('rate', 100)
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]
                # print(t_size)
                # print('c2 :- ',x1+t_size[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), [255, 0, 255], 0, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        yield img

cv2.destroyAllWindows()


def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)
        if ref:
            frame=buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
        else:
            print("no valid input")

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection)
        if ref:
            frame=buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
        else:
            print("no valid input")

@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    session.clear()
    return render_template('indexproject.html')

@app.route("/webcam", methods=['GET','POST'])
def webcam():
    session.clear()
    return render_template('ui.html')

@app.route('/FrontPage', methods=['GET','POST'])
def front():
    # Upload File Form: Create an instance for the Upload File Form
    form = UploadFileForm()
    if form.validate_on_submit():
        # Our uploaded video file path is saved here
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        session['video_path'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                             secure_filename(file.filename))
    return render_template('videoprojectnew.html', form=form)
@app.route('/video')
def video1():
    return Response(generate_frames(path_x = session.get('video_path', None)),mimetype='multipart/x-mixed-replace; boundary=frame')

# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True,port=5000)
