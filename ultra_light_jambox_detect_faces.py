import imutils
import cv2
import face_recognition
import copy
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare

from datetime import datetime
import pandas as pd
from jambox_grabber import Jambox
from db_manager import DBManager
from video_functions import frame_compare_faces
from ultra_light import frame_detect_ultra


# turn on display functions and other details
debug = False

# load known faces data
known_names = pd.read_pickle("dataset/president_faces_df.pickle")["name"].to_list()
known_encodings = pd.read_pickle("dataset/president_faces_df.pickle")[
    "face_encodings"
].to_list()

# load the model, create runtime session & get input variable name
onnx_path = "ultra_light_model/ultra_light_640.onnx"
onnx_model = onnx.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name


# video streams setup
jb = Jambox(1)

# database manager setup
db = DBManager()
db.setup_table()

# recognition grabber
last_recognitions = {}

# main loop
try:
    while True:

        if jb.is_ready():

            # grab frames
            jb.update_frames()
            frames = jb.frames

            # loop over the frames
            for source, frame in frames.items():

                # detect faces on frame
                frame, face_locations = frame_detect_ultra(
                    frame, ort_session, input_name, min_confidence=0.7
                )

                if len(face_locations) > 0:
                    # build face encodings
                    face_encodings = face_recognition.face_encodings(
                        frame, face_locations
                    )

                    # compare encodings with known faces
                    names = [
                        frame_compare_faces(encoding, known_encodings, known_names)
                        for encoding in face_encodings
                    ]

                    for location, name in zip(face_locations, names):
                        # annotate face boxes with names
                        if debug:
                            (top, right, bottom, left) = location
                            cv2.rectangle(
                                frame, (left, top), (right, bottom), (0, 0, 255), 2
                            )
                            y = top - 15 if top - 15 > 15 else top + 15
                            cv2.putText(
                                frame,
                                name,
                                (left, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (0, 255, 0),
                                2,
                            )

                    # dump to database
                    if name != "UNKNOWN":
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if debug:
                            print(f"{timestamp}, {name}, {source}")

                        # verify last seen timestamp
                        if last_recognitions.get(f"{source} {name}") != timestamp:
                            # db.insert_data(timestamp, name, source)
                            last_recognitions[f"{source} {name}"] = timestamp

            # show the output frames as montage
            if debug:
                montage_frames = list(frames.values())
                montage = imutils.build_montages(
                    image_list=montage_frames,
                    image_shape=(
                        montage_frames[0].shape[1],
                        montage_frames[0].shape[0],
                    ),
                    montage_shape=(1, len(montage_frames)),
                )[0]
                cv2.imshow("Captured_frames", montage)

                key = cv2.waitKey(1) & 0xFF

                # press q to quit
                if key == ord("q"):
                    break

                # press s to save montage
                if key == ord("s"):
                    cv2.imwrite("montage.jpg", montage)
except Exception as e:
    print(e)
finally:
    # cleanup
    cv2.destroyAllWindows()
    jb.close_browsers()
