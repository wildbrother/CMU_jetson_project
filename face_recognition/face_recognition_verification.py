import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from matplotlib import pyplot
import mmcv, cv2
import PIL.Image
from PIL import Image, ImageDraw, ImageFont
from IPython import display

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import pandas as pd
from torchvision.utils import save_image


filepath = '/home/phj/cmu-studio-project-team5/face_recognition/test_video/'
video_name = 'ShortTC-TG.mp4'

# workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.9, post_process=True,
    device=device, keep_all=True
)

model = InceptionResnetV1(pretrained='vggface2').eval()
print('Loaded Model')

# print(mtcnn.state_dict())
# torch.save(mtcnn.state_dict(), './mtcnn.pt')
# torch.save(model.state_dict(), './inception_resnet_v1.pt')

### For loading image file
def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array
    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image contents as numpy array
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)

# Load a sample picture and learn how to recognize it.


tom_image = mtcnn(load_image_file("./image/tom.jpg"))
tom_face_encoding = model(tom_image[0].unsqueeze(0)).detach().numpy()

# Load a second sample picture and learn how to recognize it.
anthony_image = mtcnn(load_image_file("./image/anthony.jpg"))
anthony_face_encoding = model(anthony_image[0].unsqueeze(0)).detach().numpy()

# Load a third sample picture and learn how to recognize it.
val_image = mtcnn(load_image_file("./image/val.jpg"))
val_face_encoding = model(val_image[0].unsqueeze(0)).detach().numpy()

# Load a fourth sample picture and learn how to recognize it.
skerritt_image = mtcnn(load_image_file("./image/skerritt.jpg"))
skerritt_face_encoding = model(skerritt_image[0].unsqueeze(0)).detach().numpy()

print("For checking input size")
print(load_image_file("./image/tom.jpg").shape)
print(tom_image[0].unsqueeze(0).shape)

# Create arrays of known face encodings and their names
known_face_encodings = [
    tom_face_encoding,
    anthony_face_encoding,
    val_face_encoding,
    skerritt_face_encoding
]

known_face_names = [
    "Tom Cruise",
    "Anthony Edwards",
    "Val Kilmer",
    "Tom Skerritt"
]

# Run video through MTCNN
# We iterate through each frame, detect faces, and draw their bounding boxes on the video frames.


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=2)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.
    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

video = mmcv.VideoReader(filepath + video_name)
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
print("the number of frames: ", len(frames))

display.Video(filepath + video_name, width=640)
# display.Video(filepath + video_name)


### Initialize some variables
frames_tracked = []
frames_names = []

font = ImageFont.load_default()
font_size = 20
font = ImageFont.truetype("./arial.ttf", font_size)

unknown_label = 0

for i, frame in enumerate(frames):
    # print(frame)
    print(frame)
    print('\rTracking frame: {}/{}'.format(i + 1, len(frames)), end='')
    boxes, _ = mtcnn.detect(frame) ## extracting the bounding box
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    
    ## crop_image
    imgs = mtcnn(frame)

    if boxes is None:
        pass
    else:
        for box, img in zip(boxes, imgs):
            # draw box
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)
            face_encoding = model(img.unsqueeze(0)).detach().numpy()

            # See if the face is a match for the known face(s)
            matches = compare_faces(known_face_encodings, face_encoding)
            face_distances = face_distance(known_face_encodings, face_encoding)
            # print(face_distances)
        
            print('matches:', matches)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            else:
                known_face_encodings.append(face_encoding)
                known_face_names.append(f"Unlabeled_person_{unknown_label}")


                face = frame.crop((box[0], box[1], box[2], box[3]))
                
                ## try to save image
                face.save(f"./unlabeled_image/Unlabeled_person_{unknown_label}.jpg", "JPEG")

                unknown_label += 1
                
            # Or instead, use the known face with the smallest distance to the new face
            # face_distances = face_distance(known_face_encodings, face_encoding)
            # print(face_distances)
            
            # best_match_index = np.argmin(face_distances)
            # name = known_face_names[best_match_index]

            text_posi = (box[0], box[1])
            draw.text(text_posi, name, font=font)

        # Add to frame list
    frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))

    
print('\nDone')
d = display.display(frames_tracked[0], display_id=True)
dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    
video_tracked = cv2.VideoWriter('/home/phj/cmu-studio-project-team5/face_recognition/outputs/test.mp4', fourcc, 25.0, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
# for name in frames_names:
video_tracked.release()

