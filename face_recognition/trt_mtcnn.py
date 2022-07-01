import os
import time
import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import json

import cv2
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.mtcnn import TrtMtcnn
from utils.inception import TrtModel

import torch
import torch.nn as nn
from inception import InceptionResnetV1
from torch.nn.functional import interpolate


BBOX_COLOR = (0, 255, 0)  # green

def load_image_file(file, mode='RGB'):
    im = Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)

def face_distance(face_encodings, face_to_compare, cosine=True):
    if len(face_encodings) == 0:
        return np.empty((0))
    if cosine:
        similarity_metric = nn.CosineSimilarity(dim=1, eps=1e-08)
        similarities = list()
        for i in face_encodings:
            similarity = similarity_metric(torch.Tensor(i),torch.Tensor(face_to_compare))
            similarities.append(similarity.item())
        return torch.Tensor(similarities).numpy().reshape(-1, 1)
    else:
        return np.linalg.norm(face_encodings - face_to_compare, axis=2)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=1, cosine=True):
    if cosine:
        return list(face_distance(known_face_encodings, face_encoding_to_check) >= tolerance)
    else:
        return list(face_distance(known_face_encodings, face_encoding_to_check, cosine=False) <= tolerance)


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face detection with TrtMtcnn on Jetson '
            'Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40,
                        help='minsize (in pixels) for detection [40]')
    args = parser.parse_args()
    return args


def show_faces(img, boxes, landmarks):
    """Draw bounding boxes and face landmarks on image."""
    if boxes is not None:
        for bb, ll in zip(boxes, landmarks):
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            face_cropped = img[y1:y2, x1:x2]
            # cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR, 2)
            # for j in range(5):
            #     cv2.circle(img, (int(ll[j]), int(ll[j+5])), 2, BBOX_COLOR, 2)
    return face_cropped, img

def make_json(model, video_name, frame_num, descriptions, scores):
    with open("result_face_3.json", "a") as f:
        index = {
            "index": {
                "_index": video_name,
                "_type": "frame"
                # "_id": id
            }
        }
        
        data = {
                "frame_number": frame_num,
                "label": [
                ]
            }
        des_list = list()
        sco_list = list()
        for (description, score) in zip(descriptions, scores):
            # labels = {
            #             "model": model,
            #             "description": description,
            #             "score": f"{score}"
            #          }
            # data["label"].append(labels)
            des_list.append(f"{description}")
            sco_list.append(f"{score}")

        # for (description, score) in zip(descriptions, scores):
        labels = {
                    "model": model,
                    "description": des_list,
                    "score": sco_list
                    }
        data["label"].append(labels)
        json.dump(index, f)
        f.write("\n")
        json.dump(data, f)
        f.write("\n")


def main():
    args = parse_args()
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # detection
    mtcnn = TrtMtcnn()
    video = "./videos/example.mp4"
    video_capture = cv2.VideoCapture(video)

    
    frame_width = int(video_capture.get(3))
    print("frame_width:", frame_width)
    frame_height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    out = cv2.VideoWriter('./outputs/result_tensorRT.mp4', fourcc, 15.0, (frame_width, frame_height))
    # cam.release()
    # cv2.destroyAllWindows()

    # classification
    model = TrtModel("inception.engine")

    known_face_names = set()
    face_list = list()
    person_dict = dict()
    for i in os.listdir("./images"):
        name = i.split("_")[0]
        known_face_names.add(name)
        face_list.append(name)

    for i in known_face_names:
        num = face_list.count(i)
        person_dict[i] = num
    known_face_names = list(known_face_names)

    # Make embeddings for known faces
    known_face_embeddings = list()

    for i in list(person_dict.keys()):
        embeddings = list()
        for j in range(person_dict[i]):
            try:      
                img = load_image_file(f"./images/{i}_{j}.jpeg")
            except:
                img = load_image_file(f"./images/{i}_{j}.jpg")
            print(f"This is {i} person {j}th image")
            dets, landmarks = mtcnn.detect(img)
            print("dets: ", dets)
            if not len(dets) == 0:
                face_cropped, img = show_faces(img, dets, landmarks)

                ####
                if face_cropped is not None:
                    frame = Image.fromarray(cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB))
                    frame.save("./frames/face.jpg", "JPEG")
                face = cv2.resize(face_cropped, (160, 160), interpolation=cv2.INTER_AREA).copy()
                face = np.transpose(face, (2, 0, 1))
                face = np.float32(np.expand_dims(face/255.0, axis=0))
                # import pdb
                # pdb.set_trace()
                # embed = nn.Embedding(num_embeddings=1, embedding_dim=512)
                embed = model(face, 1)[0]
                embeddings.append(embed)

        if not len(embeddings) == 0:
            embedding = np.nanmean(tuple(embeddings), axis=0)
            known_face_embeddings.append(embedding)

    fps = 0.0
    idx = 0
    # unknown_label = 0

    frame_num = 0
    while True:
        tic = time.time()
        print("frame_num: ", frame_num)
        img = cam.read()
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        _, frame = video_capture.read()
        if img is not None:
            dets, landmarks = mtcnn.detect(img)
            print('{} face(s) found'.format(len(dets)))
            # if len(dets) == 0:
            #     frame_num += 1
            #     toc = time.time()
            #     curr_fps = 1.0 / (toc - tic)
            #     # calculate an exponentially decaying average of fps number
            #     fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            #     print("fps: ", fps)
            #     idx += 1

            #     fps_mean.append(1/fps)

            #     continue
            names = list()
            distances = list()


            for det in dets:
                # print(det)
                face_cropped, img = show_faces(img, [det], landmarks)
                face = cv2.resize(face_cropped, (160, 160), interpolation=cv2.INTER_AREA).copy()
                face = np.transpose(face, (2, 0, 1))
                face = np.float32(np.expand_dims(face/255.0, axis=0))
                face_embedding = model(face, 1)[0]
                # print("face: ", face.shape())
                
                # face_embedding = nn.Embedding(num_embeddings=1, embedding_dim=512)
                # print(face_embedding.weight)
                matches = compare_faces(known_face_embeddings, face_embedding, tolerance=0.4)
                face_distances = face_distance(known_face_embeddings, face_embedding)
                print(' Matches: ', matches)
                # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:

                print(face_distances)
                max_index = np.argmax(face_distances)
                distance = np.max(face_distances)
                distances.append(distance)
                name = known_face_names[max_index]
                names.append(name)
                print(name)
                
                ## 1080 * 1920
                x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                cv2.rectangle(frame, (x2, y1), (x1, y2), (0, 0, 255), 2) #left top right bottom 
                # x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                face_cropped = img[y1:y2, x1:x2]
            
                # cv2.rectangle(frame, (y2, x2 - 35), (y1, x2), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x2 + 6, y2 - 6), font, 1.0, (255, 255, 255), 1)

            # else:
            #     known_face_embeddings.append(face_embedding)
            #     known_face_names.append(f"Unlabeled_person_{unknown_label}")

            #     unknown_face = Image.fromarray(cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB))
            #     unknown_face.save(f"./unlabeled_image/Unlabeled_person_{unknown_label}.jpg", "JPEG")

            #     unknown_label += 1
            
            # if face is not None:
            #     frame = Image.fromarray(cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB))
            #     frame.save(f"./frames/{idx}.jpg", "JPEG")
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            print("fps: ", fps)
            idx += 1

            # Example metadata
            video_name = 'vid_0'
            model_name = 'face'
            make_json(model_name, video_name, frame_num, names, distances)

            # cv2.imshow('Video', frame)
            frame_num += 1
            out.write(frame)
        
        else:
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
