import os
import torch.nn as nn

import mmcv, cv2
import PIL.Image
from PIL import Image, ImageDraw, ImageFont

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import argparse

def load_image_file(file, mode='RGB'):
    im = PIL.Image.open(file)
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

def make_embedding(detection_model, verification_model, person_name, number_of_images=1):
    
    embeddings = list()
    for i in range(number_of_images):
        try:
            img = detection_model(load_image_file(f"./image/{person_name}_{i}.jpeg"))
        except:
            img = detection_model(load_image_file(f"./image/{person_name}_{i}.jpg"))
        embedding = verification_model(img[0].unsqueeze(0)).detach().numpy()
        embeddings.append(embedding)
    return np.mean(tuple(embeddings), axis=0)

def main(args):
    device = args.device
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.9, post_process=True,
        device=device, keep_all=True
    )

    model = InceptionResnetV1(pretrained='vggface2').eval()
    # model.logits = nn.Identity()
    
    # model.load_state_dict(torch.load(args.model), strict=False) # Using transfer-learned model
    print('Face Recognition model is loaded')

    known_face_names = set()
    face_list = list()
    person_dict = dict()
    for i in os.listdir("./image"):
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
        emb = make_embedding(mtcnn, model, person_name=i, number_of_images=person_dict[i])
        known_face_embeddings.append(emb)
    
    # Run video through MTCNN
    video = mmcv.VideoReader(args.video)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    print("the number of frames: ", len(frames))

    frames_tracked = list()

    font = ImageFont.load_default()
    font_size = 20
    font = ImageFont.truetype("./arial.ttf", font_size)

    unknown_label = 0

    for i, frame in enumerate(frames):
        print('Tracking frame: {}/{}'.format(i, len(frames)-1), end='')
        boxes, _ = mtcnn.detect(frame) ## extracting the bounding box
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)

        # crop_image using detection model
        imgs = mtcnn(frame)

        if boxes is None:
            pass
        else:
            for box, img in zip(boxes, imgs):
                # draw box
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=2)
                face_embedding = model(img.unsqueeze(0)).detach().numpy()

                # See if the face is a match for the known face(s)
                matches = compare_faces(known_face_embeddings, face_embedding, tolerance=args.tolerance)
                face_distances = face_distance(known_face_embeddings, face_embedding)

                print(' Matches: ', matches)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    max_index = np.argmax(face_distances)
                    name = known_face_names[max_index]
                
                else:
                    known_face_embeddings.append(face_embedding)
                    known_face_names.append(f"Unlabeled_person_{unknown_label}")

                    face = frame.crop((box[0], box[1], box[2], box[3]))
                    face.save(f"./unlabeled_image/Unlabeled_person_{unknown_label}.jpg", "JPEG")

                    unknown_label += 1

                text_posi = (box[0], box[1])
                draw.text(text_posi, name, font=font)

        # Add to frame list
        frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    
    print('\nDone')
    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    
    video_tracked = cv2.VideoWriter(os.path.join("./outputs", "onscreen_" + args.video.split("/")[-1]), fourcc, 25.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()
    print("----- Video saved -----")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="Select cuda:0 or cuda:1")
    parser.add_argument("--model", type=str, help="Set model file path ex, ./saved_models/val_5.83.pth")
    parser.add_argument("--video", type=str, default="./test_video/obama2.mp4")
    parser.add_argument("--output", type=str, default="./outputs/test.mp4")
    parser.add_argument("--tolerance", type=float, default=0.5)

    args = parser.parse_args()
    main(args)