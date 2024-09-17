import cv2
import face_recognition
import numpy as np
import os
def load_images_from_folder(folder):
    images=[]
    filenames=[]
    for filename in os.listdir(folder):
        img=cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames
def encode_faces(images):
    encoded_faces=[]
    for image in images:
        face_locations=face_recognition.face_locations(image)
        face_encodings=face_recognition.face_encodings(image, face_locations)
        encoded_faces.append(face_encodings)
    return encoded_faces
def find_individual_in_photos(individual_encoding, group_encodings):
    matching_photos=[]
    for idx, group_photo_encodings in enumerate(group_encodings):
        matches=face_recognition.compare_faces(group_photo_encodings, individual_encoding)
        if any(matches):
            matching_photos.append(idx)
    return matching_photos
group_photos_folder=r"D:\Users\Desktop\grp"
group_images, group_filenames=load_images_from_folder(group_photos_folder)
individual_photo_path=r"D:\Users\Desktop\nialll.jpeg"
individual_image=cv2.imread(individual_photo_path)
group_encodings=encode_faces(group_images)
individual_face_location=face_recognition.face_locations(individual_image)[0]
individual_face_encoding=face_recognition.face_encodings(individual_image, [individual_face_location])[0]
matching_photos=find_individual_in_photos(individual_face_encoding, group_encodings)
print("Individual found in the following group photos:")
for idx in matching_photos:
    print(group_filenames[idx])
