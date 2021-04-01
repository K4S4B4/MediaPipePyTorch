import numpy as np
import torch
import cv2
import sys

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazepalm import BlazePalm
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

back_detector = True

face_detector = BlazeFace(back_model=back_detector).to(gpu)
if back_detector:
    face_detector.load_weights("blazefaceback.pth")
    face_detector.load_anchors("anchors_face_back.npy")
else:
    face_detector.load_weights("blazeface.pth")
    face_detector.load_anchors("anchors_face.npy")

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights("blazepalm.pth")
palm_detector.load_anchors("anchors_palm.npy")
palm_detector.min_score_thresh = .75

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights("blazehand_landmark.pth")

face_regressor = BlazeFaceLandmark().to(gpu)
face_regressor.load_weights("blazeface_landmark.pth")


WINDOW='test'
cv2.namedWindow(WINDOW)
if len(sys.argv) > 1:
    capture = cv2.VideoCapture(sys.argv[1])
    mirror_img = False
else:
    capture = cv2.VideoCapture(0)
    mirror_img = True

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

while hasFrame:
    frame_ct +=1

    #if mirror_img:
    #    frame = np.ascontiguousarray(frame[:,::-1,::-1])
    #else:
    #    frame = np.ascontiguousarray(frame[:,:,::-1]) # BRG to RGB

    img1, img2, scale, pad = resize_pad(frame)



    x = torch.from_numpy(img1).to(gpu)
    x = x.unsqueeze(0)
    points = palm_detector(x)
    #normalized_palm_detections = palm_detector.predict_on_image(img1)
    #palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)
    #xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())

    res = 256
    points1 = np.array([[0, 0, res-1],
                        [0, res-1, 0]], dtype=np.float32).T
    affines = []
    imgs = []
    for i in range(points.shape[0]):
        pts = points[i, :, :3].cpu().numpy().T
        M = cv2.getAffineTransform(pts, points1)
        img = cv2.warpAffine(img1, M, (res,res))#, borderValue=127.5)
        img = torch.tensor(img).to(gpu)
        imgs.append(img)

        affine = cv2.invertAffineTransform(M).astype('float32')
        affine = torch.tensor(affine).to(gpu)
        affines.append(affine)

    if imgs:
        imgs = torch.stack(imgs) #.permute(0,3,1,2).float() / 255. #/ 127.5 - 1.0
        affines = torch.stack(affines)
    else:
        imgs = torch.zeros((0, 3, res, res)).to(gpu)
        affines = torch.zeros((0, 2, 3)).to(gpu)

    #img, affine2, box2 = hand_regressor.extract_roi(img1, xc, yc, theta, scale)
    flags2, handed2, normalized_landmarks2 = hand_regressor(imgs)
    landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2, affines)
    
    for i in range(len(flags2)):
        landmark, flag = landmarks2[i], flags2[i]
        if flag>.5:
            draw_landmarks(img1, landmark[:,:2], HAND_CONNECTIONS, size=2)

    draw_roi(img1, points)

    cv2.imshow(WINDOW, img1)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
