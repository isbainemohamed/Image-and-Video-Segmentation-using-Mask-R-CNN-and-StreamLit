import streamlit as st
import cv2
import os
from PIL import Image

# necessary imports
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

################

 ## Root directory of the project
from Mask_RCNN.cocoapi.PythonAPI.pycocotools.coco import CocoConfig

ROOT_DIR = os.path.abspath("./")

import warnings
warnings.filterwarnings("ignore")

 ## Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
 ## Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from pycocotools import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('', "Mask_RCNN/samples/coco/mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)
# Load weights trained on MS-COCO
model.load_weights('Mask_RCNN/samples/coco/mask_rcnn_coco.h5', by_name=True)



# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# define functions to overlay masks on original image
# define random colors
def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


# apply mask to image
def apply_mask(image, mask, color, alpha=0.5):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image

    # take the image and apply the mask, box, and Label


def display_instances(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)
    if not n_instances:
        print("NO INSTANCES TO DISPLAY")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
    return image


def process_video(video_path, output_path, verbose=True):
    # get the fps rate of the video:
    video = cv2.VideoCapture(video_path)

    framespersecond = int(video.get(cv2.CAP_PROP_FPS))
    if verbose:
        print("The total number of frames in this video is ", framespersecond)
    # get the size of the frame (weight, height)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = video.read()
    # h, w, c = image.shape
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    # define a list of object detected in the video
    detected = []
    # loop over frames and predict content

    out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), framespersecond, size)
    pbar = tqdm(desc='while loop', total=length)
    while success:
        # cv2.imwrite("/content/Mask_RCNN/dance/frame%d.jpg" % count, image)     # save frame as JPEG file
        # Run detection
        results = model.detect([image], verbose=1)
        r = results[0]
        output = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        detected.append(r['class_ids'])
        out_video.write(image)
        pbar.update(1)
        success, image = video.read()
    out_video.release()
    if verbose:
        print("Output video is created at {}".format(output_path))
    return detected

def process_image(image_path,output_path):
    # Load a random image from the images folder
    image = skimage.io.imread(image_path)

    # original image
    #plt.figure(figsize=(12, 10))
    #skimage.io.imshow(image)
    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    output = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    cv2.imwrite(output_path, output)
    print("image processed")

    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def download_success():
    #st.balloons()
    st.success('✅ Download Successful !!')

#######################
st.set_page_config(
    page_title="Instance Segmentator",
    page_icon="🖼",
    layout="centered",

)



upload_path = "uploads/"
download_path = "downloads/"



st.title("🖼📷 Instance Segmentator 🏡🏙")

segmentation_type = st.sidebar.selectbox('Select segmentation type 🎯',["Image","Video","Live Feed"])



st.title("🖼📷 Instance Segmentator 🏡🏙")
if segmentation_type == "Image":
    st.info('✨ Supports all popular image formats 📷 - PNG, JPG, BMP 😉')
    uploaded_file = st.file_uploader("Upload Image 🖼", type=["png","jpg","bmp","jpeg"])

    if uploaded_file is not None:
        with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
            f.write((uploaded_file).getbuffer())
        with st.spinner(f"Working... 💫"):
            uploaded_image = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
            downloaded_image = os.path.abspath(os.path.join(download_path,str("segmented_"+uploaded_file.name)))
            print("input",uploaded_image)
            print("input", downloaded_image)
            process_image(uploaded_image, downloaded_image)

            final_image = Image.open(downloaded_image)
            print("Opening ",final_image)
            st.markdown("---")
            st.image(final_image, caption='This is how your final image looks like 😉')
            with open(downloaded_image, "rb") as file:
                if uploaded_file.name.endswith('.jpg') or uploaded_file.name.endswith('.JPG'):
                    if st.download_button(
                                            label="Download Segmented Image 📷",
                                            data=file,
                                            file_name=str("segmented_"+uploaded_file.name),
                                            mime='image/jpg'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.jpeg') or uploaded_file.name.endswith('.JPEG'):
                    if st.download_button(
                                            label="Download Segmented Image 📷",
                                            data=file,
                                            file_name=str("segmented_"+uploaded_file.name),
                                            mime='image/jpeg'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.PNG'):
                    if st.download_button(
                                            label="Download Segmented Image 📷",
                                            data=file,
                                            file_name=str("segmented_"+uploaded_file.name),
                                            mime='image/png'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.bmp') or uploaded_file.name.endswith('.BMP'):
                    if st.download_button(
                                            label="Download Segmented Image 📷",
                                            data=file,
                                            file_name=str("segmented_"+uploaded_file.name),
                                            mime='image/bmp'
                                         ):
                        download_success()
        st.warning('⚠ Please upload your Image file 😯')

if segmentation_type == "Video":
    st.info('✨ Supports all popular video formats 🎥 - MP4, MOV, MKV, AVI 😉')
    uploaded_file = st.file_uploader("Upload Video 📽", type=["mp4","avi","mov","mkv"])
    if uploaded_file is not None:
        with open(os.path.join(upload_path,uploaded_file.name),"wb") as f:
            f.write((uploaded_file).getbuffer())
        with st.spinner(f"Working... 💫"):
            uploaded_video = os.path.abspath(os.path.join(upload_path,uploaded_file.name))
            downloaded_video = os.path.abspath(os.path.join(download_path,str("segmented_"+uploaded_file.name)))
            process_video(uploaded_video,downloaded_video)


            final_video = open(downloaded_video, 'rb')
            video_bytes = final_video.read()
            print("Opening ",final_video)
            st.markdown("---")
            with open(downloaded_video, "rb") as file:
                if uploaded_file.name.endswith('.avi') or uploaded_file.name.endswith('.AVI'):
                    st.success('✅ Your results are ready !! 😲')
                    if st.download_button(
                                            label="Download Segmented Video 📽",
                                            data=file,
                                            file_name=str("segmented_"+uploaded_file.name),
                                            mime='video/x-msvideo'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.mp4') or uploaded_file.name.endswith('.MP4'):
                    st.success('✅ Your results are ready !! 😲')
                    if st.download_button(
                                            label="Download Segmented Video 📽",
                                            data=file,
                                            file_name=str("segmented_"+uploaded_file.name),
                                            mime='video/mp4'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.mov') or uploaded_file.name.endswith('.MOV'):
                    st.success('✅ Your results are ready !! 😲')
                    if st.download_button(
                                            label="Download Segmented Video 📽",
                                            data=file,
                                            file_name=str("segmented_"+uploaded_file.name),
                                            mime='video/quicktime'
                                         ):
                        download_success()

                if uploaded_file.name.endswith('.mkv') or uploaded_file.name.endswith('.MKV'):
                    st.success('✅ Your results are ready !! 😲')
                    if st.download_button(
                                            label="Download Segmented Video 📽",
                                            data=file,
                                            file_name=str("segmented_"+uploaded_file.name),
                                            mime='video/x-matroska'
                                         ):
                        download_success()
    else:
        st.warning('⚠ Please upload your Video file 😯')




