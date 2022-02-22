import io
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from io import *
import requests
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils


header = st.container()
csv_upload = st.container()
pictures = st.container()
model = st.container()
examples = st.container()
csv_download = st.container()
analytics = st.container()

@st.cache()
def load_model(selected_model):
    filepath_centernet = r'C:\Users\paul9\Documents\Tchibo\StreamlitWebApp\Centernet_HG104_512x512_COCO17\saved_model' 
    filepath_yolo = r''

    if selected_model == 'Centernet HG104 512x512 COCO17':
        model = tf.saved_model.load(filepath_centernet)
    else:
        model = tf.saved_model.load(filepath_yolo)

    return model

with header:
    st.title('This is our webapp for the Image Recognition Problem provided by Tchibo')
    st.text('The result of this project is captured in a model that classifies images \nfrom the Tchibo community. The goal is to detect faces and the Tchibo logo \non images.')
    

with csv_upload:
    st.header('CSV File Upload')

    csv = st.file_uploader('Select csv-file containing the links', 'csv')
    if csv is not None:
        df_pics = pd.read_csv(csv)
        st.text('Display the first 5 rows in the selected csv')
        st.write(df_pics.head())


with pictures:
    st.header('Pictures')
    selected_processing = st.radio('Select if the images should only be tagged or also saved with bounding boxes',
    ('Tag pictures in csv file','Save tagged pictures with bounding boxes'))
    height = st.slider('Select min height in pixels',0,5000) 
    width = st.slider('Select min width in pixels',0,5000) 
    st.write('Only the pictures with a resolution of at least {f1} * {f2} are considered'.format(f1=height, f2=width))

with model:
    st.header('Model')
    selected_model = st.radio('You can choose between two models:',('Centernet HG104 512x512 COCO17','YOLO Classifier'))
    model = load_model(selected_model)
    
    classification_threshold = st.slider('Select min detection score',0.0,1.0,step=0.01)

    st.text('Model detects logos and faces...')
    progress= 0
    progress_bar = st.progress(progress)

    for picture in df_pics['image']:
        r = requests.get(picture, stream=True, timeout=10)
        
        if r.status_code != 200:
            continue
        pic = Image.open(io.BytesIO(r.content))
        pic_tensor = tf.convert_to_tensor(pic)
        
        if pic_tensor.shape[0] >= height and pic_tensor.shape[1] >= width:
            pic_tensor = pic_tensor[tf.newaxis, ...]
            detections = model(pic_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}

            detections['detection_classes'] = np.where(detections['detection_classes']==1,'Tchibo Logo','Face') 

            n_detection = 0
            for detection in detections['detection_scores']:
                if detections['detection_scores'][n_detection] >= classification_threshold:
                    df_pics.loc[df_pics['image']==picture,'detectionclass_'+str(n_detection)] = detections['detection_classes'][n_detection]
                    df_pics.loc[df_pics['image']==picture,'detectionscore_'+str(n_detection)] = detections['detection_scores'][n_detection]
                    df_pics.loc[df_pics['image']==picture,'x_min_'+str(n_detection)] = detections['detection_boxes'][n_detection][0] * 1000
                    df_pics.loc[df_pics['image']==picture,'y_min_'+str(n_detection)] = detections['detection_boxes'][n_detection][1] * 1000
                    df_pics.loc[df_pics['image']==picture,'x_max_'+str(n_detection)] = detections['detection_boxes'][n_detection][2] * 1000
                    df_pics.loc[df_pics['image']==picture,'y_max_'+str(n_detection)] = detections['detection_boxes'][n_detection][3] * 1000
                     
                    n_detection += 1
                else:
                    n_detection += 1 
        progress += 1 
        progress_bar.progress(100//df_pics['image'].shape[0]*progress)


    st.text('Display the first 10 rows in the tagged csv')
    st.write(df_pics.head(10))


with examples:
    st.header('Pictures (examples)')
    for picture in df_pics[df_pics['detectionclass_0']!=np.nan]['image'].head(3):
        r = requests.get(picture, stream=True, timeout=10)
        if r.status_code != 200:
            continue
        pic = Image.open(io.BytesIO(r.content))

        category_index=label_map_util.create_category_index_from_labelmap(r'C:\Users\paul9\Documents\Tchibo\StreamlitWebApp\Centernet_HG104_512x512_COCO17\label_map.pbtxt',use_display_name=True),

        visualization_utils.visualize_boxes_and_labels_on_image_array(
            pic,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=0.5,
            agnostic_mode=False)

        
        
with csv_download:
    st.header('Download CSV')
    st.download_button('Download the csv-file containing with assigned categories', df_pics.to_csv(), file_name='tagged_images.csv')

with analytics:
    st.header('Some superficial analytics')
