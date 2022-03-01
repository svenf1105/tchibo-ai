import io
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
tf.gfile = tf.io.gfile

from PIL import Image
import datetime
from skimage import io as skio
import requests
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
from io import *
import plotly.express as px

st.set_page_config(layout='wide')
showWarningOnDirectExecution = False

header = st.container()
csv_upload = st.container()
pictures = st.container()
model = st.container()
examples = st.container()
csv_download = st.container()
analytics = st.container()

@st.cache()
def load_model(selected_model):
    filepath_centernet = r'C:\Users\paul9\Documents\Tchibo\tchibo-ai\StreamlitWebApp\Exported_centernet_512_final\saved_model' 
    filepath_mobilenet = r'C:\Users\paul9\Documents\Tchibo\tchibo-ai\StreamlitWebApp\Exported_ssd_mobilenet_v1_640_final\saved_model'
    filepath_resnet = r'C:\Users\paul9\Documents\Tchibo\tchibo-ai\StreamlitWebApp\Exported_ssd_resnet50_v1_640_final\saved_model'

    if 'Centernet' in selected_model:
        model = tf.saved_model.load(filepath_centernet)
        st.write('1')
    elif 'mobilenet' in selected_model:
        model = tf.saved_model.load(filepath_mobilenet)
        st.write('2')
    else:
        model = tf.saved_model.load(filepath_resnet)
        st.write('3')
    return model

def load_image_from_url(url):
    ret = None
    while ret is None:            
        try:
            ret = skio.imread(url)
        except:
            ret = None
            pass
    return ret
    

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
        df_pics = df_pics.drop_duplicates(subset='image')

with pictures:
    st.header('Pictures')
    selected_processing = st.radio('Select if the images should only be tagged or also saved with bounding boxes',
    ('Tag pictures in csv file','Save tagged pictures with bounding boxes'))
    if selected_processing == 'Save tagged pictures with bounding boxes':
        filepath_images = st.text_input('Determine a filepath where the images are stored')
    height = st.slider('Select min height in pixels',0,5000) 
    width = st.slider('Select min width in pixels',0,5000) 
    st.write('Only the pictures with a resolution of at least {f1} * {f2} are considered'.format(f1=height, f2=width))

with model:
    st.header('Model')
    selected_model = st.radio('You can choose between two models:',('Centernet HG104 512x512 COCO17','ssd_mobilenet_v1_fpn_keras','ssd_resnet50_v1_fpn_keras'))
    model = load_model(selected_model)
    
    classification_threshold = st.slider('Select min detection score',0.0,1.0,step=0.01)

    st.text('Model detects logos and faces...')
    progress= 0
    progress_bar = st.progress(progress)
    
    n_pic_text = st.empty()
    n_pic = 1
    time_start = datetime.datetime.now()

    df_pics['n_tchibologos'] = 0
    df_pics['n_faces'] = 0 

    for picture in df_pics['image']:
        

        pic = load_image_from_url(picture)
        pic_tensor = tf.convert_to_tensor(pic)
        
        if pic_tensor.shape[0] >= height and pic_tensor.shape[1] >= width:
            pic_tensor = pic_tensor[tf.newaxis, ...]
            detections = model(pic_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}

            detections['detection_classes_str'] = np.where(detections['detection_classes']==1,'Tchibo Logo','Face') 
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            n_detection = 0
            for detection in detections['detection_scores']:
                if detections['detection_scores'][n_detection] >= classification_threshold:
                    df_pics.loc[df_pics['image']==picture,'detectionclass_'+str(n_detection)] = detections['detection_classes_str'][n_detection]
                    df_pics.loc[df_pics['image']==picture,'detectionscore_'+str(n_detection)] = detections['detection_scores'][n_detection]
                    df_pics.loc[df_pics['image']==picture,'x_min_'+str(n_detection)] = detections['detection_boxes'][n_detection][0] * 1000
                    df_pics.loc[df_pics['image']==picture,'y_min_'+str(n_detection)] = detections['detection_boxes'][n_detection][1] * 1000
                    df_pics.loc[df_pics['image']==picture,'x_max_'+str(n_detection)] = detections['detection_boxes'][n_detection][2] * 1000
                    df_pics.loc[df_pics['image']==picture,'y_max_'+str(n_detection)] = detections['detection_boxes'][n_detection][3] * 1000 

                    if df_pics.loc[df_pics['image']==picture,'detectionclass_'+str(n_detection)].item() == 'Tchibo Logo':
                        df_pics.loc[df_pics['image']==picture,'n_tchibologos'] += 1 
                    else: 
                        df_pics.loc[df_pics['image']==picture,'n_faces'] += 1 

                    if (selected_processing == 'Save tagged pictures with bounding boxes') and (filepath_images is not None):
                        pic_with_detections = pic.copy()
                        category_index = label_map_util.create_category_index_from_labelmap(r"C:\Users\paul9\OneDrive\StreamlitWebApp\label_map.pbtxt", use_display_name=True)
                        viz_utils.visualize_boxes_and_labels_on_image_array(
                            pic_with_detections,
                            detections['detection_boxes'],
                            detections['detection_classes'],
                            detections['detection_scores'],
                            category_index,
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=5,
                            min_score_thresh=classification_threshold,
                            agnostic_mode=False)
                        file_name = picture.split('/')[-1]
                        pic_with_detections = Image.fromarray(pic_with_detections.astype('uint8'), 'RGB')
                        pic_with_detections.save(filepath_images+'\\'+file_name)

                    n_detection += 1
                else:
                    n_detection += 1 
        progress += 1
        progress_bar.progress(100//df_pics['image'].shape[0]*progress)

        time_now = datetime.datetime.now()
        timedelta_minutes = round((time_now-time_start).total_seconds()/60,2)
        timedelta_hours = round((time_now-time_start).total_seconds()/3600,2)
        n_pic_text.write('Image '+str(n_pic)+' out of '+str(df_pics['image'].shape[0])+' images - running for '+str(timedelta_minutes)+' minutes / '+str(timedelta_hours)+' hours')
        n_pic += 1



    st.text('Display the first 10 rows in the tagged csv')
    st.write(df_pics.head(10))


with examples:
    st.header('Pictures (3 examples)')

    category_index = label_map_util.create_category_index_from_labelmap(r"C:\Users\paul9\Documents\Tchibo\tchibo-ai\StreamlitWebApp\Centernet_HG104_512x512_COCO17\label_map.pbtxt", use_display_name=True)
    url_list = df_pics.loc[df_pics.detectionscore_0 > classification_threshold,].image.tolist()[0:3]
    
    for picture in url_list:
        image = load_image_from_url(picture)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_with_detections = image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=classification_threshold,
            agnostic_mode=False)

        # display output image
        st.image(image_with_detections )


with csv_download:
    st.header('Download CSV')
    st.download_button('Download the csv-file containing with assigned categories', df_pics.to_csv(), file_name='tagged_images.csv')

with analytics:
    st.header('Some superficial analytics with an classification threshold of: '+str(classification_threshold))
    
    col1, col2 = st.columns(2)
    
    with col1:
        detection_larger0 = df_pics[(df_pics['n_tchibologos']>0) | (df_pics['n_faces']>0)].shape[0]
        detections_equals0 = df_pics[(df_pics['n_tchibologos']==0) | (df_pics['n_faces']==0)].shape[0]
        fig1 = px.pie(df_pics, values=[detection_larger0,detections_equals0], names=['n-Detections >= 1','n-Detections = 0'],title=f'How many % of the images contain at least one detection')
        fig1.update_traces(textinfo='percent',marker_line_color='rgb(0,0,0)',marker_line_width=1, opacity=0.8)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        detection_bothlarger0 = df_pics[(df_pics['n_tchibologos']>0) & (df_pics['n_faces']>0)].shape[0]
        fig2 = px.pie(df_pics, values=[detection_bothlarger0,detections_equals0], names=['n-Detections of Logos & Faces >= 1','n-Detections = 0'],title=f'How many % of the images contain at least one Tchibo Logo and Face detection')
        fig2.update_traces(textinfo='percent',marker_line_color='rgb(0,0,0)',marker_line_width=1, opacity=0.8)
        st.plotly_chart(fig2,use_container_width=True)

    col3, col4 = st.columns(2)
    with col3: 
        countuserid = df_pics['id_user'].value_counts().to_frame()
        if countuserid['id_user'].max() > 100:
            activity_max = countuserid['id_user'].max()
        else:
            activity_max = 500
        bins = [0,1,3,10,100,activity_max]
        labels = ['0-1','1-3','3-10','10-100','100-'+str(activity_max)]
        countuserid['activity_count'] = pd.cut(countuserid['id_user'], bins = bins, labels=labels)
        countuserid[countuserid['activity_count']=='0-1'].shape[0]
        fig3 = px.pie(df_pics, values=[countuserid[countuserid['activity_count']=='0-1'].shape[0],countuserid[countuserid['activity_count']=='1-3'].shape[0],countuserid[countuserid['activity_count']=='3-10'].shape[0],countuserid[countuserid['activity_count']=='10-100'].shape[0],countuserid[countuserid['activity_count']=='100-'+str(activity_max)].shape[0]], names=labels,title=f'activities (posted images) by users')
        fig3.update_traces(textinfo='percent',marker_line_color='rgb(0,0,0)',marker_line_width=1, opacity=0.8)
        st.plotly_chart(fig3,use_container_width=True)

    with col4:
        only_logo = df_pics[(df_pics['n_tchibologos']>0) & (df_pics['n_faces']==0)].shape[0]
        only_faces = df_pics[(df_pics['n_faces']>0) & (df_pics['n_tchibologos']==0)].shape[0]

        fig4 = px.pie(df_pics, values=[only_logo,only_faces,detection_bothlarger0], names=['only Tchibo Logo(s)','only Face(s)','both'],title=f'Comparison Logos and Faces')
        fig4.update_traces(textinfo='percent',marker_line_color='rgb(0,0,0)',marker_line_width=1, opacity=0.8)
        st.plotly_chart(fig4,use_container_width=True)
