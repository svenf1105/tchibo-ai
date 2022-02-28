# Streamlit webapp using CenterNet model for face and logo detection in images from the Tchibo GmbH online community

## Project description and goals
The aim of this team project was to build and train an image recognition algorithm to support the Tchibo GmbH in gaining insights from images uploaded to their online community. We generated a specifically annotated database and trained a Tensorflow CenterNet model on it to detect the Tchibo logo and human faces in the images. The model was implemented in a Streamlit webapp where files can be uploaded and passed into the model, parameters of the models chosen to the individual use case and the final results exported in a .csv file including the respective predictions


## Short Description for each file

### Files for gathering data
[Bing_Image_Downloader.ipynb](https://github.com/svenf1105/tchibo-ai/blob/f8b511d0c1d50c02015216e60a77110a9dcec8ca/Bing_Image_Downloader.ipynb) --> Download pictures from Bing Images based on a certain search term.

[image_downloader.py](https://github.com/svenf1105/tchibo-ai/blob/f8b511d0c1d50c02015216e60a77110a9dcec8ca/image_downloader.py) --> Download pictures from a csv file filled with links.

[Openimagedataset_fiftyone_downloader.ipynb](https://github.com/svenf1105/tchibo-ai/blob/4d13db5bda14a74a62c0054d40c1b2d61934a388/Openimagedataset_fiftyone_downloader.ipynb) --> Download pictures from the openimage dataset based on certain classes.

### Files regarding first tries with image classification
[facerecognition_OpenCV.ipynb](https://github.com/svenf1105/tchibo-ai/blob/6d094bc3474fc1613369edd94f508ca3f8bc6e66/facerecognition_OpenCV.ipynb) --> Small experimental and warm up project.

[TchiboLogoDetection_MobilNetV2.ipynb](https://github.com/svenf1105/tchibo-ai/blob/6d094bc3474fc1613369edd94f508ca3f8bc6e66/TchiboLogoDetection_MobilNetV2.ipynb) --> A try of a pre-trained model as well as a small net of our own to classify Tchibologos.

### Files regarding the preperation, training, inferencing of our Object Detector 
[1_explorative_analysis_of_picdataset.ipynb](https://github.com/svenf1105/tchibo-ai/blob/ff9905d76854fd1c604e10f655320cd1864bde42/1_explorative_analysis_of_picdataset.ipynb) --> With this notebook we can explore the shapes of our input pictures for our modell. Based on the results we can further adjust our dataset to provide the optimal input for the model.

[2_Filter_train_dataset_based_on_image_size.ipynb](https://github.com/svenf1105/tchibo-ai/blob/ff9905d76854fd1c604e10f655320cd1864bde42/2_Filter_train_dataset_based_on_image_size.ipynb) --> With this notebook we can modify our train/test data by filtering out pictures with specific dimensions. Then we can move the pictures and the belonging annotation into a new folder.

[3_TFOD_Tchiboproject_Training_Eval_Export_.ipynb](https://github.com/svenf1105/tchibo-ai/blob/ff9905d76854fd1c604e10f655320cd1864bde42/3_TFOD_Tchiboproject_Training_Eval_Export_.ipynb) --> This notebook set ups the TensorFlow Object Detection Api, provides all tasks needed to start training, trains the model, evaluates the result and saves the finished model.

[4_Model_Inference_with_visualization.ipynb](https://github.com/svenf1105/tchibo-ai/blob/ff9905d76854fd1c604e10f655320cd1864bde42/4_Model_Inference_with_visualization.ipynb) --> For model inference we can plot our predictions with this notebook.

[4_Tchiboproject_predict_tag_and_move_tcompics.ipynb](https://github.com/svenf1105/tchibo-ai/blob/ff9905d76854fd1c604e10f655320cd1864bde42/4_Tchiboproject_predict_tag_and_move_tcompics.ipynb) --> Predict pictures and move them to a certain folder based on the predictions made by the model. The notebook can be used to extend the dataset and/or detect strength and weaknesses of the model.

[4_Tchiboproject_predict_to_csv.ipynb](https://github.com/svenf1105/tchibo-ai/blob/ff9905d76854fd1c604e10f655320cd1864bde42/4_Tchiboproject_predict_to_csv.ipynb) --> The notebook provides the functionality to tag links of pictures provided in a csv file with the predictions (class, confidence score and bonding box coordinates) made by the model.

### File for the streamlit application
[frontend_streamlit_tchibo.py](https://github.com/svenf1105/tchibo-ai/blob/5a46ddc366e8519b6a8e0f3d1c8c01042734b50c/frontend_streamlit_tchibo.py) --> Provide the developed model and functionalities inside our notebooks as convenient as possible as a webapp. 

### Medium article of the project.
(link)

