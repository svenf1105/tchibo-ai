# TechLabs Digital Shaper program: Streamlit webapp using CenterNet model for face and logo detection in images from the Tchibo GmbH online community

## Project description and goals
The aim of this team project was to build and train an image recognition algorithm to support the Tchibo GmbH in gaining insights from images uploaded to their online community. We generated a specifically annotated database and trained a Tensorflow CenterNet model on it to detect the Tchibo logo and human faces in the images. The model was implemented in a Streamlit webapp where files can be uploaded and passed into the model, parameters of the models chosen to the individual use case and the final results exported in a .csv file including the respective predictions


## Short Description for each file

### Files for gathering data
[Bing_Image_Downloader.ipynb](https://github.com/svenf1105/tchibo-ai/blob/f8b511d0c1d50c02015216e60a77110a9dcec8ca/Bing_Image_Downloader.ipynb) --> Download pictures from Bing Images by a certain search term.

[image_downloader.py](https://github.com/svenf1105/tchibo-ai/blob/f8b511d0c1d50c02015216e60a77110a9dcec8ca/image_downloader.py) --> Download pictures from a csv file filled with links.

[Openimagedataset_fiftyone_downloader.ipynb](https://github.com/svenf1105/tchibo-ai/blob/4d13db5bda14a74a62c0054d40c1b2d61934a388/Openimagedataset_fiftyone_downloader.ipynb) --> Download pictures from the openimage dataset based on certain classes.

### Files regarding first tries with image classification

### Files regarding the preperation, training, inferencing of our Object Detector 

### File for the streamlit application






--> provide link for the medium article to comprehend the project
