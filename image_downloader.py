import pandas as pd 
import urllib.request

#----------------------------------
def url_to_jpg(i, url, file_path):
	'''
		Args:
			-- i: number of image
			-- url: a URL address of a given image 
			-- file_path : where to save the final image 

	'''
	filename = 'image-{}.jpg'.format(i)
	full_path = '{}{}'.format(file_path, filename) 
	urllib.request.urlretrieve(url, full_path)

	print('{} saved'.format(filename))

	return None 

#-------------------------------------


#Set Constants
FILENAME = 'tchibo_community_images.csv'
FILE_PATH = "Tchibo_images/"

#Read List of URLs as Pandas data frame
urls = pd.read_csv(FILENAME)

# Save Images to the Directory by iterating through the list

for i, url in enumerate(urls.values):
	url_to_jpg(i, url[0], FILE_PATH)
