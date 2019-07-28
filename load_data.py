import pandas as pd
import pickle
def load_image(path, size=None):
	img = Image.open(path)
	if not size is None:
	    img = img.resize(size=size, resample=Image.LANCZOS)
	img = np.array(img)

	img = img / 255.0

	if (len(img.shape) == 2):
		img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
	return img
def load_dataset(filename):	
	data = pd.read_csv(filename, error_bad_lines=False, sep = '|')
	filenames_train=data["image_name"]
	filenames_train=filenames_train.unique()	
	#run this if u want to create a .pkl file containing transfer_values
	"""this data in the pkl file can be obtained when running the program without computing 
	and processing through the images again"""
	"""
	with open('transfer_values.pkl', 'wb') as f:
	    #creates transfer_values.pkl if the file does not exist
	    pickle.dump(transfer_values, f)
	"""
	"""
	transfer_values=np.asarray(process_images('flickr30k_images', filenames_train[:150]))
	transfer_values.shape
	"""
	#loading previously saved  pickle file and storing in variable called transfer_values
	with open('transfer_values.pkl', 'rb') as f:
	      transfer_values = pickle.load(f)
	return (data,transfer_values,filenames_train)