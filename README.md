#IMAGE CAPTIONING

The problem of generaing captions based on the image provided can be effectively solved using **Deep Neural Networks** <br />
in the following program we use a <br />
**CNN + RNN** architecture<br />
here the Convolutional Neural Network is used for extracting features from an image before passing it through the pipeline<br />
We use a VGG16 model that is trained for classifying images ,but instead of using the last classification layer,<br />
we redirect the output of the previous layer.This gives us a vector with 4096 elements that summarizes the image-contents.<br />
We will use this vector as the initial state of the Gated Recurrent Units(GRU).However we need to map the 4096 elements down to a <br />
vector with only 512 as this is the internal state-size of the GRU .To do this we need an intermediate fully-connected(dense) layer.<br />

**INPUT** : RGB Image size of (224,224)

**OUTPUT**: complete captions describing the image

**DATASET**: we are using Flickr30k dataset for training the model.

**LOSS FUNCTION:**
		We use a loss-function like sparse cross-entropy.
**OPTIMIZER:**
	We chose to use RMSprop over Adam optimizer as in some cases Adam Optimizer seems to diverge with Recurrent Neural Networks.
**Implemented using:**
				Tensorflow,keras
