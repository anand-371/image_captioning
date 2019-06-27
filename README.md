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

The following is the summary of the VGG 16 model
				
![VGG16](https://user-images.githubusercontent.com/40825655/60294124-48358580-993e-11e9-823c-f0a302d7336b.PNG)
![VGG16-2](https://user-images.githubusercontent.com/40825655/60294202-74e99d00-993e-11e9-8760-4e4245a4fe64.PNG)

The following is the summary of the Recurrent layer
![Rnn](https://user-images.githubusercontent.com/40825655/60294464-0e18b380-993f-11e9-9526-62e5eec598f1.PNG)


the processed Tensorboard graphs are as follows
![graph_large_attrs_key=_too_large_attrs limit_attr_size=1024 run=](https://user-images.githubusercontent.com/40825655/60294609-6780e280-993f-11e9-9cde-89f5fec972f6.png)
