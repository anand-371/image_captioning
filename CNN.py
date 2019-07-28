from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
def process_images(data_dir,filenames, batch_size=32):
    num_images = len(filenames)
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)
    start_index = 0

    while start_index < num_images:
        print_progress(count=start_index, max_count=num_images)

        end_index = start_index + batch_size

        if end_index > num_images:
            end_index = num_images
        current_batch_size = end_index - start_index

        for i, filename in enumerate(filenames[start_index:end_index]):
            path = os.path.join(data_dir, filename)
            img = load_image(path, size=img_size)
            image_batch[i] = img

        transfer_values_batch = image_model_transfer.predict(image_batch[0:current_batch_size])

        transfer_values[start_index:end_index] =transfer_values_batch[0:current_batch_size]

        start_index = end_index


    return transfer_values
def CNN():
    image_model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = image_model.get_layer('fc2')
    image_model_transfer = Model(inputs=image_model.input,outputs=transfer_layer.output)
    img_size = K.int_shape(image_model.input)[1:3]
    transfer_values_size = K.int_shape(transfer_layer.output)[1]
    return transfer_values_size