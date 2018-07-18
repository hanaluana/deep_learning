import os
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
#import inception_resnet_v2 
import inception_preprocessing
from tensorflow.contrib import slim

image_size = inception_resnet_v2.default_image_size
print(image_size)

# Input for checkpoint
#checkpoints_dir = '/Users/kimj0a/Documents/Project/yummy_project/transfer_learning_tutorial/floyd_classifier1_add/logs/wrapup1/'
checkpoints_dir = '/dataset/checkpoint'
train_from_raw = True
raw_checkpoint_file = 'inception_resnet_v2_2016_08_30.ckpt'

# Input for dataset
num_classes = 9
#dataset_dir = '/Users/kimj0a/Documents/Project/yummy_project/database/test'
dataset_dir = '/dataset'
#dataset_dir = '../dataset'
file_pattern_for_counting = 'classifier1'
file_pattern = file_pattern_for_counting + '_%s_*.tfrecord'
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured images related for foods.',
    'label': 'A label that is as such -- 0:normal 1:zoom-in 2:zoom-out 3:multiple 4:ganpan!! 5:inside 6:outside 7:menus 8:exclude '
}

# Get labels info
labels_file = dataset_dir + '/labels.txt' 
labels = open(labels_file, 'r')

# Training setting
num_epochs = 50
batch_size = 32  
initial_learning_rate = 0.001 #0.0009
learning_rate_decay_factor = 0.94
num_epochs_before_decay = 2
#log_dir = '../logs/codetest'
log_dir = '/output/log'


def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    
    if not tf.gfile.IsDirectory(checkpoints_dir):
        print("Recheck checkpoint_path!")
    else :
        print("Load checkpoint")

    if train_from_raw == True :
        checkpoint_path = os.path.join(checkpoints_dir, raw_checkpoint_file)
    else : 
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    # from slim tutorial
    checkpoint_exclude_scopes=["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    #exclude = ["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]
    #variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
    #variables_to_restore = slim.get_model_variables('InceptionResnetV2',exclude = exclude)
    return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore)

def get_split(split_name, dataset_dir, file_pattern=file_pattern, file_pattern_for_counting=file_pattern_for_counting):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later. 

    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting

    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    #Create a dictionary to refer each label to their string name
    labels_to_name = {}
    for line in labels:
        label, string_name = line.split(':')
        string_name = string_name[:-1] #Remove newline
        labels_to_name[int(label)] = string_name
   

    #First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    print("num_samples:", num_samples)

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size=batch_size, height=image_size, width=image_size, is_training=False):
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32,
            common_queue_min=8)

    image_raw, label = data_provider.get(['image', 'label'])
    
    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    
    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
            [image, image_raw, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=2 * batch_size,
            allow_smaller_final_batch=True)
    
    return images, images_raw, labels


with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = get_split('train', dataset_dir, file_pattern=file_pattern, file_pattern_for_counting = file_pattern_for_counting)
    images, _, labels = load_batch(dataset, height = image_size, width = image_size, is_training= True)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(images, num_classes = dataset.num_classes, is_training = True, dropout_keep_prob=0.8)

    # Specify the loss function:
    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
    total_loss = tf.losses.get_total_loss()

    # Create some summaries to visualize the training process:
    tf.summary.scalar('losses/Total_Loss', total_loss)
    
    predictions = tf.argmax(end_points['Predictions'], 1)
    accuracy, _ = tf.contrib.metrics.streaming_accuracy(predictions, labels)

    tf.summary.scalar('accuracy', accuracy)
    
    #Create the global step for monitoring the learning_rate and training.
    global_step = get_or_create_global_step()

    num_steps_per_epoch = int(dataset.num_samples / batch_size)
    total_steps = num_steps_per_epoch * num_epochs 
    decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

    #Define your exponentially decaying learning rate
    lr = tf.train.exponential_decay(
        learning_rate = initial_learning_rate,
        global_step = global_step,
        decay_steps = decay_steps,
        decay_rate = learning_rate_decay_factor,
        staircase = True)
    tf.summary.scalar('learning_rate', lr)
    #my_summary_op = tf.summary.merge_all()

    # Specify the optimizer and create the train op:
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    
    # Run the training:
    final_loss = slim.learning.train(
        train_op,
        logdir=log_dir,
        init_fn=get_init_fn(),
        number_of_steps=total_steps)
        
  
print('Finished training. Last batch loss %f' % final_loss)





