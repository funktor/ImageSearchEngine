from __future__ import print_function
import tensorflow as tf
import numpy as np
import boto3
import os, math

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_file_list(root_dir):
    file_list, labels = [], []
    for root, directories, filenames in os.walk(root_dir):
        for d in directories:
            for filename in os.listdir(os.path.join(root, d)):
                if any(ext in filename for ext in extensions):
                    file_list.append(os.path.join(root, d, filename))
                    labels.append(d)
     
    u = list(set(labels))
    h = {u[i]:i for i in range(len(u))}
    return file_list, [h[x] for x in labels]

def image_example(image_string, label):
    image_shape = tf.io.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def convert_to_tfrecords(img_data_dir, tfrecords_file):
    filenames, labels = get_file_list(img_data_dir)
    
    with tf.io.TFRecordWriter(tfrecords_file) as writer:
        for i in range(len(filenames)):
            image_string = open(filenames[i], 'rb').read()
            tf_example = image_example(image_string, labels[i])
            writer.write(tf_example.SerializeToString())

if __name__=="__main__":
    data_dir = '/Users/amondal/Documents/datasets/Natural Images/natural_images'
    tfrecords_file = 'image_search_data.tfrecords'
    
    print('Converting to TFRecords...')
    convert_to_tfrecords(data_dir, tfrecords_file)
    
    print('Uploading to s3...')
    s3 = boto3.resource('s3')
    BUCKET = "data-bucket-sagemaker-image-search"

    s3.Bucket(BUCKET).upload_file(tfrecords_file, tfrecords_file)