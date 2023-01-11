from shutil import move
import os
from glob import glob
import pandas as pd
from functools import reduce
from xml.etree import ElementTree as et

import warnings
warnings.filterwarnings('ignore')


xmlfiles = glob('./pics/*.xml')
# replace \\ with /
def replace_text(x): return x.replace('\\', '/')


xmlfiles = list(map(replace_text, xmlfiles))
xmlfiles

# data cleaning
# xml_list = list(map(lambda x: x.replace('\\', '/'), xml_list))
# xml_list

# read xml files


def extract_text(filename):
    tree = et.parse(filename)
    root = tree.getroot()

    # extract filename
    image_name = root.find('filename').text
    # width and height of the image
    width = root.find('size').find('width').text
    height = root.find('size').find('height').text
    objs = root.findall('object')
    parser = []
    for obj in objs:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin = bndbox.find('ymin').text
        ymax = bndbox.find('ymax').text
        parser.append([image_name, width, height,
                      name, xmin, xmax, ymin, ymax])

    return parser


parser_all = list(map(extract_text, xmlfiles))
data = reduce(lambda x, y: x+y, parser_all)
df = pd.DataFrame(data, columns=[
                  'filename', 'width', 'height', 'name', 'xmin', 'xmax', 'ymin', 'ymax'])
df.head()
df.shape
df['name'].value_counts()

# type conversion
cols = ['width', 'height', 'xmin', 'xmax', 'ymin', 'ymax']
df[cols] = df[cols].astype(int)
df.info()
# df.categorical_column_name.value_counts().plot.bar()

#!pip install pandas --upgrade --user.

# center x, center y
df['center_x'] = ((df['xmax']+df['xmin'])/2)/df['width']
df['center_y'] = ((df['ymax']+df['ymin'])/2)/df['height']
# w
df['w'] = (df['xmax']-df['xmin'])/df['width']
# h
df['h'] = (df['ymax']-df['ymin'])/df['height']

df.head()

images = df['filename'].unique()
len(images)

# 80% train and 20% test
img_df = pd.DataFrame(images, columns=['filename'])
# shuffle and pick 80% of images
img_train = tuple(img_df.sample(frac=0.8)['filename'])

img_test = tuple(img_df.query(f'filename not in {img_train}')[
                 'filename'])  # take rest 20% images

len(img_train), len(img_test)

train_df = df.query(f'filename in {img_train}')
test_df = df.query(f'filename in {img_test}')

train_df.head()

test_df.head()

# label encoding


def label_encoding(x):
    labels = {'rat': 0}
    return labels[x]


train_df['id'] = train_df['name'].apply(label_encoding)
test_df['id'] = test_df['name'].apply(label_encoding)

train_df.head(10)


train_folder = 'pics/train'
test_folder = 'pics/test'

os.mkdir(train_folder)
os.mkdir(test_folder)

cols = ['filename', 'id', 'center_x', 'center_y', 'w', 'h']
groupby_obj_train = train_df[cols].groupby('filename')
groupby_obj_test = test_df[cols].groupby('filename')

# groupby_obj_train.get_group('000009.jpg').set_index('filename').to_csv('sample.txt',index=False,header=False)
# save each image in train/test folder and repective labels in .txt


def save_data(filename, folder_path, group_obj):
    # move image
    src = os.path.join('pics', filename)
    dst = os.path.join(folder_path, filename)
    move(src, dst)  # move image to the destination folder

    # save the labels
    text_filename = os.path.join(folder_path,
                                 os.path.splitext(filename)[0]+'.txt')
    group_obj.get_group(filename).set_index('filename').to_csv(
        text_filename, sep=' ', index=False, header=False)


filename_series = pd.Series(groupby_obj_train.groups.keys())
filename_series.apply(save_data, args=(train_folder, groupby_obj_train))

filename_series_test = pd.Series(groupby_obj_test.groups.keys())
filename_series_test.apply(save_data, args=(test_folder, groupby_obj_test))
