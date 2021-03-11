# importing required libraries
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import patches

train = pd.read_csv(r'C:/Users/NAY3HC/Desktop/core test/CV/KERAS-FASTERRCNN/keras-frcnn/train.csv')
train.head()
# Number of unique training images
train['image_id'].nunique()
train['class_id'].value_counts()

fig = plt.figure()
#add axes to the image
ax = fig.add_axes([0,0,1,1])

# read and plot the image
image = plt.imread(r'C:/Users/NAY3HC/Desktop/core test/CV/KERAS-FASTERRCNN/keras-frcnn/train/50a418190bc3fb1ef1633bf9678929b3.jpeg')
plt.imshow(image)

for _,row in train[train.image_id == "50a418190bc3fb1ef1633bf9678929b3"].iterrows():

    xmin = row.xmin
    xmax = row.xmax
    ymin = row.ymin
    ymax = row.ymax
    
    width = xmax - xmin
    height = ymax - ymin

# add bounding boxes to the image
    rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = None, facecolor = 'none')
    
    ax.add_patch(rect)

data = pd.DataFrame()
data['format'] = train['image_id']

# as the images are in train_images folder, add train_images before the image name
for i in range(data.shape[0]):
    data['format'][i] = 'image_id/' + data['format'][i]

# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] +'.jpeg' +',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['class_id'][i]

data.to_csv('annotate.txt', header=None, index=None, sep=' ')