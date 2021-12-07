import imageio
import glob
import numpy as np
from sklearn import preprocessing, metrics, svm

train_images = []
for im_path in glob.glob("train/*.png"):
    train_images.append(imageio.imread(im_path).reshape(-1)) #fiecare imagine este o lista de 1024 pixeli

test_images = []
for im_path in glob.glob("test/*.png"):
    test_images.append(imageio.imread(im_path).reshape(-1))

validation_images = []
for im_path in glob.glob("validation/*.png"):
    validation_images.append(imageio.imread(im_path).reshape(-1))

train_labels = []
file = open('train.txt', 'r')
for line in file.readlines():
    ln = [x for x in line.split(",")]
    train_labels.append(int(ln[1]))

validation_labels = []
file2 = open('validation.txt', 'r')
for line in file2.readlines():
    ln = [x for x in line.split(",")]
    validation_labels.append(int(ln[1]))

test = []
file2 = open('test.txt', 'r')
for line in file2.readlines():
    ln = line.rstrip("\n")
    test.append(ln)

def normalize_data( train_images, validation_images, test_images, type = None):
    if type == None:
        return train_images,test_images
    if type == 'standard': #normalizare standard
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_images)
        return scaler.transform(train_images), scaler.transform(validation_images), scaler.transform(test_images)
    if type == 'l1': #normalizam datele folosind norma l1
        normalizer = preprocessing.Normalizer(norm='l1')
        normalizer.fit(train_images)
        return normalizer.transform(train_images), normalizer.transform(validation_images), normalizer.transform(test_images)
    if type == 'l2': #normalizam datele folosind norma l2
        normalizer = preprocessing.Normalizer(norm='l2')
        normalizer.fit(train_images)
        return normalizer.transform(train_images), normalizer.transform(validation_images), normalizer.transform(test_images)

train_images, validation_images, test_images = normalize_data(train_images, validation_images, test_images, 'l2')

clf = svm.SVC(C=2.0, kernel='rbf')
clf.fit(train_images, train_labels)

####predictie pentru verificarea acuratetii pe datele de validare
prediction_labels = clf.predict(validation_images)
print("Scorul obtinut pe datele de validare este: ")
print(metrics.accuracy_score(validation_labels, prediction_labels)) #0.7416

def confusion_matrix(y_true, y_pred):
    num_classes = max(y_true.max(), y_pred.max()) + 1
    conf_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(y_true)):
        conf_matrix[int(y_true[i]), int(y_pred[i])] += 1
    return conf_matrix

predicted_labels = np.array(prediction_labels)
validation_labels = np.array(validation_labels)


print("Matricea de confuzie este:")
print(confusion_matrix(validation_labels, predicted_labels))


predictions = clf.predict(test_images)

import csv

with open('submission3.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    for i in range(len(test_images)):
        writer.writerow([test[i], predictions[i]])
