import imageio
import glob
import numpy as np

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

class KnnClassifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def classify_image(self, test_image, num_neighbors=3, metric='l2'):
        if metric == 'l2':
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis=1)) #distanta L2
        else:
            distances = np.sum(np.abs(self.train_images - test_image), axis=1) #distanta L1
        sort_neighbors = np.argsort(distances)[:num_neighbors] #ia primii num_neighbors vecini
        nearest_neighbors_labels = [self.train_labels[index] for index in sort_neighbors]
        count_label_occurrences = np.bincount(nearest_neighbors_labels) #numara aparitiile label-urilor
        return np.argmax(count_label_occurrences) #alege labelul cel mai intalnit

    def classify_images(self, validation_images, num_neighbors=3, metric='l2'):
        pred_labels = np.zeros(len(validation_images))
        for i in range(len(validation_labels)):
            pred_labels[i] = self.classify_image(validation_images[i], num_neighbors, metric)
        return pred_labels


knn_classifier = KnnClassifier(train_images, train_labels)
pred_labels = knn_classifier.classify_images(validation_images)
accuracy = (pred_labels == validation_labels).mean()
print(accuracy) #0.22