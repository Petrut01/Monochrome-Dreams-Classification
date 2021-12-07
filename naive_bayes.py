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


bins = np.linspace(start=0, stop=255, num=11) # returneaza intervalele

def values_to_bins(train_images, bins): #discretizeaza datele
    x_to_bins = np.digitize(train_images,bins)-1
    return x_to_bins


from sklearn.naive_bayes import MultinomialNB

X_train = values_to_bins(train_images, bins)
X_validation = values_to_bins(validation_images,bins)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, train_labels) #antrenam modelul
test_labels = naive_bayes_model.predict(test_images) #predictie pentru submisie
predicted_labels = naive_bayes_model.predict(X_validation) #predictie pentru matricea de confuzie
print("Scorul obtinut pe datele de validare este: ")
print(naive_bayes_model.score(X_validation, validation_labels)) #0.3916

def confusion_matrix(y_true, y_pred):
    num_classes = max(y_true.max(), y_pred.max()) + 1
    conf_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(y_true)):
        conf_matrix[int(y_true[i]), int(y_pred[i])] += 1
    return conf_matrix

predicted_labels = np.array(predicted_labels)
validation_labels = np.array(validation_labels)


print("Matricea de confuzie este:")
print(confusion_matrix(validation_labels, predicted_labels))

import csv

with open('submission.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    for i in range(len(test_images)):
        writer.writerow([test[i], test_labels[i]])