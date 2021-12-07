import imageio
import glob
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.neural_network import MLPClassifier

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

def normalize(training_data, validation_data, testing_data): #normalizare standard
    scaler = preprocessing.StandardScaler()
    scaler.fit(training_data)
    return scaler.transform(training_data), scaler.transform(validation_data), scaler.transform(testing_data)

def mlptrain(X_train, Y_train, activation_fct, hidden_layers, lr, momentum, epochs, alfa=0.0001):
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation_fct, solver='sgd',
                                             learning_rate='constant', learning_rate_init=lr,
                                             max_iter=epochs, momentum=momentum, alpha=alfa)
    model.fit(X_train, Y_train)
    return model

train_images, validation_images, test_images = normalize(train_images, validation_images, test_images)

model = mlptrain(train_images, train_labels, 'relu', (500,500), 0.01, 0.9, 2000)
predicted_labels = model.predict(validation_images) #predictie pentru matricea de confuzie
predictions = model.predict(test_images) #predictie pentru submisie
print(model.score(validation_images, validation_labels)) #0,7698

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

with open('submission4.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    for i in range(len(test_images)):
        writer.writerow([test[i], predictions[i]])