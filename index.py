import cv2 as cv
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = os.listdir(f'pelak/1')

x = np.empty((0, 256))
y = np.array([])

for filenames in file_path:
    im = cv.imread(f"pelak/1/"+filenames)
    im2 = cv.resize(im, (8,32))
    im3 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    im4 = im3.flatten()
    x = np.append(x, [im4], axis=0)
    y = np.append(y, 1)

file_path = os.listdir(f'pelak/2')

x2 = np.empty((0, 256))
y2 = np.array([])

for filenames in file_path:
    im = cv.imread(f"pelak/2/"+filenames)
    im2 = cv.resize(im, (8,32))
    im3 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    im4 = im3.flatten()
    x = np.append(x, [im4], axis=0)
    y = np.append(y, 2)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

model = LogisticRegression()

model.fit(x_train,y_train)


x1 = np.empty((0, 256))

im = cv.imread(f"pelak/1/1 (789).png")
im2 = cv.resize(im, (8,32))
im3 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
im4 = im3.flatten()
x1 = np.append(x1, [im4], axis=0)

y_pred_test = model.predict(x_test)

print(y_test)
print("---------------------------")
print(y_pred_test)

print("metric information")
print("-----------------------------")
print(f"accuracy_score: {accuracy_score(y_test,y_pred_test)*100:02f}")

out = model.predict(x1)
print(out)