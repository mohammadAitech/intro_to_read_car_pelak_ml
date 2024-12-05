# README

## Logistic Regression for Image Classification

### Overview
This project demonstrates the use of Logistic Regression to classify grayscale images of license plates into two classes. The images are preprocessed, flattened, and fed into a Logistic Regression model for training and evaluation. The model is also used to predict the class of a single input image.

---

### Features
- Image preprocessing (resize, grayscale conversion, flattening)
- Classification of images into two classes (`Class 1` and `Class 2`)
- Evaluation of model performance using accuracy score
- Single-image prediction

---

### Project Structure
```
project/
├── pelak/
│   ├── 1/   # Images for Class 1
│   ├── 2/   # Images for Class 2
├── index.py # Main script
```

---

### Requirements
- Python 3.x
- OpenCV
- NumPy
- scikit-learn

Install dependencies using:
```bash
pip install opencv-python numpy scikit-learn
```

---

### How It Works

1. **Data Preparation:**
   - Images from the folders `pelak/1` and `pelak/2` are loaded.
   - Each image is resized to `8x32`, converted to grayscale, and flattened into a vector of size `256`.
   - The vectors are stored in arrays `x` and `y`, where `x` holds the features and `y` holds the class labels (`1` for `pelak/1`, `2` for `pelak/2`).

2. **Model Training:**
   - The dataset is split into training and testing sets using `train_test_split`.
   - A Logistic Regression model is trained on the training set.

3. **Evaluation:**
   - The trained model is evaluated on the test set using accuracy as the metric.

4. **Single Image Prediction:**
   - A single test image (`pelak/1/1 (789).png`) is processed and classified using the trained model.

---

### How to Run

1. Ensure the dataset (`pelak/1` and `pelak/2`) is correctly structured and contains the appropriate images.

2. Run the script:
   ```bash
   python index.py
   ```

3. The output will display:
   - True labels (`y_test`) and predicted labels (`y_pred_test`) for the test set.
   - The accuracy of the model.
   - The prediction for the single test image.

---

### Sample Output
```plaintext
[y_test labels]
---------------------------
[y_pred_test labels]

metric information
-----------------------------
accuracy_score: 95.000000

[single image prediction]
```

---

### Notes
- Ensure the input images in `pelak/1` and `pelak/2` are valid and readable by OpenCV.
- The dataset must be balanced for better model performance.
- Modify the script to handle more classes or implement additional preprocessing steps as needed.

---

### Author
Developed as a demonstration of machine learning for image classification using Logistic Regression.