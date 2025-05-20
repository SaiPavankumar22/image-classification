# 🧠 Face Recognition Classifier with Wavelet Features

A Python-based machine learning project that detects celebrity faces in images using OpenCV's Haar cascade, applies advanced wavelet transforms for feature extraction, and classifies them using ML models like SVM, Random Forest, and Logistic Regression.

---

## 🚀 Project Overview

This project builds a **celebrity face classification system** using a dataset of facial images. The core components include:

* Face & eye detection using **OpenCV**
* Image cropping and preprocessing
* Feature extraction using **Wavelet Transforms (PyWavelets)**
* Model training using multiple **ML classifiers** with hyperparameter tuning via **GridSearchCV**
* Model persistence using **Joblib**
* Deployment-ready with a **saved model** and **class dictionary**

---

## 🛠️ Technologies & Libraries Used

### ✅ **Computer Vision**

* **OpenCV**: For face and eye detection using Haar cascades
* **Haarcascade Classifiers**: Pre-trained XML models for frontal face and eye detection

### ✅ **Feature Engineering**

* **PyWavelets (pywt)**: Used **Discrete Wavelet Transform (DWT)** to extract edge features

  * Combined with raw pixel data for richer representation

### ✅ **Machine Learning**

* **Scikit-learn**: Core ML library used for:

  * Model training (SVM, Random Forest, Logistic Regression)
  * **Pipeline** creation
  * **GridSearchCV** for hyperparameter tuning
  * Performance evaluation using **classification report**

### ✅ **Model Deployment**

* **Joblib**: For saving the best model to disk
* **JSON**: To store class-to-label mapping for later inference

---

## 📸 Face & Eye Detection Logic

* Uses `cv2.CascadeClassifier` with Haarcascade XMLs
* Crops face regions **only if at least two eyes are detected**
* Extracted face is resized and saved to the `cropped/` directory

---

## 🧪 Feature Extraction

Each image is processed into:

* **Raw pixel values** resized to `32x32x3`
* **Wavelet-transformed features** (converted to grayscale first)

These are combined into a single feature vector by stacking vertically:

```python
combined_img = np.vstack((raw_img.reshape(32*32*3,1), wavelet_img.reshape(32*32,1)))
```

---

## 🤖 ML Models Trained

Three models are trained and evaluated using **GridSearchCV**:

| Model               | Parameters Tuned |
| ------------------- | ---------------- |
| SVM                 | `kernel`, `C`    |
| Random Forest       | `n_estimators`   |
| Logistic Regression | `C`              |

Results are stored in a Pandas DataFrame, and the best-performing model is serialized.

---

## 🧠 Sample Results (Example Output)

| Model               | Best Score | Best Params                                    |
| ------------------- | ---------- | ---------------------------------------------- |
| SVM                 | 0.91       | `{'svc__C': 100, 'svc__kernel': 'linear'}`     |
| Random Forest       | 0.82       | `{'randomforestclassifier__n_estimators': 10}` |
| Logistic Regression | 0.87       | `{'logisticregression__C': 5}`                 |

---

## 💾 Model Saving

* Best model is saved as `saved_model.pkl` using `joblib`
* `class_dictionary.json` maps class indices to celebrity names

```python
joblib.dump(best_clf, 'saved_model.pkl')

with open("class_dictionary.json", "w") as f:
    f.write(json.dumps(class_dict))
```

---

## ✅ How to Run the Project

1. Clone the repository and place raw images in `dataset/celebrity_name/`
2. Ensure Haarcascade XML files are placed correctly in `./opencv/haarcascades/`
3. Run the script to:

   * Detect and crop faces
   * Extract features (raw + wavelet)
   * Train models and evaluate using cross-validation
   * Save best model and class dictionary

---

## 📌 Future Enhancements

* Deploy as a **Flask or FastAPI web app**
* Add **real-time webcam-based face detection**
* Expand dataset for better generalization
* Include **model interpretability** using tools like **SHAP or LIME**

---

## 👨‍💻 Author

**Sai Pavan Kumar Devisetti**

* AI/ML enthusiast | Backend Developer
* Projects in computer vision, healthtech, fintech, and industrial automation
