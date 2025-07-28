**Project Overview**
Developed a deep learning-based traffic sign classification system using TensorFlow and Keras. Implemented data preprocessing, augmentation, and a convolutional neural network (CNN) to accurately identify multiple traffic sign categories. Achieved robust performance with early stopping and visualization of training metrics.

**Real-Life Scenario**
***Problem:***
Self-driving cars or traffic surveillance systems must accurately detect and classify traffic signs in real time to make decisions like stopping, slowing down, or turning.
***Our Solution:***
This CNN-based Traffic Sign Classifier is trained on labeled image data to classify signs like “Stop”, “Speed Limit 60 km/h”, or “No Entry”.
It can be integrated into:
	•	Autonomous driving systems
	•	Traffic violation detection systems
	•	Road safety alert tools
***Example:***
Suppose a real-time camera captures this image:
The model correctly classifies it as:
“Right-of-way at the next intersection”

**Dataset**
- The dataset consists of images organized into class folders (e.g., `00`, `01`, `11`, etc.).
- Each class represents a unique traffic sign.
- `labels.csv` maps class numbers to human-readable names.
- To load data, the script unzips `traffic_Data.zip` into the `/DATA/` folder.

**Technologies Used**
- Python
- TensorFlow + Keras
- OpenCV
- NumPy & Pandas
- Matplotlib

**Model Highlights**
- CNN with 4 convolutional blocks
- Data Augmentation
- Rescaling
- Dropout for regularization
- Sparse Categorical Crossentropy Loss
- Adam Optimizer
- Accuracy and Loss metrics

**Visualizations**
Sample image from dataset:
![Sample Image](visualizations/sample_image.png)

Training Accuracy & Loss:
![Accuracy Plot](visualizations/accuracy_plot.png)
![Loss Plot](visualizations/loss_plot.png)

**How to Run**
```bash
# Install required packages
pip install -r requirements.txt

# Run Python script
python traffic_sign_classifier.py