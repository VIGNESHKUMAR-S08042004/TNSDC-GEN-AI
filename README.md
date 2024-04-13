# TNSDC-GEN-AI
GAN-Based Number Prediction Project
This project utilizes Generative Adversarial Networks (GANs) to predict handwritten digits from the MNIST dataset. GANs are a type of deep learning model that consists of two neural networks, a generator and a discriminator, trained in an adversarial manner.

Overview
The project follows these main steps:

Data Loading and Exploration: Load the MNIST dataset containing handwritten digit images and explore the data's structure.
Data Preprocessing: Preprocess the images by converting them to one-dimensional arrays and normalizing pixel values to the range [0, 1].
K-Means Clustering: Utilize K-Means clustering to group similar images together based on their pixel values.
Infer Cluster Labels: Associate the most probable label with each cluster in the K-Means model to map clusters to digit labels.
Predictions: Predict the labels of testing data by assigning them to the clusters with the most similar characteristics.
Evaluation: Evaluate the accuracy of the predictions using metrics such as accuracy score and homogeneity score.
Visualization: Visualize the centroids of clusters and the inferred labels using matplotlib.
Dependencies
Ensure you have the following dependencies installed:

Python (>=3.6)
Keras
NumPy
Matplotlib
Scikit-learn
Usage
Clone the repository:
bash
Copy code
git clone <repository-url>
Navigate to the project directory:
bash
Copy code
cd GAN-based-number-prediction
Run the script:
Copy code
python number_prediction.py
Results
After running the script, you will see the accuracy of the predictions on the testing dataset. Additionally, visualizations of clustered centroids and inferred labels will be displayed.

Future Improvements
Experiment with different clustering algorithms for improved performance.
Explore more sophisticated GAN architectures for better image generation and prediction accuracy.
Optimize hyperparameters for both the clustering and GAN models.
