import pandas as pd
import numpy as np
import scipy.io
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the dataset
df = pd.read_csv('data/marriage.csv', header=None)
# Drop the very last column to create X
X = df.drop(columns=[df.columns[-1]]).values
# Select the very last column to create y
y = df[df.columns[-1]].values

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Part 1: Train and evaluate classifiers ---
# Naive Bayes
nb = GaussianNB(var_smoothing=1e-3)
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
print(f"KNN Accuracy: {knn_acc:.4f}")

# --- Part 2: PCA and Decision Boundaries ---
# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X) # Apply to all data for plotting
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Retrain on 2D data
nb_pca = GaussianNB(var_smoothing=1e-3).fit(X_train_pca, y_train_pca)
lr_pca = LogisticRegression(max_iter=1000).fit(X_train_pca, y_train_pca)
knn_pca = KNeighborsClassifier().fit(X_train_pca, y_train_pca)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
classifiers = [nb_pca, lr_pca, knn_pca]
titles = ['Naive Bayes', 'Logistic Regression', 'KNN']

# Create mesh grid
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

for i, clf in enumerate(classifiers):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[i].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    axes[i].set_title(f'Decision Boundary: {titles[i]}')
    axes[i].set_xlabel('Principal Component 1')
    axes[i].set_ylabel('Principal Component 2')

plt.tight_layout()
plt.savefig('decision_boundaries.png')
print("Plot saved as decision_boundaries.png")

print(f"2D NB Accuracy: {nb_pca.score(X_test_pca, y_test_pca):.4f}")
print(f"2D LR Accuracy: {lr_pca.score(X_test_pca, y_test_pca):.4f}")
print(f"2D KNN Accuracy: {knn_pca.score(X_test_pca, y_test_pca):.4f}")