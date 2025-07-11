import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv('electronic_ceramics_data.csv')

# Basic Preprocessing
df_clean = df.dropna()

# ======================= Exploratory Data Analysis ===========================
print("Basic Info:")
print(df_clean.info())
print("\nDescriptive Stats:")
print(df_clean.describe())

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# ======================= Feature Scaling ===========================
features = df_clean[['Dielectric_Constant', 'Loss_Tangent', 'Permittivity_Stability']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ======================= Optimal Cluster Check ===========================
inertias = []
silhouettes = []

for k in range(2, 7):
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    preds = kmeans_temp.fit_predict(features_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouettes.append(silhouette_score(features_scaled, preds))

# Elbow Plot
plt.figure(figsize=(6, 4))
plt.plot(range(2, 7), inertias, marker='o')
plt.title('Elbow Method - Inertia vs Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid()
plt.savefig('elbow_plot.png')
plt.show()

# Silhouette Plot
plt.figure(figsize=(6, 4))
plt.plot(range(2, 7), silhouettes, marker='o', color='green')
plt.title('Silhouette Score vs Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid()
plt.savefig('silhouette_score_plot.png')
plt.show()

# ======================= Final Clustering ===========================
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(features_scaled)

# ======================= Cluster Center Info ===========================
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=features.columns)
print("\nCluster Centers (Original Scale):")
print(centroid_df)

# ======================= Cluster Pairplot ===========================
sns.pairplot(df_clean, hue='Cluster', palette='Set2', diag_kind='kde')
plt.suptitle('Cluster Analysis of Ceramic Samples', y=1.02)
plt.savefig('cluster_pairplot.png')
plt.show()

# ======================= Cluster-wise Distribution ===========================
plt.figure(figsize=(10, 4))
for i, col in enumerate(features.columns):
    plt.subplot(1, 3, i+1)
    sns.boxplot(data=df_clean, x='Cluster', y=col, palette='Set2')
    plt.title(f'{col} by Cluster')
plt.tight_layout()
plt.savefig('cluster_distributions.png')
plt.show()

# ======================= Save Output ===========================
df_clean.to_csv('clustered_ceramics.csv', index=False)
centroid_df.to_csv('cluster_centers.csv', index=False)
print("\nSaved clustered data and centroid info.")
