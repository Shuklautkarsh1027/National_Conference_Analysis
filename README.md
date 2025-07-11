# Electronic Ceramics Clustering using K-Means

This project applies unsupervised machine learning (K-Means Clustering) on a dataset of electronic ceramics to group materials based on their dielectric properties. This kind of clustering helps in identifying patterns and grouping similar ceramic materials for research and industrial applications.

## Objectives

- Clean and preprocess ceramic material data
- Explore relationships between key electrical properties
- Scale features for accurate clustering
- Apply K-Means clustering and determine optimal number of clusters
- Visualize cluster behavior and distribution
- Export clustered results for further use

## Dataset Features

The dataset includes the following key features:

- Dielectric_Constant  
- Loss_Tangent  
- Permittivity_Stability

These are important electrical characteristics of ceramic materials used in capacitors, insulators, and sensors.

## ML Techniques Used

- Preprocessing & Cleaning  
  (Handling missing values, feature scaling with StandardScaler)
  
- Clustering Algorithm  
  KMeans from sklearn.cluster with n_clusters=3

- Model Selection  
  - Elbow Method (Inertia)
  - Silhouette Score

- Post-analysis  
  - Cluster center interpretation
  - Cluster-wise feature distribution
  - Correlation heatmap

## Visualizations

Generated plots included:

- correlation_heatmap.png — Heatmap of all numerical features  
- elbow_plot.png — Inertia vs Clusters  
- silhouette_score_plot.png — Silhouette Score vs Clusters  
- cluster_pairplot.png — Pairwise scatterplot with clusters  
- cluster_distributions.png — Boxplots of features per cluster

## Output Files

- clustered_ceramics.csv — Final dataset with cluster labels
- cluster_centers.csv — Cluster centers in original scale
- PNG image files — Visualizations for analysis and reporting


