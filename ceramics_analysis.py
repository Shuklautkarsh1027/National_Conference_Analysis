
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv('electronic_ceramics_data.csv')

# Basic Preprocessing
df_clean = df.dropna()

# Clustering
features = df_clean[['Dielectric_Constant', 'Loss_Tangent', 'Permittivity_Stability']]
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(features)

# Visualization
sns.pairplot(df_clean, hue='Cluster', palette='Set2')
plt.suptitle('Cluster Analysis of Ceramic Samples', y=1.02)
plt.savefig('cluster_plot.png')
plt.show()

# Save clustered data
df_clean.to_csv('clustered_ceramics.csv', index=False)
