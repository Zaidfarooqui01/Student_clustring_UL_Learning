import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

def run_pipeline(csv_path="student_behavior.csv", n_clusters=3):
    df = pd.read_csv(csv_path)
    X = df[['quiz_time','video_time','notes_time']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_pca)

    df['cluster'] = labels
    df_pca = pd.DataFrame(X_pca, columns=['pc1','pc2'])
    df_final = pd.concat([df, df_pca], axis=1)

    joblib.dump({'scaler':scaler, 'pca':pca, 'kmeans':kmeans}, "pipeline.joblib")
    df_final.to_csv("student_clustered.csv", index=False)
    print("Pipeline completed. Results saved to student_clustered.csv")

if __name__ == "__main__":
    run_pipeline()
