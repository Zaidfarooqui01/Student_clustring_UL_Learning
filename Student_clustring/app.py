import streamlit as st
import pandas as pd
import joblib
import altair as alt

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("student_clustered.csv")
    pipeline = joblib.load("pipeline.joblib")
    return df, pipeline

def main():
    st.title("🎓 Student Learning Behavior Clustering")

    df, pipeline = load_data()
    
    st.subheader("📊 Cluster Overview")
    st.write(df.groupby('cluster')[['quiz_time','video_time','notes_time']].mean())

    st.subheader("🗺️ PCA Projection")
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x='pc1', y='pc2', color='cluster:N', tooltip=['student_id','quiz_time','video_time','notes_time']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.subheader("🔍 Student Lookup")
    student = st.selectbox("Choose a student:", df['student_id'])
    row = df[df['student_id']==student]
    st.write(row[['quiz_time','video_time','notes_time','cluster']])

    st.subheader("✅ New Student Cluster Prediction")
    q = st.number_input("Quiz time (minutes)", min_value=0.0, value=10.0)
    v = st.number_input("Video time (minutes)", min_value=0.0, value=10.0)
    n = st.number_input("Notes time (minutes)", min_value=0.0, value=10.0)
    if st.button("Predict cluster"):
        import numpy as np
        X_new = np.array([[q,v,n]])
        scaler, pca, kmeans = pipeline['scaler'], pipeline['pca'], pipeline['kmeans']
        x_s = scaler.transform(X_new)
        x_p = pca.transform(x_s)
        cl = kmeans.predict(x_p)[0]
        st.success(f"Belongs to cluster {cl}")

if __name__ == "__main__":
    main()
