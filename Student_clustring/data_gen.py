import numpy as np
import pandas as pd

def generate_student_data(n_students=500, seed=42):
    np.random.seed(seed)
    # Simulating 3 learning behavior profiles
    clusters = {
        0: {'quiz':(30,5), 'video':(20,5), 'notes':(15,3)},
        1: {'quiz':(10,3), 'video':(50,8), 'notes':(5,2)},
        2: {'quiz':(50,7), 'video':(10,3), 'notes':(25,5)},
    }

    data = []
    for i in range(n_students):
        label = np.random.choice([0,1,2], p=[0.4,0.3,0.3])
        q_mean, v_mean, n_mean = clusters[label]['quiz'][0], clusters[label]['video'][0], clusters[label]['notes'][0]
        quiz_time = max(1, np.random.normal(q_mean, clusters[label]['quiz'][1]))
        video_time = max(1, np.random.normal(v_mean, clusters[label]['video'][1]))
        notes_time = max(1, np.random.normal(n_mean, clusters[label]['notes'][1]))
        data.append([f"student_{i}", quiz_time, video_time, notes_time, label])
    
    df = pd.DataFrame(data, columns=['student_id','quiz_time','video_time','notes_time','true_cluster'])
    return df

if __name__ == "__main__":
    df = generate_student_data()
    df.to_csv("student_behavior.csv", index=False)
    print("Dataset generated: student_behavior.csv")
