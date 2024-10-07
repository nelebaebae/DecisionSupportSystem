import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fungsi untuk metode SAW
def saw_method(matrix, weights, criteria_types):
    normalized_matrix = np.zeros_like(matrix)
    
    for j in range(matrix.shape[1]):
        if criteria_types[j] == 'Keuntungan':
            normalized_matrix[:, j] = matrix[:, j] / np.max(matrix[:, j])
        elif criteria_types[j] == 'Biaya':
            normalized_matrix[:, j] = np.min(matrix[:, j]) / matrix[:, j]
    
    weighted_matrix = normalized_matrix * weights
    scores = weighted_matrix.sum(axis=1)
    
    return scores

# Fungsi untuk metode WP dengan perhitungan vektor V
def wp_method(matrix, weights, criteria_types):
    weighted_matrix = np.zeros_like(matrix)
    
    for j in range(matrix.shape[1]):
        if criteria_types[j] == 'Keuntungan':
            weighted_matrix[:, j] = matrix[:, j] ** weights[j]
        elif criteria_types[j] == 'Biaya':
            weighted_matrix[:, j] = (1 / matrix[:, j]) ** weights[j]
    
    scores = np.prod(weighted_matrix, axis=1)
    v_vector = scores / scores.sum()
    
    return scores, v_vector

# Fungsi untuk metode TOPSIS
def topsis_method(matrix, weights, criteria_types):
    normalized_matrix = np.zeros_like(matrix)
    
    for j in range(matrix.shape[1]):
        if criteria_types[j] == 'Keuntungan':
            normalized_matrix[:, j] = matrix[:, j] / np.sqrt((matrix[:, j] ** 2).sum())
        elif criteria_types[j] == 'Biaya':
            normalized_matrix[:, j] = np.min(matrix[:, j]) / matrix[:, j]
    
    weighted_matrix = normalized_matrix * weights
    ideal_positive = np.max(weighted_matrix, axis=0)
    ideal_negative = np.min(weighted_matrix, axis=0)
    
    distance_positive = np.sqrt(((weighted_matrix - ideal_positive) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((weighted_matrix - ideal_negative) ** 2).sum(axis=1))
    
    scores = distance_negative / (distance_positive + distance_negative)
    
    return scores

# Streamlit app
st.title("Decision-Making Application")
st.markdown("This application helps you perform decision-making using SAW, WP, and TOPSIS methods. Please input the criteria and alternatives to begin.")

# Input jumlah kriteria dan alternatif
st.sidebar.header("Input Parameters")
num_criteria = st.sidebar.number_input("Number of Criteria", min_value=1, value=3)
num_alternatives = st.sidebar.number_input("Number of Alternatives", min_value=1, value=3)

# Input nama kriteria dan jenisnya (Keuntungan/Biaya)
criteria_names = []
criteria_types = []
criteria_weights = []

st.write("### Input Criteria and Weights")
for i in range(num_criteria):
    cols = st.columns([2, 2, 1])  # Organize input side by side
    criteria_name = cols[0].text_input(f"Criteria Name {i+1}", value=f"Criteria {i+1}")
    criteria_type = cols[1].selectbox(f"Criteria Type {i+1}", options=['Keuntungan', 'Biaya'], index=0, key=f"type_{i}")
    weight = cols[2].slider(f"Weight {i+1}", 0.0, 1.0, 0.5, key=f"weight_{i}")
    
    criteria_names.append(criteria_name)
    criteria_types.append(criteria_type)
    criteria_weights.append(weight)

criteria_weights = np.array(criteria_weights)

# Input nama alternatif dan nilai matriks keputusan
alternative_names = []
matrix = np.zeros((num_alternatives, num_criteria))

st.write("### Input Alternatives and Their Values")
for i in range(num_alternatives):
    alternative_name = st.text_input(f"Alternative Name {i+1}", value=f"Alternative {i+1}")
    alternative_names.append(alternative_name)
    
    cols = st.columns(num_criteria)
    for j in range(num_criteria):
        matrix[i, j] = cols[j].number_input(f"{criteria_names[j]} for {alternative_name}", value=5.0, key=f"value_{i}_{j}")

# Tampilkan matriks keputusan
st.write("### Decision Matrix")
df_matrix = pd.DataFrame(matrix, columns=criteria_names, index=alternative_names)
st.dataframe(df_matrix)

# Pilih metode
method = st.selectbox("Select Method", ("SAW", "WP", "TOPSIS"))

# Hitung dan tampilkan hasil
if st.button("Calculate"):
    if method == "SAW":
        scores = saw_method(matrix, criteria_weights, criteria_types)
    elif method == "WP":
        scores, v_vector = wp_method(matrix, criteria_weights, criteria_types)
    elif method == "TOPSIS":
        scores = topsis_method(matrix, criteria_weights, criteria_types)

    # Tampilkan skor akhir
    st.write(f"### Final Scores using {method} Method")
    df_scores = pd.DataFrame(scores, columns=["Score"], index=alternative_names)
    st.dataframe(df_scores)
    
    # Jika metode WP, tampilkan juga vektor V
    if method == "WP":
        st.write("### Vector V for WP Method")
        df_v_vector = pd.DataFrame(v_vector, columns=["V"], index=alternative_names)
        st.dataframe(df_v_vector)

    # Rekomendasi alternatif terbaik
    best_alternative = alternative_names[np.argmax(scores)]
    st.success(f"**Recommended Alternative:** {best_alternative}")

    # Visualisasi hasil
    st.write("### Visualization of Scores")
    fig, ax = plt.subplots()
    ax.bar(alternative_names, scores, color='skyblue')
    ax.set_ylabel("Scores")
    ax.set_title(f"Scores by {method} Method")
    st.pyplot(fig)
