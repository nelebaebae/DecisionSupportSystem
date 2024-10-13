import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CSS
custom_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #1f1f2e;
    color: #f1f1f1;
}

[data-testid="stHeader"] {
    background-color: #1f1f2e;
}

h1, h2, h3, p {
    color: #f1f1f1 !important;
}

.stButton>button {
    background-color: #00b894;
    color: white !important;
}

.stDataFrame {
    background-color: #2d3436;
    color: #fff;
}

table tbody tr td {
    color: #dfe6e9 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# DSS Method
def saw_method(matrix, weights, criteria_types):
    norm_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[1]):
        if criteria_types[i] == 1:  # Benefit
            norm_matrix[:, i] = matrix[:, i] / np.max(matrix[:, i])
        else:  # Cost
            norm_matrix[:, i] = np.min(matrix[:, i]) / matrix[:, i]
    weighted_matrix = norm_matrix * weights
    scores = np.sum(weighted_matrix, axis=1)
    return norm_matrix, weighted_matrix, scores

# WP Method
def wp_method(matrix, weights, criteria_types):
    weighted_matrix = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        if criteria_types[j] == 'Keuntungan':  # Benefit criterion
            weighted_matrix[:, j] = matrix[:, j] ** weights[j]
        elif criteria_types[j] == 'Biaya':  # Cost criterion
            weighted_matrix[:, j] = (1 / matrix[:, j]) ** weights[j]
    scores = np.prod(weighted_matrix, axis=1)
    v_vector = scores / scores.sum()
    return scores, v_vector

# Topsis Method
def topsis_method(matrix, weights, criteria_types):
    norm_matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=0))
    weighted_matrix = norm_matrix * weights
    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])

    for i in range(weighted_matrix.shape[1]):
        if criteria_types[i] == 1:
            ideal_best[i] = np.max(weighted_matrix[:, i])
            ideal_worst[i] = np.min(weighted_matrix[:, i])
        else:
            ideal_best[i] = np.min(weighted_matrix[:, i])
            ideal_worst[i] = np.max(weighted_matrix[:, i])

    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))
    scores = dist_worst / (dist_best + dist_worst)
    return norm_matrix, weighted_matrix, scores

# AHP Method
def ahp_method(pairwise_matrix):
    sum_columns = np.sum(pairwise_matrix, axis=0)
    normalized_matrix = pairwise_matrix / sum_columns
    priority_vector = np.mean(normalized_matrix, axis=1)
    return priority_vector, normalized_matrix, sum_columns

def calculate_consistency(criteria_matrix, priority_vector):
    weighted_sum = np.dot(criteria_matrix, priority_vector)
    lambda_max = np.mean(weighted_sum / priority_vector)
    n = len(criteria_matrix)
    CI = (lambda_max - n) / (n - 1)
    RI_values = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_values.get(n, 1.12)
    CR = CI / RI
    return lambda_max, CI, CR

# Visualization 
def visualize_results(scores, method, num_alternatives):
    st.subheader(f"Final Scores for {method}")
    st.write(pd.DataFrame({
        "Alternatives": [f"Alternative {i+1}" for i in range(num_alternatives)],
        "Scores": scores
    }).sort_values(by="Scores", ascending=False))

    fig, ax = plt.subplots()
    ax.bar([f"Alternative {i+1}" for i in range(num_alternatives)], scores, color="#00b894")
    ax.set_title(f"{method} - Alternative Scores")
    ax.set_ylabel("Scores")
    st.pyplot(fig)

    best_alternative = np.argmax(scores) + 1
    st.markdown(f"**Recommendation: The best alternative is Alternative {best_alternative}**")
    return best_alternative

# Streamlit Layout
st.title("Decision Support System (DSS) Calculator")

# Method Selection
method_choice = st.sidebar.selectbox("Choose a Method", ["SAW", "WP", "TOPSIS", "AHP"])

# SAW Method
if method_choice == "SAW":
    st.header("Simple Additive Weighting (SAW)")
    num_criteria = st.number_input("Number of Criteria", min_value=2, max_value=10, value=3)
    num_alternatives = st.number_input("Number of Alternatives", min_value=2, max_value=10, value=3)

    st.subheader("Decision Matrix")
    decision_matrix = pd.DataFrame(np.zeros((num_alternatives, num_criteria)), 
                                   columns=[f"Criterion {i+1}" for i in range(num_criteria)])
    decision_matrix = st.data_editor(decision_matrix)

    st.subheader("Weights")
    weights = pd.DataFrame(np.ones((1, num_criteria)), columns=decision_matrix.columns)
    weights = st.data_editor(weights)

    st.subheader("Criteria Types (Benefit/Cost)")
    criteria_types = []
    for i in range(num_criteria):
        criteria_type = st.selectbox(f"Criterion {i+1} Type", ["Benefit", "Cost"], key=f"saw_criteria_{i}")
        criteria_types.append(1 if criteria_type == "Benefit" else 0)

    if st.button("Calculate SAW"):
        decision_matrix_values = decision_matrix.to_numpy()
        weights_values = weights.to_numpy().flatten()
        criteria_types_values = np.array(criteria_types)

        norm_matrix, weighted_matrix, scores = saw_method(decision_matrix_values, weights_values, criteria_types_values)

        st.subheader("Normalized Decision Matrix")
        st.write(pd.DataFrame(norm_matrix, columns=decision_matrix.columns))

        st.subheader("Weighted Normalized Decision Matrix")
        st.write(pd.DataFrame(weighted_matrix, columns=decision_matrix.columns))

        visualize_results(scores, "SAW", num_alternatives)

# WP Method
elif method_choice == "WP":
    st.header("Weighted Product (WP)")
    num_criteria = st.number_input("Number of Criteria", min_value=2, max_value=10, value=3)
    num_alternatives = st.number_input("Number of Alternatives", min_value=2, max_value=10, value=3)

    st.subheader("Decision Matrix")
    decision_matrix = pd.DataFrame(np.zeros((num_alternatives, num_criteria)), 
                                   columns=[f"Criterion {i+1}" for i in range(num_criteria)])
    decision_matrix = st.data_editor(decision_matrix)

    st.subheader("Weights")
    weights = pd.DataFrame(np.ones((1, num_criteria)), columns=decision_matrix.columns)
    weights = st.data_editor(weights)

    st.subheader("Criteria Types (Benefit/Cost)")
    criteria_types = []
    for i in range(num_criteria):
        criteria_type = st.selectbox(f"Criterion {i+1} Type", ["Keuntungan", "Biaya"], key=f"wp_criteria_{i}")
        criteria_types.append(criteria_type)

    if st.button("Calculate WP"):
        decision_matrix_values = decision_matrix.to_numpy()
        weights_values = weights.to_numpy().flatten()

        scores, v_vector = wp_method(decision_matrix_values, weights_values, criteria_types)

        visualize_results(v_vector, "WP", num_alternatives)

# TOPSIS Method
elif method_choice == "TOPSIS":
    st.header("TOPSIS")
    num_criteria = st.number_input("Number of Criteria", min_value=2, max_value=10, value=3)
    num_alternatives = st.number_input("Number of Alternatives", min_value=2, max_value=10, value=3)

    st.subheader("Decision Matrix")
    decision_matrix = pd.DataFrame(np.zeros((num_alternatives, num_criteria)), 
                                   columns=[f"Criterion {i+1}" for i in range(num_criteria)])
    decision_matrix = st.data_editor(decision_matrix)

    st.subheader("Weights")
    weights = pd.DataFrame(np.ones((1, num_criteria)), columns=decision_matrix.columns)
    weights = st.data_editor(weights)

    st.subheader("Criteria Types (Benefit/Cost)")
    criteria_types = []
    for i in range(num_criteria):
        criteria_type = st.selectbox(f"Criterion {i+1} Type", ["Benefit", "Cost"], key=f"topsis_criteria_{i}")
        criteria_types.append(1 if criteria_type == "Benefit" else 0)

    if st.button("Calculate TOPSIS"):
        decision_matrix_values = decision_matrix.to_numpy()
        weights_values = weights.to_numpy().flatten()
        criteria_types_values = np.array(criteria_types)

        norm_matrix, weighted_matrix, scores = topsis_method(decision_matrix_values, weights_values, criteria_types_values)

        st.subheader("Normalized Decision Matrix")
        st.write(pd.DataFrame(norm_matrix, columns=decision_matrix.columns))

        st.subheader("Weighted Normalized Decision Matrix")
        st.write(pd.DataFrame(weighted_matrix, columns=decision_matrix.columns))

        visualize_results(scores, "TOPSIS", num_alternatives)

# AHP Method
elif method_choice == "AHP":
    st.header("Analytic Hierarchy Process (AHP)")

    # AHP Comparison Values Guide
    st.write("""
    ### Panduan Nilai Perbandingan AHP
    | Kode | Nilai                                       |
    |------|---------------------------------------------|
    | 1    | Sama penting dengan                         |
    | 2    | Mendekati sedikit lebih penting dari        |
    | 3    | Sedikit lebih penting dari                  |
    | 4    | Mendekati lebih penting dari                |
    | 5    | Lebih penting dari                          |
    | 6    | Mendekati sangat penting dari               |
    | 7    | Sangat penting dari                         |
    | 8    | Mendekati mutlak dari                       |
    | 9    | Mutlak sangat penting dari                  |
    """)

    num_criteria = st.number_input("Number of criteria", min_value=2, max_value=10, value=3, key="ahp_num_criteria")
    num_alternatives = st.number_input("Number of alternatives", min_value=2, max_value=10, value=3, key="ahp_num_alternatives")

    if num_criteria and num_alternatives:
        # Pairwise comparison matrix for criteria
        st.subheader("Pairwise Comparison Matrix (Criteria)")
        criteria_matrix = pd.DataFrame(np.ones((num_criteria, num_criteria)),
                                       columns=[f"Criterion {i + 1}" for i in range(num_criteria)],
                                       index=[f"Criterion {i + 1}" for i in range(num_criteria)])
        criteria_matrix = st.data_editor(criteria_matrix, key="ahp_criteria_matrix")

        # Pairwise comparison matrices for alternatives
        st.subheader("Pairwise Comparison Matrices for Alternatives")
        alternative_results = []
        
        for k in range(num_criteria):
            st.write(f"Comparison Matrix for Criterion {k + 1}")
            alt_matrix = pd.DataFrame(np.ones((num_alternatives, num_alternatives)),
                                      columns=[f"Alternative {i + 1}" for i in range(num_alternatives)],
                                      index=[f"Alternative {i + 1}" for i in range(num_alternatives)])
            alt_matrix = st.data_editor(alt_matrix, key=f"alt_matrix_{k}")
            alternative_results.append(alt_matrix)

        if st.button("Calculate AHP", key="ahp_calculate"):
            try:
                # Ensure the matrices are filled in properly
                criteria_matrix_values = criteria_matrix.to_numpy()

                if np.all(criteria_matrix_values == 1):
                    st.error("Please fill in the Pairwise Comparison Matrix with meaningful values.")
                else:
                    # Calculate priority vector for criteria
                    priority_vector, normalized_matrix, criteria_sum = ahp_method(criteria_matrix_values)

                    # Prepare DataFrame for display
                    st.subheader("Normalized Criteria Matrix")
                    st.write(pd.DataFrame(normalized_matrix, columns=criteria_matrix.columns, index=criteria_matrix.index))

                    # Display priority vector for criteria
                    st.subheader("Priority Vector (Criteria Weights)")
                    st.write(pd.DataFrame(priority_vector, index=criteria_matrix.index, columns=["Priority Vector"]))

                    # Calculate λ_max, CI, CR
                    lambda_max, CI, CR = calculate_consistency(criteria_matrix_values, priority_vector)
                    st.subheader("λ_max, CI, and CR")
                    st.write(f"λ_max: {lambda_max}, CI: {CI}, CR: {CR}")
                    if CR < 0.1:
                        st.success("Consistency Ratio is acceptable (CR < 0.1)")
                    else:
                        st.warning("Consistency Ratio is not acceptable (CR ≥ 0.1)")

                    # Final scores for alternatives based on AHP
                    final_scores = np.zeros(num_alternatives)
                    for i in range(num_criteria):
                        alt_matrix_values = alternative_results[i].to_numpy()
                        alt_priority_vector, _, _ = ahp_method(alt_matrix_values)
                        final_scores += alt_priority_vector * priority_vector[i]

                    # Display final scores and ranking
                    visualize_results(final_scores, "AHP", num_alternatives)

            except Exception as e:
                st.error(f"Error: {e}")
