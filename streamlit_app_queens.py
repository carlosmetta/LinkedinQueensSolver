import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

from solver_functions import (
    find_best_grid,
    find_best_clusters,
    plot_grid,
    plot_grid_sol,
    solver
)

# Streamlit Page Config
st.set_page_config(page_title="Clustered Grid Solver", layout="centered")
st.title("♟️ Clustered Grid Solver from Image")

# Sidebar Inputs
st.sidebar.header("Settings")
grid_size = st.slider("Grid Size (n x n)", min_value=2, max_value=20, value=9)
# Session State Initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'grid_shape' not in st.session_state:
    st.session_state.grid_shape = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = None
if 'image_rgb' not in st.session_state:
    st.session_state.image_rgb = None
if 'solution_grid' not in st.session_state:
    st.session_state.solution_grid = None

# File Upload
uploaded_file = st.file_uploader("Upload a Grid Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and store image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.image(st.session_state.image_rgb, caption="Uploaded Image", use_column_width=True)

    # Button 1: Generate Grid
    if st.button("📊 Generate Grid"):
        with st.spinner("Detecting grid and clustering colors..."):
            num_clusters = grid_size  # same as grid size
            colors, positions, grid_shape = find_best_grid(
                st.session_state.image_rgb,
                fixed_clusters=num_clusters,
                grid_range=(grid_size, grid_size)
            )

            df, n_clusters, silhouette = find_best_clusters(
                colors,
                positions,
                cluster_range=(num_clusters, num_clusters)
            )

            # Store results
            st.session_state.df = df
            st.session_state.grid_shape = grid_shape
            st.session_state.n_clusters = n_clusters

        st.success(f"Detected {n_clusters} clusters on a {grid_shape[0]}x{grid_shape[1]} grid")

        st.subheader("🎨 Clustered Grid")
        fig1 = plot_grid(df, grid_shape, n_clusters)
        st.pyplot(fig1)

    # Button 2: Solve
    if st.session_state.df is not None and st.button("♟️ Solve"):
        with st.spinner("Solving constraint optimization..."):
            solution_grid = solver(st.session_state.grid_shape[0], st.session_state.df)

        if solution_grid is None:
            st.error("❌ No solution found. The ILP model is infeasible with current grid and clusters.")
        else:
            st.session_state.solution_grid = solution_grid
            st.subheader("✅ Solved Grid with Queens")
            fig2 = plot_grid_sol(
                st.session_state.df,
                st.session_state.grid_shape,
                st.session_state.n_clusters,
                solution_grid
            )
            st.pyplot(fig2)
