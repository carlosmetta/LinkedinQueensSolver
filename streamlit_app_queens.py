from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from solver_functions import (
    find_best_grid,
    find_best_clusters,
    plot_grid,
    plot_grid_sol,
    solver,
)

EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLES = {
    "Example 1": ("queens_7x7.png", 7),
    "Example 2": ("queens_8x8.png", 8),
    "Example 3": ("queens_9x9_a.jpg", 9),
    "Example 4": ("queens_9x9_b.jpg", 9),
    "Example 5": ("queens_10x10.png", 10),
}

st.set_page_config(page_title="Queens Puzzle Solver", layout="centered")
st.title("♟️ Queens Puzzle Solver from Image")
st.caption("Reads a LinkedIn Queens board from an image, clusters its color regions, "
           "and solves it as a binary integer program.")

# Session state
for key in ("df", "grid_shape", "n_clusters", "image_rgb", "solution_grid", "source", "mode"):
    st.session_state.setdefault(key, None)


def decode(data: bytes):
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# Input: pick an example from the gallery, or upload a cropped screenshot
st.session_state.setdefault("example", list(EXAMPLES)[0])
tab_examples, tab_upload = st.tabs(["🖼️ Choose an example", "📤 Upload your own"])

with tab_examples:
    st.write("Click **Use** under any puzzle to load it.")
    cols = st.columns(len(EXAMPLES))
    for col, (name, (filename, n)) in zip(cols, EXAMPLES.items()):
        with col:
            st.image(str(EXAMPLES_DIR / filename), caption=f"{n} x {n}", use_container_width=True)
            if st.button("Use", key=f"use_{filename}", use_container_width=True):
                st.session_state.example = name
                st.session_state.mode = "example"

with tab_upload:
    st.markdown(
        "**Crop the screenshot tightly to the board.** The solver splits the image into "
        "an equal n x n grid, so extra margins, titles, or borders will misalign every cell.\n"
        "- Include only the colored squares, edge to edge\n"
        "- Keep the board square and unrotated\n"
        "- Set the grid size below to match the board"
    )
    upload_size = st.slider("Grid size (n x n)", min_value=4, max_value=12, value=9)
    uploaded = st.file_uploader("Upload a cropped board image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.session_state.mode = "upload"

image_rgb, source = None, None
if st.session_state.get("mode") == "upload" and uploaded:
    grid_size = upload_size
    image_rgb = decode(uploaded.getvalue())
    source = f"upload:{uploaded.name}:{uploaded.size}"
    if image_rgb is None:
        st.error("Could not read that image. Try a PNG or JPG screenshot of the board.")
    else:
        h, w = image_rgb.shape[:2]
        if abs(h - w) / max(h, w) > 0.05:
            st.warning(f"The image is {w} x {h} px, not square. Crop it to just the board "
                       "or the grid will be misread.")
else:
    name = st.session_state.example
    filename, grid_size = EXAMPLES[name]
    image_rgb = decode((EXAMPLES_DIR / filename).read_bytes())
    source = f"example:{filename}"

st.divider()
st.subheader(f"Puzzle: {'your upload' if source and source.startswith('upload') else st.session_state.example} "
             f"({grid_size} x {grid_size})")

# Reset results when the puzzle or grid size changes
source_key = f"{source}:{grid_size}"
if source_key != st.session_state.source:
    st.session_state.update(df=None, grid_shape=None, n_clusters=None,
                            solution_grid=None, source=source_key)
st.session_state.image_rgb = image_rgb

if image_rgb is not None:
    st.image(image_rgb, width=320)

    if st.button("📊 Detect grid"):
        with st.spinner("Detecting grid and clustering colors..."):
            colors, positions, grid_shape = find_best_grid(
                image_rgb, fixed_clusters=grid_size, grid_range=(grid_size, grid_size)
            )
            df, n_clusters, _ = find_best_clusters(
                colors, positions, cluster_range=(grid_size, grid_size)
            )
        st.session_state.update(df=df, grid_shape=grid_shape, n_clusters=n_clusters)

    if st.session_state.df is not None:
        rows, cols = st.session_state.grid_shape
        st.success(f"Detected {st.session_state.n_clusters} regions on a {rows} x {cols} grid")
        st.subheader("🎨 Detected regions")
        st.pyplot(plot_grid(st.session_state.df, st.session_state.grid_shape,
                            st.session_state.n_clusters))

        if st.button("♟️ Solve"):
            with st.spinner("Solving the integer program..."):
                st.session_state.solution_grid = solver(rows, st.session_state.df)
            if st.session_state.solution_grid is None:
                st.error("No solution found. Check that the grid size matches the board.")

    if st.session_state.solution_grid is not None:
        st.subheader("✅ Solution")
        st.pyplot(plot_grid_sol(st.session_state.df, st.session_state.grid_shape,
                                st.session_state.n_clusters, st.session_state.solution_grid))
