
import io
import zipfile
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Professional Pushover Dashboard", layout="wide")

# ============================================================
# Helper data structures
# ============================================================
@dataclass
class AnalysisResults:
    displacement: np.ndarray
    base_shear: np.ndarray
    story_disp: np.ndarray
    story_drift_ratio: np.ndarray
    story_shear_capacity: np.ndarray
    story_yield_disp: np.ndarray
    story_ult_disp: np.ndarray
    hinge_state: np.ndarray
    story_status: list
    target_displacement: float
    bilinear_disp: np.ndarray
    bilinear_shear: np.ndarray
    effective_yield_disp: float
    effective_yield_shear: float
    ultimate_disp: float
    ultimate_shear: float
    story_force_pattern: np.ndarray
    story_stiffness: np.ndarray
    notes: list


# ============================================================
# Core mechanics (improved smooth nonlinear surrogate model)
# ============================================================
def calc_story_stiffness(n_cols, ei_col, height, beam_factor=0.35):
    col_part = 12.0 * np.maximum(ei_col, 1e-9) * np.maximum(n_cols, 1e-9) / np.maximum(height, 1e-6) ** 3
    return col_part * (1.0 + beam_factor)


def calc_story_shear_capacity(m_col, m_beam, n_cols, n_beams, height):
    overturning_resistance = 2.0 * n_cols * np.maximum(m_col, 0.0) + 2.0 * n_beams * np.maximum(m_beam, 0.0)
    return overturning_resistance / np.maximum(height, 1e-6)


def default_force_pattern(n, pattern):
    if pattern == "Uniform":
        f = np.ones(n)
    elif pattern == "Triangular":
        f = np.arange(1, n + 1, dtype=float)
    elif pattern == "First-mode-like":
        z = np.arange(1, n + 1, dtype=float)
        f = np.sin((z / (n + 1.0)) * np.pi / 2.0)
    else:
        f = np.ones(n)
    return f / np.sum(f)


def smoothstep(a, b, x):
    """C1-smooth transition from 0 to 1."""
    if b <= a:
        return 1.0 if x >= b else 0.0
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_clamp(x, xmin, xmax):
    return np.minimum(np.maximum(x, xmin), xmax)


def smooth_story_force(story_disp, k_story, vy_story, dy_story, du_story, post_yield_ratio):
    """
    Smooth backbone replacing abrupt piecewise jumps.
    Regions are blended using smoothstep so the global base shear-displacement
    curve becomes much more continuous.

    States returned:
      0 Elastic
      1 IO / post-yield onset
      2 LS
      3 CP / degrading
      4 Failed / residual plateau
    """
    eps = 1e-9
    dy_story = max(dy_story, eps)
    du_story = max(du_story, 1.25 * dy_story)

    d1 = dy_story
    d2 = min(2.0 * dy_story, 0.55 * du_story)
    d3 = min(max(0.75 * du_story, d2 + 0.25 * dy_story), 0.92 * du_story)
    d4 = du_story

    # elastic branch
    f_el = k_story * story_disp

    # smooth post-yield hardening branch
    f_py = vy_story + post_yield_ratio * k_story * max(story_disp - d1, 0.0)
    f_py = min(f_py, 1.08 * vy_story)

    # LS plateau / mild hardening
    f_ls = min(vy_story + 0.5 * post_yield_ratio * k_story * max(story_disp - d1, 0.0), 1.10 * vy_story)

    # CP softening branch
    cp_start = max(f_ls, 0.98 * vy_story)
    if d4 > d3:
        frac = soft_clamp((story_disp - d3) / (d4 - d3), 0.0, 1.0)
    else:
        frac = 1.0
    f_cp = cp_start - frac * (cp_start - 0.85 * vy_story)

    # residual branch after failure
    f_res = 0.20 * vy_story

    # Blend branches smoothly
    w12 = smoothstep(d1 * 0.85, d1 * 1.15, story_disp)
    f_12 = (1.0 - w12) * f_el + w12 * f_py

    w23 = smoothstep(d2 * 0.90, d2 * 1.10, story_disp)
    f_23 = (1.0 - w23) * f_12 + w23 * f_ls

    w34 = smoothstep(d3 * 0.92, d3 * 1.08, story_disp)
    f_34 = (1.0 - w34) * f_23 + w34 * f_cp

    w45 = smoothstep(d4 * 0.98, d4 * 1.05, story_disp)
    force = (1.0 - w45) * f_34 + w45 * f_res
    force = max(force, 0.0)

    # Tangent-like state label for display only
    if story_disp <= 0.95 * d1:
        state = 0
    elif story_disp <= d2:
        state = 1
    elif story_disp <= d3:
        state = 2
    elif story_disp <= d4:
        state = 3
    else:
        state = 4

    return force, state


def compute_story_displacements(roof_disp, weights):
    """
    Smooth displacement allocation.
    Instead of a very sharp upper-storey concentration, this produces
    a more realistic monotonic displacement profile.
    """
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 1e-12)
    weights = weights / np.sum(weights)
    return roof_disp * weights


def moving_average(y, window=7):
    if window <= 1 or len(y) < window:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")


def run_pushover(df, pattern_name, user_pattern, n_steps, roof_disp_max, post_yield_ratio, pdelta_alpha,
                 damping_ratio, importance_factor, smooth_window=9, internal_substeps=5):
    n = len(df)
    h = df["Height_m"].to_numpy(dtype=float)
    n_cols = df["Columns"].to_numpy(dtype=float)
    n_beams = df["Beams"].to_numpy(dtype=float)
    ei_col = df["EI_column_kNm2"].to_numpy(dtype=float)
    mcol = df["Mpc_kNm"].to_numpy(dtype=float)
    mbeam = df["Mpb_kNm"].to_numpy(dtype=float)
    w = df["Weight_kN"].to_numpy(dtype=float)

    k_story = calc_story_stiffness(n_cols=n_cols, ei_col=ei_col, height=h)
    vy_story = calc_story_shear_capacity(m_col=mcol, m_beam=mbeam, n_cols=n_cols, n_beams=n_beams, height=h)
    dy_story = vy_story / np.maximum(k_story, 1e-9)
    du_story = df["Ultimate_Drift_Ratio"].to_numpy(dtype=float) * h

    if pattern_name == "User-defined":
        f = np.asarray(user_pattern, dtype=float)
        f = np.where(f < 0, 0, f)
        if np.sum(f) <= 0:
            f = np.ones(n)
        f = f / np.sum(f)
    else:
        f = default_force_pattern(n, pattern_name)

    z = np.cumsum(h)
    # Smoother displacement weights than previous version
    disp_weights = 0.45 * f + 0.55 * (z / np.max(z))
    disp_weights = np.maximum(disp_weights, 1e-12)
    disp_weights = disp_weights / np.sum(disp_weights)

    # Internal substeps make the response much smoother even when the UI uses fewer visible steps
    n_internal = max(int(n_steps) * int(max(1, internal_substeps)), int(n_steps))
    roof_disp_internal = np.linspace(0.0, roof_disp_max, n_internal)
    base_shear_internal = np.zeros_like(roof_disp_internal)
    story_disp_hist_internal = np.zeros((n_internal, n))
    hinge_state_hist_internal = np.zeros((n_internal, n), dtype=int)

    cumulative_weight = np.flip(np.cumsum(np.flip(w)))

    for j, d_roof in enumerate(roof_disp_internal):
        story_disp = compute_story_displacements(d_roof, disp_weights)
        story_force = np.zeros(n)
        states = np.zeros(n, dtype=int)

        drift_ratio = np.maximum(story_disp / np.maximum(h, 1e-6), 0.0)

        # Smooth P-delta reduction instead of abrupt clipping
        theta = pdelta_alpha * cumulative_weight * drift_ratio / np.maximum(vy_story, 1e-6)
        reduction = 1.0 - 0.55 * (1.0 - np.exp(-theta))
        reduction = np.clip(reduction, 0.45, 1.0)

        for i in range(n):
            force_i, state_i = smooth_story_force(
                story_disp=story_disp[i],
                k_story=k_story[i],
                vy_story=vy_story[i] * reduction[i],
                dy_story=dy_story[i],
                du_story=du_story[i],
                post_yield_ratio=post_yield_ratio,
            )
            story_force[i] = force_i
            states[i] = state_i

        story_disp_hist_internal[j, :] = story_disp
        hinge_state_hist_internal[j, :] = states
        base_shear_internal[j] = np.sum(story_force)

    # Optional light smoothing of plotted curve only
    base_shear_internal_smooth = moving_average(base_shear_internal, window=smooth_window)

    # Sample back to user-visible step count while preserving a smooth curve
    visible_idx = np.linspace(0, n_internal - 1, n_steps).astype(int)
    roof_disp = roof_disp_internal[visible_idx]
    base_shear = base_shear_internal_smooth[visible_idx]
    story_disp_hist = story_disp_hist_internal[visible_idx, :]
    hinge_state_hist = hinge_state_hist_internal[visible_idx, :]

    idx_peak = int(np.argmax(base_shear))
    peak_v = float(base_shear[idx_peak])
    peak_d = float(roof_disp[idx_peak])

    initial_k = np.sum(k_story * disp_weights)
    if initial_k <= 0:
        initial_k = 1.0

    dy_eff = min(peak_v / initial_k, peak_d if peak_d > 0 else roof_disp_max * 0.25)
    bilinear_disp = np.array([0.0, dy_eff, roof_disp_max])

    post_k_eff = (base_shear[-1] - peak_v) / max(roof_disp_max - max(dy_eff, peak_d), 1e-9)
    end_shear = max(peak_v + post_k_eff * (roof_disp_max - dy_eff), 0.0)
    bilinear_shear = np.array([0.0, peak_v, end_shear])

    mu = max(roof_disp_max / max(dy_eff, 1e-9), 1.0)
    c_damp = 1.0 + 2.5 * damping_ratio
    c_imp = importance_factor
    target_displacement = min(roof_disp_max, dy_eff * mu ** 0.6 * c_damp * c_imp)

    idx_target = int(np.argmin(np.abs(roof_disp - target_displacement)))
    final_story_disp = story_disp_hist[idx_target, :]
    final_drift_ratio = final_story_disp / np.maximum(h, 1e-9)
    final_hinge = hinge_state_hist[idx_target, :]

    story_status = []
    for val in final_hinge:
        story_status.append({0: "Elastic", 1: "IO", 2: "LS", 3: "CP", 4: "Failed"}.get(int(val), "?"))

    notes = [
        "This version uses a smoother nonlinear surrogate backbone, internal substepping, and smooth P-delta degradation.",
        "The smoother pushover curve is closer visually to commercial software output, but it is still not a full nonlinear FEM solver.",
        "For SAP2000/SeismoBuild-like mechanics, the next upgrade is tangent-stiffness iteration with member-by-member hinge updating.",
    ]

    return AnalysisResults(
        displacement=roof_disp,
        base_shear=base_shear,
        story_disp=final_story_disp,
        story_drift_ratio=final_drift_ratio,
        story_shear_capacity=vy_story,
        story_yield_disp=dy_story,
        story_ult_disp=du_story,
        hinge_state=final_hinge,
        story_status=story_status,
        target_displacement=target_displacement,
        bilinear_disp=bilinear_disp,
        bilinear_shear=bilinear_shear,
        effective_yield_disp=dy_eff,
        effective_yield_shear=peak_v,
        ultimate_disp=float(roof_disp[-1]),
        ultimate_shear=float(base_shear[-1]),
        story_force_pattern=f,
        story_stiffness=k_story,
        notes=notes,
    )


# ============================================================
# UI utilities
# ============================================================
def default_table(n_storey):
    return pd.DataFrame({
        "Storey": np.arange(1, n_storey + 1),
        "Height_m": [3.0] * n_storey,
        "Columns": [4] * n_storey,
        "Beams": [4] * n_storey,
        "EI_column_kNm2": [2.5e5] * n_storey,
        "Mpc_kNm": [280.0] * n_storey,
        "Mpb_kNm": [220.0] * n_storey,
        "Weight_kN": [900.0] * n_storey,
        "Ultimate_Drift_Ratio": [0.04] * n_storey,
    })


def hinge_color(state):
    return {
        0: "#dbeafe",
        1: "#bbf7d0",
        2: "#fde68a",
        3: "#fca5a5",
        4: "#991b1b",
    }.get(int(state), "#e5e7eb")


def to_excel_bytes(inputs_df, results_df, summary_df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        inputs_df.to_excel(writer, index=False, sheet_name="Inputs")
        results_df.to_excel(writer, index=False, sheet_name="Results")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
    return buffer.getvalue()


def to_zip_bytes(inputs_df, results_df, summary_df):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("inputs.csv", inputs_df.to_csv(index=False))
        zf.writestr("results.csv", results_df.to_csv(index=False))
        zf.writestr("summary.csv", summary_df.to_csv(index=False))
    return zip_buffer.getvalue()


# ============================================================
# Main app
# ============================================================
st.title("Professional Nonlinear Pushover Dashboard for RC/Steel Frames")
st.caption("Educational nonlinear pushover simulator with smoother capacity curve behavior, hinge states, bilinear idealization, target displacement, drift checks, soft-storey screening, and export tools.")

with st.sidebar:
    st.header("Model Setup")
    n_storey = st.slider("Number of Storeys", 1, 10, 5)
    pattern_name = st.selectbox("Lateral Load Pattern", ["Uniform", "Triangular", "First-mode-like", "User-defined"])
    roof_disp_max = st.number_input("Maximum Roof Displacement (m)", min_value=0.01, value=0.30, step=0.01)
    n_steps = st.slider("Pushover Steps", 30, 300, 120, 10)
    post_yield_ratio = st.slider("Post-Yield Stiffness Ratio", 0.00, 0.20, 0.03, 0.01)
    pdelta_alpha = st.slider("P-Delta Severity Factor", 0.00, 0.20, 0.04, 0.01)
    damping_ratio = st.slider("Equivalent Damping Ratio", 0.02, 0.20, 0.05, 0.01)
    importance_factor = st.slider("Importance / Amplification Factor", 0.80, 1.50, 1.00, 0.05)
    smooth_window = st.slider("Curve Smoothing Window", 1, 21, 9, 2)
    internal_substeps = st.slider("Internal Substeps per Visible Step", 1, 10, 5, 1)

st.subheader("Storey-by-Storey Frame Properties")
st.write("Enter section or capacity-related properties per storey. EI and plastic moments are interpreted here as equivalent storey-level values.")

if "table" not in st.session_state or len(st.session_state.table) != n_storey:
    st.session_state.table = default_table(n_storey)

edited_df = st.data_editor(
    st.session_state.table,
    num_rows="fixed",
    use_container_width=True,
    key="pushover_table",
    column_config={
        "Storey": st.column_config.NumberColumn(disabled=True),
        "Height_m": st.column_config.NumberColumn("Height (m)", min_value=2.0, step=0.1),
        "Columns": st.column_config.NumberColumn("No. of Columns", min_value=1, step=1),
        "Beams": st.column_config.NumberColumn("No. of Beams", min_value=1, step=1),
        "EI_column_kNm2": st.column_config.NumberColumn("Column EI (kN·m²)", min_value=1.0),
        "Mpc_kNm": st.column_config.NumberColumn("Column Plastic Moment, Mpc (kN·m)", min_value=1.0),
        "Mpb_kNm": st.column_config.NumberColumn("Beam Plastic Moment, Mpb (kN·m)", min_value=1.0),
        "Weight_kN": st.column_config.NumberColumn("Storey Seismic Weight (kN)", min_value=1.0),
        "Ultimate_Drift_Ratio": st.column_config.NumberColumn("Ultimate Drift Ratio", min_value=0.005, max_value=0.15, step=0.005),
    },
)
st.session_state.table = edited_df.copy()

user_pattern = None
if pattern_name == "User-defined":
    st.subheader("User-Defined Lateral Force Pattern")
    user_pat_df = pd.DataFrame({"Storey": np.arange(1, n_storey + 1), "Relative Force": [1.0] * n_storey})
    user_pat_df = st.data_editor(user_pat_df, num_rows="fixed", use_container_width=True, key="pattern_table")
    user_pattern = user_pat_df["Relative Force"].to_numpy(dtype=float)

run = st.button("Run Professional Pushover Analysis", type="primary")

if run:
    df = edited_df.copy()
    results = run_pushover(
        df=df,
        pattern_name=pattern_name,
        user_pattern=user_pattern,
        n_steps=n_steps,
        roof_disp_max=roof_disp_max,
        post_yield_ratio=post_yield_ratio,
        pdelta_alpha=pdelta_alpha,
        damping_ratio=damping_ratio,
        importance_factor=importance_factor,
        smooth_window=smooth_window,
        internal_substeps=internal_substeps,
    )

    results_df = pd.DataFrame({
        "Storey": df["Storey"],
        "Height_m": df["Height_m"],
        "Story_Stiffness_kN_per_m": results.story_stiffness,
        "Shear_Capacity_kN": results.story_shear_capacity,
        "Yield_Displacement_m": results.story_yield_disp,
        "Ultimate_Displacement_m": results.story_ult_disp,
        "Target_Story_Displacement_m": results.story_disp,
        "Target_Drift_Ratio": results.story_drift_ratio,
        "Hinge_State": results.story_status,
        "Force_Pattern": results.story_force_pattern,
    })

    soft_storey_limit = 0.70 * np.max(results.story_stiffness)
    results_df["Soft_Storey_Flag"] = np.where(results.story_stiffness < soft_storey_limit, "Possible", "No")

    summary_df = pd.DataFrame({
        "Metric": [
            "Peak Base Shear (kN)",
            "Effective Yield Displacement (m)",
            "Target Roof Displacement (m)",
            "Ultimate Roof Displacement Considered (m)",
            "Base Shear at Final Step (kN)",
            "Max Story Drift Ratio at Target",
            "No. of Storeys at LS or Worse",
            "No. of Storeys at CP or Failed",
        ],
        "Value": [
            results.effective_yield_shear,
            results.effective_yield_disp,
            results.target_displacement,
            results.ultimate_disp,
            results.ultimate_shear,
            np.max(results.story_drift_ratio),
            int(np.sum(np.isin(results.hinge_state, [2, 3, 4]))),
            int(np.sum(np.isin(results.hinge_state, [3, 4]))),
        ],
    })

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Base Shear", f"{results.effective_yield_shear:,.1f} kN")
    c2.metric("Target Roof Disp.", f"{results.target_displacement:.4f} m")
    c3.metric("Max Drift Ratio", f"{np.max(results.story_drift_ratio):.4f}")
    c4.metric("Critical Hinge Level", max(results.story_status, key=lambda x: ["Elastic", "IO", "LS", "CP", "Failed"].index(x)))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Capacity Curve",
        "Storey Results",
        "Hinge Map",
        "Checks & Interpretation",
        "Downloads",
    ])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(results.displacement, results.base_shear, linewidth=2.2, label="Smoothed Pushover Curve")
        ax1.plot(results.bilinear_disp, results.bilinear_shear, linestyle="--", linewidth=2, label="Idealized Bilinear")
        ax1.axvline(results.target_displacement, linestyle=":", linewidth=2, label="Target Displacement")
        ax1.set_xlabel("Roof Displacement (m)")
        ax1.set_ylabel("Base Shear (kN)")
        ax1.set_title("Nonlinear Pushover Capacity Curve")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.bar(results_df["Storey"].astype(str), results_df["Target_Drift_Ratio"])
        ax2.set_xlabel("Storey")
        ax2.set_ylabel("Drift Ratio")
        ax2.set_title("Interstorey Drift Ratio at Target Displacement")
        ax2.grid(True, axis="y")
        st.pyplot(fig2)

    with tab2:
        st.dataframe(results_df, use_container_width=True)
        st.dataframe(summary_df, use_container_width=True)

    with tab3:
        st.write("Hinge State Legend: Elastic, IO, LS, CP, Failed")
        cols = st.columns(len(results_df))
        for i, row in results_df.iterrows():
            with cols[i]:
                st.markdown(
                    f"<div style='padding:18px;border-radius:10px;background:{hinge_color(results.hinge_state[i])};text-align:center;color:#111;'>"
                    f"<b>Storey {int(row['Storey'])}</b><br>{row['Hinge_State']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    with tab4:
        st.subheader("Engineering Checks")
        drift_limit_io = 0.01
        drift_limit_ls = 0.02
        drift_limit_cp = 0.04

        interp = []
        max_drift = float(np.max(results.story_drift_ratio))
        if max_drift <= drift_limit_io:
            interp.append("Overall drift demand is within a typical Immediate Occupancy range.")
        elif max_drift <= drift_limit_ls:
            interp.append("Overall drift demand has entered a typical Life Safety range.")
        elif max_drift <= drift_limit_cp:
            interp.append("Overall drift demand is approaching or within a Collapse Prevention range.")
        else:
            interp.append("Overall drift demand exceeds a common Collapse Prevention screening limit.")

        softies = results_df.loc[results_df["Soft_Storey_Flag"] == "Possible", "Storey"].tolist()
        if softies:
            interp.append(f"Possible soft-storey behavior detected at storey/storeys: {', '.join(map(str, softies))}.")
        else:
            interp.append("No obvious soft-storey flag based on relative stiffness screening.")

        severe = results_df.loc[results_df["Hinge_State"].isin(["CP", "Failed"]), "Storey"].tolist()
        if severe:
            interp.append(f"Storey/storeys with CP or failed hinge state at target displacement: {', '.join(map(str, severe))}.")
        else:
            interp.append("No storey reached CP or failed state at the selected target displacement.")

        for item in interp:
            st.write(f"- {item}")
        st.info("Interpretation is screening-level only. Use a detailed nonlinear model for design decisions.")
        for note in results.notes:
            st.caption(note)

    with tab5:
        excel_bytes = to_excel_bytes(edited_df, results_df, summary_df)
        zip_bytes = to_zip_bytes(edited_df, results_df, summary_df)
        st.download_button(
            "Download Excel Results",
            data=excel_bytes,
            file_name="pushover_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download CSV ZIP Bundle",
            data=zip_bytes,
            file_name="pushover_results_bundle.zip",
            mime="application/zip",
        )

st.markdown("---")
st.write("Suggested next upgrade: tangent-stiffness iteration + element-by-element hinges + FEMA/ASCE performance point option + OpenSees comparison mode.")
