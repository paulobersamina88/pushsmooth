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
# Core mechanics (educational nonlinear surrogate model)
# ============================================================
def calc_story_stiffness(n_cols, ei_col, height, beam_factor=0.35):
    """Story lateral stiffness from cantilever columns plus beam participation."""
    col_part = 12.0 * np.maximum(ei_col, 1e-9) * np.maximum(n_cols, 1e-9) / np.maximum(height, 1e-6) ** 3
    return col_part * (1.0 + beam_factor)


def calc_story_shear_capacity(m_col, m_beam, n_cols, n_beams, height):
    """Approximate storey shear yield capacity from plastic moments."""
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


def piecewise_story_force(story_disp, k_story, vy_story, dy_story, du_story, post_yield_ratio):
    """Elastic-perfectly plastic with mild hardening/softening until ultimate.
    Returns resisting force and hinge state index.
      0 elastic
      1 IO
      2 LS
      3 CP
      4 failed / residual
    """
    if story_disp <= dy_story:
        return k_story * story_disp, 0

    if story_disp <= 2.0 * dy_story:
        force = vy_story + post_yield_ratio * k_story * (story_disp - dy_story)
        return min(force, 1.05 * vy_story), 1

    ls_limit = min(0.75 * du_story, 4.0 * dy_story)
    if story_disp <= ls_limit:
        force = vy_story + post_yield_ratio * k_story * (story_disp - dy_story)
        return min(force, 1.10 * vy_story), 2

    if story_disp <= du_story:
        # degrade gently toward 85% of peak
        peak = min(vy_story + post_yield_ratio * k_story * (ls_limit - dy_story), 1.10 * vy_story)
        frac = (story_disp - ls_limit) / max(du_story - ls_limit, 1e-9)
        force = peak - frac * (peak - 0.85 * vy_story)
        return force, 3

    return 0.20 * vy_story, 4


def compute_story_displacements(roof_disp, weights):
    """Allocate roof displacement to stories; larger upper-storey participation by default."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return roof_disp * weights


def run_pushover(df, pattern_name, user_pattern, n_steps, roof_disp_max, post_yield_ratio, pdelta_alpha,
                 damping_ratio, importance_factor):
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

    # Story displacement participation weights using cumulative elevation and lateral force pattern
    z = np.cumsum(h)
    disp_weights = f * (1.0 + z / np.max(z))
    disp_weights = disp_weights / np.sum(disp_weights)

    roof_disp = np.linspace(0.0, roof_disp_max, n_steps)
    base_shear = np.zeros_like(roof_disp)
    story_disp_hist = np.zeros((n_steps, n))
    hinge_state_hist = np.zeros((n_steps, n), dtype=int)

    for j, d_roof in enumerate(roof_disp):
        story_disp = compute_story_displacements(d_roof, disp_weights)
        story_force = np.zeros(n)
        states = np.zeros(n, dtype=int)

        # Simple P-delta degradation proportional to cumulative gravity load and drift
        cumulative_weight = np.flip(np.cumsum(np.flip(w)))
        theta = pdelta_alpha * cumulative_weight * np.maximum(story_disp / np.maximum(h, 1e-6), 0) / np.maximum(vy_story, 1e-6)
        reduction = np.clip(1.0 - theta, 0.35, 1.0)

        for i in range(n):
            force_i, state_i = piecewise_story_force(
                story_disp=story_disp[i],
                k_story=k_story[i],
                vy_story=vy_story[i] * reduction[i],
                dy_story=dy_story[i],
                du_story=du_story[i],
                post_yield_ratio=post_yield_ratio,
            )
            story_force[i] = force_i
            states[i] = state_i

        story_disp_hist[j, :] = story_disp
        hinge_state_hist[j, :] = states
        base_shear[j] = np.sum(story_force)

    # Peak and target displacement
    idx_peak = int(np.argmax(base_shear))
    peak_v = float(base_shear[idx_peak])
    peak_d = float(roof_disp[idx_peak])

    # Effective bilinear idealization (equal-area style surrogate)
    initial_k = np.sum(k_story * disp_weights)
    if initial_k <= 0:
        initial_k = 1.0
    dy_eff = min(peak_v / initial_k, peak_d if peak_d > 0 else roof_disp_max * 0.25)
    bilinear_disp = np.array([0.0, dy_eff, roof_disp_max])
    post_k_eff = (base_shear[-1] - peak_v) / max(roof_disp_max - max(dy_eff, peak_d), 1e-9)
    end_shear = max(peak_v + post_k_eff * (roof_disp_max - dy_eff), 0.0)
    bilinear_shear = np.array([0.0, peak_v, end_shear])

    # Target displacement surrogate using damping and importance modifiers
    mu = max(roof_disp_max / max(dy_eff, 1e-9), 1.0)
    c_damp = 1.0 + 2.5 * damping_ratio
    c_imp = importance_factor
    target_displacement = min(roof_disp_max, dy_eff * mu ** 0.6 * c_damp * c_imp)

    # Final step results at target displacement
    idx_target = int(np.argmin(np.abs(roof_disp - target_displacement)))
    final_story_disp = story_disp_hist[idx_target, :]
    final_drift_ratio = final_story_disp / np.maximum(h, 1e-9)
    final_hinge = hinge_state_hist[idx_target, :]

    story_status = []
    for val in final_hinge:
        story_status.append({0: "Elastic", 1: "IO", 2: "LS", 3: "CP", 4: "Failed"}.get(int(val), "?"))

    notes = [
        "This app uses a simplified nonlinear surrogate suitable for teaching and early conceptual studies.",
        "Results are not a substitute for full fiber- or hinge-based nonlinear FEM in ETABS, SAP2000, OpenSees, Ruaumoko, or SeismoStruct.",
        "Target displacement and bilinear idealization are approximate educational implementations.",
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
st.caption("Educational nonlinear pushover simulator for up to 10 storeys with hinge states, bilinear idealization, target displacement, drift checks, soft-storey screening, and export tools.")

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

st.subheader("Storey-by-Storey Frame Properties")
st.write("Enter section or capacity-related properties per storey. You may interpret EI and plastic moments as equivalent values from your chosen beam and column sections.")

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
    c4.metric("Critical Hinge Level", max(results.story_status, key=lambda x: ["Elastic","IO","LS","CP","Failed"].index(x)))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Capacity Curve",
        "Storey Results",
        "Hinge Map",
        "Checks & Interpretation",
        "Downloads",
    ])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(results.displacement, results.base_shear, linewidth=2, label="Pushover Curve")
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
st.write("Suggested next upgrade for your class: section library + moment-curvature generator + FEMA/ASCE performance point option + portal frame sketch + OpenSees comparison mode.")
