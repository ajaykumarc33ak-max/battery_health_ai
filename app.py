"""
Battery Health AI Dashboard
Research prototype for battery SOH estimation, uncertainty estimation,
and unusual-pattern detection. Not a certified battery safety tool.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Battery Health AI",
    page_icon="🔋",
    layout="wide"
)

FEATURES = ["cycle", "voltage", "current", "temperature"]


def make_demo_data(seed=42, cells=8, cycles_per_cell=150):
    """Create clearly labelled synthetic data for testing the application."""
    rng = np.random.default_rng(seed)
    frames = []

    for cell in range(cells):
        cycles = np.arange(1, cycles_per_cell + 1)
        initial_capacity = rng.uniform(1.9, 2.1)
        degradation_rate = rng.uniform(0.0007, 0.0015)
        capacity = initial_capacity * (
            1 - degradation_rate * cycles - 0.0000015 * cycles**1.35
        )
        capacity += rng.normal(0, 0.008, size=len(cycles))
        capacity = np.clip(capacity, initial_capacity * 0.55, initial_capacity)

        temperature = 25 + 5 * np.sin(cycles / 18) + rng.normal(0, 1.5, len(cycles))
        current = 1.0 + 0.15 * np.sin(cycles / 11) + rng.normal(0, 0.06, len(cycles))
        voltage = 4.15 - 0.003 * cycles / cycles_per_cell + rng.normal(0, 0.025, len(cycles))
        voltage = np.clip(voltage, 3.0, 4.25)

        frame = pd.DataFrame({
            "cell_id": f"DEMO_CELL_{cell + 1}",
            "cycle": cycles,
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "capacity_ah": capacity,
        })
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["soh"] = data.groupby("cell_id")["capacity_ah"].transform(
        lambda s: 100 * s / s.iloc[0]
    )
    return data


def prepare_uploaded_data(uploaded):
    """Read a CSV with cycle, voltage, current, temperature, capacity_ah columns."""
    data = pd.read_csv(uploaded)
    data.columns = [str(c).strip().lower() for c in data.columns]

    # Accept common alternative column names.
    aliases = {
        "cycle_index": "cycle",
        "cycle_number": "cycle",
        "voltage_measured": "voltage",
        "voltage_v": "voltage",
        "current_measured": "current",
        "current_a": "current",
        "temperature_measured": "temperature",
        "temperature_c": "temperature",
        "capacity": "capacity_ah",
        "discharge_capacity": "capacity_ah",
        "discharge_capacity_ah": "capacity_ah",
    }
    data = data.rename(columns={k: v for k, v in aliases.items() if k in data.columns})

    required = ["cycle", "voltage", "current", "temperature", "capacity_ah"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing) +
            ". Use the template CSV available in the app."
        )

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=required).copy()
    data = data[data["capacity_ah"] > 0]
    if len(data) < 30:
        raise ValueError("Please upload at least 30 valid rows for model training.")

    if "cell_id" not in data.columns:
        data["cell_id"] = "UPLOADED_CELL"

    data["soh"] = data.groupby("cell_id")["capacity_ah"].transform(
        lambda s: 100 * s / s.iloc[0] if len(s) else s
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURES + ["soh", "capacity_ah"]
    )
    return data


def fit_models(data):
    X = data[FEATURES].astype(float)
    y = data["soh"].astype(float).clip(0, 120)

    if len(data) < 30:
        raise ValueError("Not enough valid rows after cleaning; need at least 30.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, test_predictions)))
    r2 = r2_score(y_test, test_predictions) if len(y_test) > 1 else float("nan")

    anomaly_model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    anomaly_model.fit(X_train)

    return model, anomaly_model, {
        "mae": mae, "rmse": rmse, "r2": r2,
        "X_test": X_test, "y_test": y_test,
        "test_predictions": test_predictions
    }


def add_predictions(data, model, anomaly_model):
    result = data.copy()
    X = result[FEATURES].astype(float)
    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
    result["predicted_soh"] = model.predict(X)
    result["uncertainty_low"] = np.percentile(tree_predictions, 10, axis=0)
    result["uncertainty_high"] = np.percentile(tree_predictions, 90, axis=0)
    result["anomaly"] = anomaly_model.predict(X) == -1
    result["anomaly_score"] = anomaly_model.decision_function(X)
    return result


st.title("🔋 Battery Health AI")
st.caption(
    "SOH estimation • Prediction uncertainty • Unusual-pattern detection • "
    "Research prototype only — not a certified safety assessment"
)

with st.sidebar:
    st.header("Data source")
    source = st.radio(
        "Choose dataset",
        ["Synthetic demo data", "Upload my CSV"],
        help="Synthetic data is simulated for demonstrating the application."
    )
    uploaded_file = None
    if source == "Upload my CSV":
        uploaded_file = st.file_uploader("Upload battery cycling CSV", type=["csv"])

    st.divider()
    st.subheader("CSV format")
    st.write("Required columns:")
    st.code("cycle, voltage, current, temperature, capacity_ah", language="text")
    st.caption(
        "capacity_ah should be the measured discharge capacity in amp-hours. "
        "Each cell should have its own rows in cycle order."
    )

    template = pd.DataFrame({
        "cell_id": ["CELL_1"] * 3,
        "cycle": [1, 2, 3],
        "voltage": [4.10, 4.08, 4.06],
        "current": [1.0, 1.0, 1.0],
        "temperature": [25.0, 25.5, 26.0],
        "capacity_ah": [2.00, 1.99, 1.98],
    })
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="battery_template.csv",
        mime="text/csv"
    )

try:
    if source == "Upload my CSV":
        if uploaded_file is None:
            st.info("Upload a CSV to start, or choose Synthetic demo data to explore the dashboard.")
            st.stop()
        data = prepare_uploaded_data(uploaded_file)
        data_source_label = "Uploaded dataset"
    else:
        data = make_demo_data()
        data_source_label = "Synthetic demo data (simulated; not real battery measurements)"

    model, anomaly_model, metrics = fit_models(data)
    results = add_predictions(data, model, anomaly_model)

except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.stop()

st.info(f"Data source: **{data_source_label}**")

latest = results.sort_values(["cell_id", "cycle"]).groupby("cell_id").tail(1)
mean_soh = float(latest["predicted_soh"].mean())
anomaly_count = int(results["anomaly"].sum())
avg_uncertainty = float(
    (results["uncertainty_high"] - results["uncertainty_low"]).mean()
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average latest predicted SOH", f"{mean_soh:.1f}%")
c2.metric("Flagged unusual rows", f"{anomaly_count:,}")
c3.metric("Average 80% tree range width", f"{avg_uncertainty:.2f} percentage points")
c4.metric("Validation MAE", f"{metrics['mae']:.2f} percentage points")

tab1, tab2, tab3, tab4 = st.tabs([
    "Health trends", "Unusual behavior", "Model evaluation", "Data preview"
])

with tab1:
    st.subheader("Battery health over charge/discharge cycles")
    selected_cell = st.selectbox("Choose cell", sorted(results["cell_id"].astype(str).unique()))
    cell = results[results["cell_id"].astype(str) == selected_cell].sort_values("cycle")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(cell["cycle"], cell["soh"], label="Capacity-derived SOH", linewidth=2)
    ax.plot(cell["cycle"], cell["predicted_soh"], label="Model-predicted SOH", linewidth=2)
    ax.fill_between(
        cell["cycle"].to_numpy(),
        cell["uncertainty_low"].to_numpy(),
        cell["uncertainty_high"].to_numpy(),
        alpha=0.2,
        label="10th–90th percentile across trees"
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH (%)")
    ax.set_title(f"SOH estimate — {selected_cell}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The shaded band measures spread among Random Forest trees; it is a rough "
        "uncertainty indicator, not a statistically guaranteed confidence interval."
    )

    # Simple descriptive trend projection, explicitly not a validated RUL model.
    if len(cell) >= 10:
        tail = cell.tail(max(10, len(cell) // 4))
        slope, intercept = np.polyfit(tail["cycle"], tail["soh"], 1)
        if slope < -1e-8:
            target_soh = 80.0
            projected_cycle = (target_soh - intercept) / slope
            cycles_left = max(0.0, projected_cycle - float(cell["cycle"].max()))
            st.metric("Rough cycles to 80% SOH (trend extrapolation)", f"{cycles_left:.0f}")
            st.caption(
                "This is a simple extrapolation from the recent observed SOH trend, "
                "not a validated remaining-useful-life prediction. It may be unreliable "
                "outside the observed operating conditions."
            )
        else:
            st.write("Recent trend is not decreasing enough to estimate cycles to 80% SOH.")

with tab2:
    st.subheader("Rows flagged as unusual")
    st.write(
        "Isolation Forest flags measurements that differ from the model's training "
        "patterns. A flag is not proof of a dangerous battery condition."
    )
    flagged = results[results["anomaly"]].sort_values("anomaly_score")
    if flagged.empty:
        st.success("No unusual rows were flagged in this dataset.")
    else:
        st.warning(f"{len(flagged)} rows were flagged for review.")
        st.dataframe(
            flagged[["cell_id", "cycle", "voltage", "current", "temperature",
                     "capacity_ah", "soh", "anomaly_score"]].head(300),
            use_container_width=True
        )

with tab3:
    st.subheader("Held-out test performance")
    st.write("A random 80/20 row split is used for this starter prototype.")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['mae']:.3f}")
    m2.metric("RMSE", f"{metrics['rmse']:.3f}")
    m3.metric("R²", f"{metrics['r2']:.3f}" if np.isfinite(metrics["r2"]) else "N/A")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(metrics["y_test"], metrics["test_predictions"], alpha=0.55)
    low = min(float(metrics["y_test"].min()), float(np.min(metrics["test_predictions"])))
    high = max(float(metrics["y_test"].max()), float(np.max(metrics["test_predictions"])))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Measured SOH (%)")
    ax.set_ylabel("Predicted SOH (%)")
    ax.set_title("Measured vs predicted SOH on held-out rows")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        "For a rigorous result, split by battery cell (not random rows) and evaluate "
        "on cells the model never saw. Random row splitting can overestimate performance "
        "because nearby cycles from the same cell are similar."
    )

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    st.write("Random Forest feature importance")
    st.bar_chart(importance.set_index("Feature"))

with tab4:
    st.subheader("Cleaned dataset with predictions")
    st.dataframe(results, use_container_width=True, height=420)
    st.download_button(
        "Download predictions CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="battery_health_predictions.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Limitations: synthetic demo measurements are simulated. Uploaded data must be "
    "validated and converted to the documented column format. Software-only outputs "
    "cannot certify battery safety. Do not use this prototype to make real-world safety "
    "or charging decisions."
)
