"""
Battery Health AI — college project prototype
Manual CSV analysis + optional live telemetry API connection.
Live mode requires a battery/BMS gateway that exposes an HTTP JSON endpoint.
This software is not a certified battery safety tool.
"""
import io
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Battery Health AI", page_icon="🔋", layout="wide")

FEATURES = ["cycle", "voltage", "current", "temperature"]
REQUIRED = FEATURES + ["capacity_ah"]

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

def make_demo_data(seed=42, cells=8, cycles_per_cell=150):
    rng = np.random.default_rng(seed)
    frames = []
    for cell in range(cells):
        cycles = np.arange(1, cycles_per_cell + 1)
        initial_capacity = rng.uniform(1.9, 2.1)
        degradation_rate = rng.uniform(0.0007, 0.0015)
        capacity = initial_capacity * (1 - degradation_rate * cycles - 0.0000015 * cycles**1.35)
        capacity += rng.normal(0, 0.008, size=len(cycles))
        capacity = np.clip(capacity, initial_capacity * 0.55, initial_capacity)
        temperature = 25 + 5 * np.sin(cycles / 18) + rng.normal(0, 1.5, len(cycles))
        current = 1.0 + 0.15 * np.sin(cycles / 11) + rng.normal(0, 0.06, len(cycles))
        voltage = np.clip(4.15 - 0.003 * cycles / cycles_per_cell + rng.normal(0, 0.025, len(cycles)), 3.0, 4.25)
        frames.append(pd.DataFrame({
            "cell_id": f"DEMO_CELL_{cell+1}", "cycle": cycles, "voltage": voltage,
            "current": current, "temperature": temperature, "capacity_ah": capacity
        }))
    data = pd.concat(frames, ignore_index=True)
    data["soh"] = data.groupby("cell_id")["capacity_ah"].transform(lambda s: 100 * s / s.iloc[0])
    return data

def clean_data(df):
    data = df.copy()
    data.columns = [str(c).strip().lower() for c in data.columns]
    aliases = {
        "cycle_index":"cycle", "cycle_number":"cycle", "voltage_measured":"voltage",
        "voltage_v":"voltage", "current_measured":"current", "current_a":"current",
        "temperature_measured":"temperature", "temperature_c":"temperature",
        "capacity":"capacity_ah", "discharge_capacity":"capacity_ah",
        "discharge_capacity_ah":"capacity_ah"
    }
    data = data.rename(columns={k:v for k,v in aliases.items() if k in data.columns})
    missing = [c for c in REQUIRED if c not in data.columns]
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(missing))
    if "cell_id" not in data.columns:
        data["cell_id"] = "CELL_1"
    for c in REQUIRED:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED)
    data = data[data["capacity_ah"] > 0].copy()
    data["cell_id"] = data["cell_id"].fillna("CELL_1").astype(str)
    data = data.sort_values(["cell_id", "cycle"]).reset_index(drop=True)
    data["soh"] = data.groupby("cell_id")["capacity_ah"].transform(
        lambda s: 100 * s / s.iloc[0] if len(s) else s
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["soh"])
    if len(data) < 30:
        raise ValueError(f"Only {len(data)} valid rows found. At least 30 are required.")
    return data

def fit_models(data):
    X = data[FEATURES].astype(float)
    y = data["soh"].astype(float).clip(0, 120)
    # Avoid leakage where possible: keep entire cells out of training.
    unique_cells = data["cell_id"].nunique()
    if unique_cells >= 2:
        cells = data["cell_id"].drop_duplicates().to_numpy()
        rng = np.random.default_rng(42)
        rng.shuffle(cells)
        n_test = max(1, int(round(len(cells) * 0.2)))
        test_cells = set(cells[:n_test])
        test_mask = data["cell_id"].isin(test_cells).to_numpy()
        if test_mask.sum() == 0 or (~test_mask).sum() < 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
            y_train, y_test = y.loc[~test_mask], y.loc[test_mask]
    else:
        ordered = data.sort_values("cycle").index
        split = max(1, int(len(ordered) * 0.8))
        train_idx, test_idx = ordered[:split], ordered[split:]
        if len(test_idx) < 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx]
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    anomaly = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)
    anomaly.fit(X_train)
    return model, anomaly, {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": r2_score(y_test, preds) if len(y_test) > 1 else float("nan"),
        "X_test": X_test, "y_test": y_test, "preds": preds
    }

def add_predictions(data, model, anomaly_model):
    result = data.copy()
    X = result[FEATURES].astype(float)
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    result["predicted_soh"] = np.clip(model.predict(X), 0, 120)
    result["uncertainty_low"] = np.percentile(tree_preds, 10, axis=0)
    result["uncertainty_high"] = np.percentile(tree_preds, 90, axis=0)
    result["anomaly"] = anomaly_model.predict(X) == -1
    result["anomaly_score"] = anomaly_model.decision_function(X)
    return result

def get_api_payload(url, token, timeout):
    headers = {"Accept": "application/json"}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], (dict, list)):
        payload = payload["data"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload or not all(isinstance(x, dict) for x in payload):
        raise ValueError("API must return a JSON object or a list of JSON objects.")
    return pd.DataFrame(payload)

st.title("🔋 Battery Health AI")
st.caption("Battery data input • SOH estimation • unusual-pattern detection • research prototype")
st.info("Safety note: this is a college-project prototype, not a certified battery safety system. Do not use it to make charging or driving safety decisions.")

with st.sidebar:
    st.header("1 · Choose input")
    input_mode = st.radio("How will you provide battery data?", [
        "Demo data", "Upload CSV", "Connect to battery API"
    ])
    st.divider()
    st.subheader("What the app needs")
    st.write("Cycle number, voltage (V), current (A), temperature (°C), and measured capacity (Ah).")
    st.caption("A website cannot read an EV battery by itself. Live mode needs a BMS/IoT gateway or service that exposes an HTTP JSON endpoint.")

st.markdown("### 2 · Provide battery data")
data = None
source_label = ""

if input_mode == "Demo data":
    st.success("Demo mode selected — simulated measurements will be used.")
    st.write("Use this mode to learn the dashboard and demonstrate the workflow. These are not readings from a real battery.")
    if st.button("Load demo battery data", type="primary"):
        st.session_state["battery_data"] = make_demo_data()
        st.session_state["battery_source"] = "Simulated demo data"
    if "battery_data" in st.session_state:
        data = st.session_state["battery_data"].copy()
        source_label = st.session_state.get("battery_source", "Simulated demo data")

elif input_mode == "Upload CSV":
    st.write("Upload a battery cycling dataset. The file must contain at least 30 valid rows.")
    uploaded = st.file_uploader("Choose battery CSV file", type=["csv"], key="csv_upload")
    template = pd.DataFrame({
        "cell_id": ["CELL_1"] * 30,
        "cycle": list(range(1, 31)),
        "voltage": np.linspace(4.15, 4.00, 30).round(3),
        "current": [1.0] * 30,
        "temperature": np.linspace(25, 29, 30).round(1),
        "capacity_ah": np.linspace(2.0, 1.92, 30).round(3)
    })
    st.download_button("Download 30-row CSV example", template.to_csv(index=False).encode("utf-8"),
                       file_name="battery_csv_example.csv", mime="text/csv")
    if uploaded is not None:
        try:
            data = clean_data(pd.read_csv(uploaded))
            source_label = f"Uploaded file: {uploaded.name}"
            st.success(f"Loaded {len(data)} valid rows from {uploaded.name}.")
        except Exception as e:
            st.error(str(e))

else:
    st.write("Connect to a telemetry/history API provided by your battery management system (BMS), IoT gateway, or simulator.")
    with st.expander("Expected API response format", expanded=True):
        st.code(json.dumps({
            "cell_id": "EV_BATTERY_01", "cycle": 125, "voltage": 3.92,
            "current": 12.4, "temperature": 31.5, "capacity_ah": 42.7
        }, indent=2), language="json")
        st.caption("The endpoint must return a list of at least 30 historical records with these fields (not just one latest reading), because this prototype trains its model from the supplied data. Capacity must be measured capacity in Ah.")
    with st.form("api_connection_form"):
        api_url = st.text_input("Telemetry API URL", placeholder="https://your-bms-gateway.example/api/battery/history")
        api_token = st.text_input("Bearer token (optional)", type="password", help="Used only for this request; don't paste passwords or private keys here.")
        timeout = st.slider("Request timeout (seconds)", 3, 30, 10)
        connect = st.form_submit_button("Connect and fetch readings", type="primary")
    if connect:
        if not api_url.lower().startswith("https://"):
            st.error("For safety, enter an HTTPS endpoint. A local development gateway may require a separately configured secure connection.")
        else:
            try:
                raw = get_api_payload(api_url, api_token, timeout)
                data = clean_data(raw)
                source_label = "Live API data"
                st.session_state["battery_data"] = data
                st.session_state["battery_source"] = source_label
                st.success(f"Connected successfully. Received {len(data)} valid records.")
            except requests.RequestException as e:
                st.error(f"Could not connect to the API: {e}")
            except (ValueError, KeyError) as e:
                st.error(f"Could not use the API response: {e}")
    if data is None and st.session_state.get("battery_source") == "Live API data":
        data = st.session_state.get("battery_data")
        source_label = st.session_state.get("battery_source", "")

st.divider()
st.markdown("### 3 · Analyze battery health")

if data is None:
    st.warning("First choose Demo data and click **Load demo battery data**, upload a CSV, or connect to a compatible API.")
    st.markdown("""
    **Simple workflow**
    1. Choose where the battery data comes from.
    2. Provide measurements or connect to a supported data gateway.
    3. Click/allow analysis to estimate SOH and inspect unusual readings.
    """)
    st.stop()

try:
    data = clean_data(data)
    model, anomaly_model, metrics = fit_models(data)
    results = add_predictions(data, model, anomaly_model)
except Exception as e:
    st.error(f"Analysis could not be completed: {e}")
    st.stop()

st.caption(f"Data source: **{source_label}** · {len(results):,} valid rows · {results['cell_id'].nunique()} battery/cell ID(s)")
latest = results.sort_values(["cell_id", "cycle"]).groupby("cell_id").tail(1)
mean_soh = float(latest["predicted_soh"].mean())
anomaly_count = int(results["anomaly"].sum())
avg_range = float((results["uncertainty_high"] - results["uncertainty_low"]).mean())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Average latest estimated SOH", f"{mean_soh:.1f}%")
m2.metric("Unusual rows flagged", f"{anomaly_count:,}")
m3.metric("Average tree prediction range", f"{avg_range:.2f} pp")
m4.metric("Test MAE", f"{metrics['mae']:.2f} pp")

if mean_soh >= 90:
    st.success("Estimated health is relatively high in this dataset. This is not a safety certification.")
elif mean_soh >= 80:
    st.warning("Estimated health is lower than a new battery. Review the data and battery manufacturer's guidance.")
else:
    st.warning("Estimated health is low in this dataset. Have the battery checked by a qualified technician; this app cannot diagnose safety.")

tab1, tab2, tab3, tab4 = st.tabs(["Health trends", "Unusual readings", "Model evaluation", "Data & export"])
with tab1:
    cell_id = st.selectbox("Select battery/cell", sorted(results["cell_id"].astype(str).unique()))
    cell = results[results["cell_id"].astype(str) == cell_id].sort_values("cycle")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(cell["cycle"], cell["soh"], label="Capacity-derived SOH", linewidth=2)
    ax.plot(cell["cycle"], cell["predicted_soh"], label="Model-estimated SOH", linewidth=2)
    ax.fill_between(cell["cycle"].to_numpy(), cell["uncertainty_low"].to_numpy(),
                    cell["uncertainty_high"].to_numpy(), alpha=0.2, label="10th–90th tree range")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("SOH (%)")
    ax.set_title(f"Battery health trend — {cell_id}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("The shaded area is the spread between model trees, not a guaranteed confidence interval.")
with tab2:
    st.write("Isolation Forest flags data points that differ from patterns in the training data. A flag is not proof of a fault or danger.")
    flagged = results[results["anomaly"]].sort_values("anomaly_score")
    if flagged.empty:
        st.success("No unusual readings flagged.")
    else:
        st.warning(f"{len(flagged)} readings flagged for review.")
        st.dataframe(flagged[["cell_id","cycle","voltage","current","temperature","capacity_ah","soh","anomaly_score"]].head(500), use_container_width=True)
with tab3:
    st.write("These metrics describe a holdout split from the selected dataset. They do not establish performance on every EV or battery type.")
    a,b,c = st.columns(3)
    a.metric("MAE", f"{metrics['mae']:.3f} percentage points")
    b.metric("RMSE", f"{metrics['rmse']:.3f} percentage points")
    c.metric("R²", f"{metrics['r2']:.3f}" if np.isfinite(metrics["r2"]) else "N/A")
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.scatter(metrics["y_test"], metrics["preds"], alpha=0.55)
    lo = min(float(metrics["y_test"].min()), float(np.min(metrics["preds"])))
    hi = max(float(metrics["y_test"].max()), float(np.max(metrics["preds"])))
    ax.plot([lo,hi],[lo,hi], linestyle="--")
    ax.set_xlabel("Measured SOH (%)")
    ax.set_ylabel("Estimated SOH (%)")
    ax.set_title("Measured vs estimated on holdout data")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    st.subheader("Which inputs influenced the model most?")
    st.bar_chart(importance.set_index("Feature"))
with tab4:
    st.dataframe(results, use_container_width=True, height=420)
    st.download_button("Download analysis results (CSV)", results.to_csv(index=False).encode("utf-8"),
                       file_name="battery_health_analysis.csv", mime="text/csv")

st.divider()
st.subheader("What the results mean")
st.markdown("""
- **Estimated SOH:** estimated battery condition relative to its initial measured capacity.
- **Unusual reading:** a measurement that differs from the patterns the anomaly detector learned.
- **MAE:** average absolute difference between measured and estimated SOH on the holdout data; lower is better.
- **Important limitation:** for this prototype, SOH labels are calculated from capacity data. Accurate real-world results require suitable, representative battery data and independent validation.
""")
st.caption("Live API mode fetches data only when you press Connect and fetch readings. It is not continuous streaming. Never connect this prototype directly to high-voltage battery wiring.")