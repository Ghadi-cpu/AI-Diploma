# ==============================================================
#  Laptop Price Predictor - Streamlit App
#  Soft pastel theme with dark readable type
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------
# Pastel theme: soft tints, dark readable text
# --------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(160deg, #f3f0fb 0%, #eef6f4 50%, #fdf2f0 100%);
        color: #3a3c4e;
    }
    /* Top header bar - make it light */
    [data-testid="stHeader"] {
        background: rgba(243,240,251,0.85) !important;
        backdrop-filter: blur(6px);
    }
    [data-testid="stHeader"] * { color: #6b6e8a !important; }
    html, body, [class*="css"], p, span, label, div {
        font-family: 'Inter', sans-serif;
    }
    .stApp, .stApp p, .stApp label, .stApp span, .stMarkdown {
        color: #3a3c4e !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Fraunces', serif !important;
        color: #3a3c4e !important;
        letter-spacing: -0.01em;
    }

    /* Sidebar - soft lavender */
    [data-testid="stSidebar"] {
        background: #e7e0f7;
        border-right: 1px solid #d9cff0;
    }
    [data-testid="stSidebar"] * { color: #3a3c4e !important; }
    /* Radio button dots - light pastel instead of dark */
    [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
        background-color: #ffffff !important;
        border-color: #c4b8e8 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] > div:first-child {
        background-color: #b8a4e3 !important;
        border-color: #9d86d4 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] svg { fill: #b8a4e3 !important; }

    /* Hero */
    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: 58px; font-weight: 700;
        color: #3a3c4e; line-height: 1.02; margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 18px; color: #6b6e8a !important;
        font-weight: 400; margin-top: 12px;
    }
    .accent-bar {
        display:inline-block; width: 7px; height: 30px;
        background: linear-gradient(180deg, #b8a4e3, #8fd4c4);
        border-radius: 4px; margin-right: 14px; vertical-align: middle;
    }
    .section-head {
        font-family:'Fraunces',serif; font-size: 28px; font-weight:600;
        color:#3a3c4e; margin: 10px 0 6px 0;
    }

    /* Stat cards - pastel tints */
    .stat-card {
        border-radius: 18px; padding: 26px;
        border: 1px solid rgba(255,255,255,0.6);
        box-shadow: 0 6px 20px rgba(150,140,200,0.12);
        transition: transform .25s ease, box-shadow .25s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(150,140,200,0.22);
    }
    .card-lav { background: #ece5fb; }
    .card-mint { background: #ddf3ec; }
    .card-peach { background: #fde6df; }
    .card-sky { background: #dcecfb; }
    .stat-number {
        font-family: 'Fraunces', serif; font-size: 44px; font-weight: 700;
        color: #5e4b8b; margin: 0; line-height: 1;
    }
    .stat-label {
        font-size: 12px; color: #6b6e8a !important; margin-top: 10px;
        text-transform: uppercase; letter-spacing: 0.1em;
    }

    /* Price result card */
    .price-card {
        background: linear-gradient(135deg, #ece5fb 0%, #ddf3ec 100%);
        border: 1px solid #cdbff0;
        padding: 40px; border-radius: 22px; text-align: center;
        box-shadow: 0 10px 34px rgba(150,140,200,0.2);
        margin: 26px 0;
    }
    .price-value {
        font-family: 'Fraunces', serif; font-size: 60px; font-weight: 700;
        color: #5e4b8b; margin: 0;
    }
    .price-label {
        font-size: 13px; color: #8a7fb0 !important;
        text-transform: uppercase; letter-spacing: 0.14em; margin: 0 0 8px 0;
    }
    .spec-badge {
        display: inline-block; background: #ffffff; border: 1px solid #e0d8f2;
        color: #5e4b8b !important; padding: 7px 15px; border-radius: 10px;
        margin: 4px; font-size: 13px; font-weight: 600;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #b8a4e3, #8fd4c4);
        color: #3a3c4e; border: none; border-radius: 12px;
        padding: 13px 30px; font-size: 16px; font-weight: 700;
        font-family: 'Inter', sans-serif; width: 100%;
        transition: all .25s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(150,140,200,0.4);
        transform: translateY(-1px);
    }

    /* Inputs */
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-color: #e0d8f2 !important;
        color: #3a3c4e !important;
    }
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #3a3c4e !important; font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] { color: #5e4b8b !important; }

    /* Dataframes - light theme */
    [data-testid="stDataFrame"] {
        background: #ffffff !important;
        border: 1px solid #e4ddf2 !important;
        border-radius: 12px;
    }
    [data-testid="stDataFrame"] * {
        color: #3a3c4e !important;
    }
    [data-testid="stDataFrame"] thead tr th {
        background: #ece5fb !important;
        color: #5e4b8b !important;
    }
    [data-testid="stDataFrame"] [role="gridcell"] {
        background: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# Load data and models
# --------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("laptop_before_encoding.csv"), pd.read_csv("laptop_cleaned.csv")

@st.cache_resource
def load_models():
    dt = joblib.load("decision_tree_model.pkl")
    rf = joblib.load("random_forest_model.pkl")
    lr = joblib.load("linear_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("model_columns.pkl")
    return dt, rf, lr, scaler, columns

try:
    before, cleaned = load_data()
    dt_model, rf_model, lr_model, scaler, model_columns = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# Pastel plotly palette
PASTEL_SEQ = ["#b8a4e3", "#8fd4c4", "#f7b7a3", "#9ec5e8", "#f4c89a", "#c4a3d9"]
PASTEL_SCALE = [[0, "#dcecfb"], [0.5, "#b8a4e3"], [1, "#f7b7a3"]]

def style_fig(fig, title=""):
    fig.update_layout(
        plot_bgcolor="rgba(255,255,255,0.4)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#3a3c4e", size=13),
        title=dict(text=title, font=dict(family="Fraunces", size=19, color="#3a3c4e")),
        margin=dict(t=54, b=40, l=40, r=20),
        legend=dict(font=dict(color="#3a3c4e")),
    )
    fig.update_xaxes(gridcolor="#e4ddf2", zerolinecolor="#d8cff0", color="#3a3c4e",
                     title_font=dict(color="#3a3c4e"), tickfont=dict(color="#3a3c4e"))
    fig.update_yaxes(gridcolor="#e4ddf2", zerolinecolor="#d8cff0", color="#3a3c4e",
                     title_font=dict(color="#3a3c4e"), tickfont=dict(color="#3a3c4e"))
    fig.update_traces(textfont_color="#3a3c4e")
    return fig

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
st.sidebar.markdown("<h2 style='font-family:Fraunces;color:#5e4b8b;'>Laptop Price AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#6b6e8a;'>Predict laptop prices from specs.</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Overview", "Data Exploration", "Preprocessing", "Model Performance", "Predict Price"])
st.sidebar.markdown("---")
st.sidebar.caption("Machine learning regression project predicting laptop prices from hardware specifications.")

if not models_loaded:
    st.error(f"Could not load model or data files. Make sure all files are in the same folder.\n\nDetails: {load_error}")
    st.stop()

def hero(title, sub):
    st.markdown(f"<div class='hero-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hero-sub'>{sub}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

def head(text):
    st.markdown(f"<div class='section-head'><span class='accent-bar'></span>{text}</div>", unsafe_allow_html=True)

# ==============================================================
# Page 1: Overview
# ==============================================================
if page == "Overview":
    hero("Laptop Price<br>Predictor", "Estimate a laptop's price from its specifications using machine learning.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='stat-card card-lav'><p class='stat-number'>{before.shape[0]:,}</p><p class='stat-label'>Laptops</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='stat-card card-mint'><p class='stat-number'>3</p><p class='stat-label'>Models</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='stat-card card-peach'><p class='stat-number'>{before['Company'].nunique()}</p><p class='stat-label'>Brands</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='stat-card card-sky'><p class='stat-number'>{cleaned.shape[1]-1}</p><p class='stat-label'>Features</p></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    head("About this project")
    st.write("""
    This application predicts the price of a laptop based on its specifications brand, type,
    RAM, storage, processor, display, and more. It is built on three regression models you can
    compare and choose from. Use the sidebar to explore the data, review the preprocessing,
    compare model performance, and run interactive predictions.
    """)

    head("Sample of the data")
    st.dataframe(before.head(10), use_container_width=True)

# ==============================================================
# Page 2: Data Exploration
# ==============================================================
elif page == "Data Exploration":
    hero("Data Exploration", "A visual look at what drives laptop prices.")

    head("Price distribution")
    fig1 = px.histogram(before, x="Price", nbins=50, color_discrete_sequence=["#b8a4e3"])
    fig1.update_layout(bargap=0.05)
    st.plotly_chart(style_fig(fig1, "Distribution of laptop prices"), use_container_width=True)
    st.caption("Most laptops fall in the low-to-mid range, with a long tail toward higher prices (right-skewed).")

    head("Average price by type")
    type_avg = before.groupby("TypeName")["Price"].mean().sort_values(ascending=False).reset_index()
    fig2 = px.bar(type_avg, x="TypeName", y="Price", color="TypeName", color_discrete_sequence=PASTEL_SEQ)
    fig2.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig2, "Workstations and gaming laptops lead; notebooks are cheapest"), use_container_width=True)

    head("RAM vs price")
    fig3 = px.scatter(before, x="Ram", y="Price", color="Price", color_continuous_scale=PASTEL_SCALE, size="Ram", size_max=14)
    st.plotly_chart(style_fig(fig3, "Price rises with RAM"), use_container_width=True)

    head("SSD capacity vs price")
    fig4 = px.scatter(before, x="SSD", y="Price", color="Price", color_continuous_scale=PASTEL_SCALE)
    st.plotly_chart(style_fig(fig4, "Larger SSDs tend toward higher prices"), use_container_width=True)

    head("Average price by brand")
    comp_avg = before.groupby("Company")["Price"].mean().sort_values(ascending=False).reset_index()
    fig5 = px.bar(comp_avg, x="Company", y="Price", color="Price", color_continuous_scale=PASTEL_SCALE)
    st.plotly_chart(style_fig(fig5, "Brands ranked by average price"), use_container_width=True)

    head("Correlation heatmap")
    num_cols = ["Ram", "Weight", "Touchscreen", "IPS", "HDD", "SSD", "Cpu_Speed", "PPI", "Price"]
    corr = before[num_cols].corr()
    fig6 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=PASTEL_SCALE)
    st.plotly_chart(style_fig(fig6, "How numeric features relate to price"), use_container_width=True)
    st.caption("RAM, SSD, and PPI are among the features most correlated with price.")

# ==============================================================
# Page 3: Preprocessing
# ==============================================================
elif page == "Preprocessing":
    hero("Preprocessing", "The steps applied to prepare the data for modeling.")

    head("What was done")
    st.write("""
    1. **Missing values and duplicates** : removed a fully empty row and dropped duplicate records.
    2. **Text to numbers** : converted values like `8GB` into the number `8` for RAM.
    3. **Median imputation** : filled missing weight and screen-size values with the median, which is less sensitive to outliers than the mean.
    4. **Feature engineering** : split storage into HDD / SSD / Hybrid / Flash, extracted CPU speed and brand, and computed pixel density (PPI).
    5. **One-hot encoding** : converted text columns into numeric 0/1 columns for the models.
    """)

    c1, c2 = st.columns(2)
    with c1:
        head("Before")
        st.write(f"Columns: **{before.shape[1]}**")
        st.dataframe(before.head(5), use_container_width=True)
    with c2:
        head("After")
        st.write(f"Columns: **{cleaned.shape[1]}**")
        st.dataframe(cleaned.head(5), use_container_width=True)

# ==============================================================
# Page 4: Model Performance
# ==============================================================
elif page == "Model Performance":
    hero("Model Performance", "How the three regression models compare on the test set.")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    X = cleaned.drop(columns=["Price"]).astype(float)
    y = cleaned["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)

    results, preds = [], {}
    for name, model, data in [
        ("Decision Tree", dt_model, X_test),
        ("Random Forest", rf_model, X_test),
        ("Linear Regression", lr_model, X_test_scaled),
    ]:
        pred = model.predict(data); preds[name] = pred
        results.append({"Model": name,
                        "R2": round(r2_score(y_test, pred), 3),
                        "MAE": round(mean_absolute_error(y_test, pred), 1),
                        "MSE": round(mean_squared_error(y_test, pred), 1)})
    res_df = pd.DataFrame(results)

    head("Metrics")
    st.dataframe(res_df, use_container_width=True)

    fig = px.bar(res_df, x="Model", y="R2", color="Model", color_discrete_sequence=PASTEL_SEQ, text="R2")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, "R2 by model (higher is better)"), use_container_width=True)

    best = res_df.loc[res_df["R2"].idxmax(), "Model"]
    st.success(f"Best model: {best} — it averages many trees, which reduces overfitting and captures non-linear patterns.")

    head("Actual vs predicted — all three models")
    st.caption("The closer the points hug the dashed line, the more accurate the model. Random Forest's points cluster tightest around the line.")

    from plotly.subplots import make_subplots
    colors = {"Decision Tree": "#8fd4c4", "Random Forest": "#9ec5e8", "Linear Regression": "#c4a3d9"}
    fig_all = make_subplots(rows=1, cols=3,
                            subplot_titles=[f"{n} (R2={res_df.loc[res_df['Model']==n,'R2'].values[0]})"
                                            for n in ["Decision Tree", "Random Forest", "Linear Regression"]],
                            horizontal_spacing=0.06)
    for i, name in enumerate(["Decision Tree", "Random Forest", "Linear Regression"], start=1):
        p = preds[name]
        fig_all.add_trace(go.Scatter(x=y_test, y=p, mode="markers",
                                     marker=dict(color=colors[name], opacity=0.55, size=6),
                                     showlegend=False), row=1, col=i)
        fig_all.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                     mode="lines", line=dict(color="#f08a6d", dash="dash"),
                                     showlegend=False), row=1, col=i)
        fig_all.update_xaxes(title_text="Actual", row=1, col=i)
        fig_all.update_yaxes(title_text="Predicted", row=1, col=i)
    fig_all.update_layout(height=420)
    # color the subplot titles dark gray
    for ann in fig_all.layout.annotations:
        ann.font.color = "#3a3c4e"
        ann.font.size = 14
    st.plotly_chart(style_fig(fig_all, ""), use_container_width=True)

# ==============================================================
# Page 5: Predict Price
# ==============================================================
elif page == "Predict Price":
    hero("Predict a Price", "Choose the specs, then run the prediction.")

    model_choice = st.selectbox("Model used for prediction",
                                ["Random Forest (best)", "Decision Tree", "Linear Regression"])

    head("Laptop specifications")
    col1, col2, col3 = st.columns(3)
    with col1:
        company = st.selectbox("Brand", sorted(before["Company"].unique()))
        typename = st.selectbox("Type", sorted(before["TypeName"].unique()))
        ram = st.selectbox("RAM (GB)", sorted(before["Ram"].unique()))
        opsys = st.selectbox("Operating system", sorted(before["OpSys"].unique()))
    with col2:
        cpu_brand = st.selectbox("Processor (CPU)", sorted(before["Cpu_Brand"].unique()))
        gpu_brand = st.selectbox("Graphics (GPU)", sorted(before["Gpu_Brand"].unique()))
        cpu_speed = st.slider("CPU speed (GHz)", 0.9, 3.6, 2.5, 0.1)
        weight = st.slider("Weight (kg)", 0.7, 4.7, 2.0, 0.1)
    with col3:
        ssd = st.selectbox("SSD (GB)", sorted(before["SSD"].unique()))
        hdd = st.selectbox("HDD (GB)", sorted(before["HDD"].unique()))
        ppi = st.slider("Pixel density (PPI)", 90.0, 352.0, 140.0, 1.0)
        touchscreen = st.radio("Touchscreen", ["No", "Yes"], horizontal=True)
        ips = st.radio("IPS display", ["No", "Yes"], horizontal=True)

    if st.button("Predict price"):
        d = dict.fromkeys(model_columns, 0)
        d["Ram"], d["Weight"] = ram, weight
        d["Touchscreen"] = 1 if touchscreen == "Yes" else 0
        d["IPS"] = 1 if ips == "Yes" else 0
        d["HDD"], d["SSD"] = hdd, ssd
        d["Hybrid"], d["Flash"] = 0, 0
        d["Cpu_Speed"], d["PPI"] = cpu_speed, ppi
        for prefix, value in [("Company", company), ("TypeName", typename), ("OpSys", opsys),
                              ("Cpu_Brand", cpu_brand), ("Gpu_Brand", gpu_brand)]:
            col_name = f"{prefix}_{value}"
            if col_name in d:
                d[col_name] = 1
        input_df = pd.DataFrame([d])[model_columns].astype(float)

        if model_choice.startswith("Random Forest"):
            prediction = rf_model.predict(input_df)[0]
        elif model_choice.startswith("Decision Tree"):
            prediction = dt_model.predict(input_df)[0]
        else:
            prediction = lr_model.predict(scaler.transform(input_df))[0]
        prediction = max(prediction, 0)

        st.markdown(f"<div class='price-card'><p class='price-label'>Estimated price</p>"
                    f"<p class='price-value'>{prediction:,.0f}</p></div>", unsafe_allow_html=True)

        head("Selected specifications")
        st.markdown(
            f"<span class='spec-badge'>{company}</span><span class='spec-badge'>{typename}</span>"
            f"<span class='spec-badge'>{ram}GB RAM</span><span class='spec-badge'>{ssd}GB SSD</span>"
            f"<span class='spec-badge'>{hdd}GB HDD</span><span class='spec-badge'>{cpu_brand}</span>"
            f"<span class='spec-badge'>{gpu_brand}</span><span class='spec-badge'>{opsys}</span>",
            unsafe_allow_html=True)
