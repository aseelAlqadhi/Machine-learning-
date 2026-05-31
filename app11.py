import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, mean_squared_error, r2_score,
                              silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Delivery Prediction",
    page_icon="📦",
    layout="wide"
)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("ecommerce_orders.csv")
    df.drop_duplicates(inplace=True)
    df["gender"] = df["gender"].str.strip().str.title()
    df.drop(columns=["order_id"], inplace=True)
    return df

# ─────────────────────────────────────────────
#  TRAIN MODELS (no pkl files needed)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models(_df):
    # ── Classification ──
    X = _df.drop(columns=["reached_on_time"])
    y = _df["reached_on_time"]

    numerical_features = X.select_dtypes(include="number").columns.tolist()
    ordinal_features   = ["order_priority"]
    nominal_features   = [c for c in X.select_dtypes(include="object").columns if c not in ordinal_features]

    clf_preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), numerical_features),
        ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(categories=[["Low","Medium","High","Critical"]]))]), ordinal_features),
        ("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), nominal_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best_clf = Pipeline([("preprocessor", clf_preprocessor),
                         ("classifier", LogisticRegression(random_state=42, max_iter=1000))])
    best_clf.fit(X_train, y_train)

    # ── Regression ──
    X_reg = _df.drop(columns=["cost_of_the_product", "reached_on_time"])
    y_reg = _df["cost_of_the_product"]

    num_reg = X_reg.select_dtypes(include="number").columns.tolist()
    ord_reg = ["order_priority"]
    nom_reg = [c for c in X_reg.select_dtypes(include="object").columns if c not in ord_reg]

    reg_preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), num_reg),
        ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(categories=[["Low","Medium","High","Critical"]]))]), ord_reg),
        ("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), nom_reg),
    ])

    X_reg_train, _, y_reg_train, _ = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    best_reg = Pipeline([("preprocessor", reg_preprocessor), ("regressor", LinearRegression())])
    best_reg.fit(X_reg_train, y_reg_train)

    # ── Clustering ──
    cluster_features = ["customer_age", "cost_of_the_product", "prior_purchases",
                        "discount_offered", "weight_in_gms", "customer_rating"]
    cluster_pre = Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())])
    cluster_pre.fit(_df[cluster_features])
    X_scaled = cluster_pre.transform(_df[cluster_features])
    cluster_model = KMeans(n_clusters=2, random_state=42, n_init="auto")
    cluster_model.fit(X_scaled)

    return best_clf, best_reg, cluster_pre, cluster_model


df = load_data()
best_clf, best_reg, cluster_pre, cluster_model = load_models(df)

# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1170/1170678.png", width=80)
st.sidebar.title("📦 E-Commerce ML App")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Analysis & EDA", "🎯 Classification", "📈 Regression", "🔵 Clustering", "🔮 Live Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Stats**")
st.sidebar.metric("Total Rows", f"{len(df):,}")
st.sidebar.metric("Total Columns", df.shape[1])
st.sidebar.metric("Missing Values", int(df.isna().sum().sum()))

# ═══════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📦 E-Commerce Order Delivery Prediction")
    st.markdown("### Machine Learning Project — End-to-End Pipeline")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🗂️ Rows", f"{len(df):,}")
    col2.metric("📋 Columns", df.shape[1])
    col3.metric("🎯 On-Time Rate", f"{df['reached_on_time'].mean()*100:.1f}%")
    col4.metric("❓ Missing Values", int(df.isna().sum().sum()))

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🎯 Project Goals")
        st.markdown("""
- **Classification** — Predict if an order will be delivered on time (Yes/No)
- **Regression** — Predict the cost of a product ($)
- **Clustering** — Discover natural customer segments

All models are built using **Scikit-learn Pipelines** to prevent data leakage.
        """)

    with col_r:
        st.markdown("### 📦 Dataset Features")
        feature_info = {
            "Feature": ["customer_age", "gender", "product_category", "order_priority",
                        "warehouse_block", "mode_of_shipment", "customer_rating",
                        "cost_of_the_product", "prior_purchases", "discount_offered", "weight_in_gms"],
            "Type": ["Numerical", "Categorical", "Categorical", "Ordinal",
                     "Categorical", "Categorical", "Numerical",
                     "Numerical", "Numerical", "Numerical", "Numerical"],
            "Description": ["Age of customer", "Customer gender", "Product type",
                            "Urgency level (Low→Critical)", "Shipping warehouse",
                            "Ship / Flight / Road", "Rating 1–5", "Product price ($)",
                            "Past purchases count", "Discount %", "Product weight (grams)"]
        }
        st.dataframe(pd.DataFrame(feature_info), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 👀 Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ═══════════════════════════════════════════════════════
elif page == "📊 Analysis & EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 Overview", "❓ Missing Values", "🎯 Target Distribution",
         "📈 Feature Analysis", "🔥 Correlations"]
    )

    with tab1:
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Numerical Statistics**")
            st.dataframe(df.describe().round(2), use_container_width=True)
        with col2:
            st.markdown("**Categorical Statistics**")
            st.dataframe(df.describe(include="object"), use_container_width=True)

        st.markdown("---")
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.values.astype(str),
            "Non-Null Count": df.notna().sum().values,
            "Null Count": df.isna().sum().values
        })
        st.dataframe(dtype_df, hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Missing Values Analysis")
        missing = df.isna().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        missing = missing[missing["Missing Count"] > 0]

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(missing, hide_index=True, use_container_width=True)
        with col2:
            fig = px.bar(missing, x="Column", y="Missing %",
                         title="Missing Values by Column (%)",
                         color="Missing %", color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

        st.info("💡 These missing values are handled inside the Pipeline using **SimpleImputer**: mean for numerical columns, most_frequent for categorical columns.")

    with tab3:
        st.subheader("Target Variable: reached_on_time")
        col1, col2 = st.columns(2)

        counts = df["reached_on_time"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        counts["Label"] = counts["Status"].map({1: "On Time ✅", 0: "Delayed ❌"})

        with col1:
            fig = px.pie(counts, names="Label", values="Count",
                         title="Delivery Status Distribution",
                         color_discrete_sequence=["#2ecc71", "#e74c3c"])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(counts, x="Label", y="Count",
                          title="Count per Class",
                          color="Label",
                          color_discrete_map={"On Time ✅": "#2ecc71", "Delayed ❌": "#e74c3c"})
            st.plotly_chart(fig2, use_container_width=True)

        on_time_pct = df["reached_on_time"].mean() * 100
        st.success(f"✅ **{on_time_pct:.1f}%** of orders are delivered on time — **{100-on_time_pct:.1f}%** are delayed.")
        st.info("💡 The classes are fairly balanced (roughly 60/40), so no special resampling is needed.")

    with tab4:
        st.subheader("Categorical Features vs Delivery Status")
        cat_cols = ["product_category", "order_priority", "mode_of_shipment", "gender", "warehouse_block"]
        selected_cat = st.selectbox("Select a categorical feature:", cat_cols)

        grp = df.groupby([selected_cat, "reached_on_time"]).size().reset_index(name="Count")
        grp["Status"] = grp["reached_on_time"].map({1: "On Time", 0: "Delayed"})
        fig = px.bar(grp, x=selected_cat, y="Count", color="Status",
                     barmode="group",
                     title=f"{selected_cat} vs Delivery Status",
                     color_discrete_map={"On Time": "#2ecc71", "Delayed": "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Numerical Feature Distributions")
        num_cols = ["customer_age", "cost_of_the_product", "discount_offered",
                    "weight_in_gms", "customer_rating", "prior_purchases"]
        selected_num = st.selectbox("Select a numerical feature:", num_cols)

        fig2 = px.histogram(df, x=selected_num, color=df["reached_on_time"].map({1:"On Time",0:"Delayed"}),
                            nbins=40, barmode="overlay", opacity=0.7,
                            title=f"Distribution of {selected_num} by Delivery Status",
                            color_discrete_map={"On Time": "#2ecc71", "Delayed": "#e74c3c"},
                            labels={"color": "Status"})
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Boxplot — Outlier Detection")
        box_col = st.selectbox("Select column for boxplot:", num_cols, key="box")
        fig3 = px.box(df, x="reached_on_time", y=box_col,
                      color=df["reached_on_time"].map({1:"On Time",0:"Delayed"}),
                      title=f"Boxplot of {box_col}",
                      color_discrete_map={"On Time": "#2ecc71", "Delayed": "#e74c3c"})
        st.plotly_chart(fig3, use_container_width=True)

    with tab5:
        st.subheader("Correlation Heatmap — Numerical Features")
        corr = df.select_dtypes(include="number").corr()
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close()

        st.info("💡 Values close to +1 or -1 show strong correlation. Values near 0 mean no linear relationship.")


# ═══════════════════════════════════════════════════════
#  PAGE 3 — CLASSIFICATION
# ═══════════════════════════════════════════════════════
elif page == "🎯 Classification":
    st.title("🎯 Classification — Predicting Delivery Status")
    st.markdown("**Goal:** Predict whether an order will be delivered on time (1) or delayed (0).")
    st.markdown("---")

    st.subheader("⚙️ Pipeline Architecture")
    st.code("""
Pipeline(steps=[
    ('preprocessor', ColumnTransformer([
        ('num', Pipeline([imputer(mean) → StandardScaler]), numerical_features),
        ('ord', Pipeline([imputer(mode) → OrdinalEncoder]),  ['order_priority']),
        ('nom', Pipeline([imputer(mode) → OneHotEncoder]),   nominal_features)
    ])),
    ('classifier', <Best Model>)
])
    """, language="python")

    X = df.drop(columns=["reached_on_time"])
    y = df["reached_on_time"]

    numerical_features = X.select_dtypes(include="number").columns.tolist()
    ordinal_features   = ["order_priority"]
    nominal_features   = [c for c in X.select_dtypes(include="object").columns if c not in ordinal_features]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), numerical_features),
        ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(categories=[["Low","Medium","High","Critical"]]))]), ordinal_features),
        ("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), nominal_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    @st.cache_resource
    def train_classifiers():
        classifiers = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Support Vector Machine": SVC(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        }
        results = {}
        pipelines = {}
        for name, clf in classifiers.items():
            pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm     = confusion_matrix(y_test, y_pred)
            results[name]   = {"accuracy": acc, "report": report, "cm": cm, "y_pred": y_pred}
            pipelines[name] = pipe
        return results, pipelines

    with st.spinner("Training classifiers..."):
        clf_results, clf_pipelines = train_classifiers()

    st.subheader("📊 Model Comparison")
    comp_df = pd.DataFrame({
        "Model": list(clf_results.keys()),
        "Accuracy": [clf_results[m]["accuracy"] for m in clf_results]
    }).sort_values("Accuracy", ascending=False)

    fig = px.bar(comp_df, x="Model", y="Accuracy",
                 title="Classification Accuracy by Model",
                 color="Accuracy", color_continuous_scale="Viridis",
                 text=comp_df["Accuracy"].apply(lambda x: f"{x:.4f}"))
    fig.update_layout(yaxis_range=[0.5, 1.0])
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    best_name = comp_df.iloc[0]["Model"]
    st.success(f"🏆 Best Model: **{best_name}** with Accuracy = **{comp_df.iloc[0]['Accuracy']:.4f}**")

    st.markdown("---")
    st.subheader("🔍 Detailed Model Metrics")
    selected_model = st.selectbox("Select a model to inspect:", list(clf_results.keys()))
    res = clf_results[selected_model]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{res['accuracy']:.4f}")
    col2.metric("Precision (weighted)", f"{res['report']['weighted avg']['precision']:.4f}")
    col3.metric("Recall (weighted)",    f"{res['report']['weighted avg']['recall']:.4f}")
    col4.metric("F1 Score (weighted)",  f"{res['report']['weighted avg']['f1-score']:.4f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(res["report"]).T.drop(columns=["support"], errors="ignore")
        st.dataframe(report_df.round(4), use_container_width=True)

    with col_b:
        st.markdown("**Confusion Matrix**")
        cm = res["cm"]
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Delayed", "On Time"],
                    yticklabels=["Delayed", "On Time"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {selected_model}")
        st.pyplot(fig_cm)
        plt.close()


# ═══════════════════════════════════════════════════════
#  PAGE 4 — REGRESSION
# ═══════════════════════════════════════════════════════
elif page == "📈 Regression":
    st.title("📈 Regression — Predicting Product Cost")
    st.markdown("**Goal:** Predict `cost_of_the_product` — a continuous numerical value (in $).")
    st.markdown("---")

    X_reg = df.drop(columns=["cost_of_the_product", "reached_on_time"])
    y_reg = df["cost_of_the_product"]

    num_reg = X_reg.select_dtypes(include="number").columns.tolist()
    ord_reg = ["order_priority"]
    nom_reg = [c for c in X_reg.select_dtypes(include="object").columns if c not in ord_reg]

    reg_preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), num_reg),
        ("ord", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(categories=[["Low","Medium","High","Critical"]]))]), ord_reg),
        ("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), nom_reg),
    ])

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    @st.cache_resource
    def train_regressors():
        regressors = {
            "Linear Regression":       LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        }
        results = {}
        for name, reg in regressors.items():
            pipe = Pipeline([("preprocessor", reg_preprocessor), ("regressor", reg)])
            pipe.fit(X_reg_train, y_reg_train)
            y_pred = pipe.predict(X_reg_test)
            results[name] = {
                "MSE":    round(mean_squared_error(y_reg_test, y_pred), 2),
                "RMSE":   round(mean_squared_error(y_reg_test, y_pred)**0.5, 2),
                "R2":     round(r2_score(y_reg_test, y_pred), 4),
                "y_pred": y_pred
            }
        return results

    with st.spinner("Training regression models..."):
        reg_results = train_regressors()

    st.subheader("📊 Model Comparison")
    comp_df = pd.DataFrame({
        "Model":    list(reg_results.keys()),
        "R² Score": [reg_results[m]["R2"]   for m in reg_results],
        "MSE":      [reg_results[m]["MSE"]  for m in reg_results],
        "RMSE":     [reg_results[m]["RMSE"] for m in reg_results],
    }).sort_values("R² Score", ascending=False)
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    best_reg_name = comp_df.iloc[0]["Model"]
    st.success(f"🏆 Best Model: **{best_reg_name}** with R² = **{comp_df.iloc[0]['R² Score']:.4f}**")

    fig = px.bar(comp_df, x="Model", y="R² Score",
                 title="R² Score Comparison",
                 color="R² Score", color_continuous_scale="Teal",
                 text=comp_df["R² Score"].apply(lambda x: f"{x:.4f}"))
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Actual vs Predicted")
    selected_reg = st.selectbox("Select model:", list(reg_results.keys()))
    y_pred_plot  = reg_results[selected_reg]["y_pred"]

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{reg_results[selected_reg]['R2']:.4f}")
    col2.metric("MSE",      f"{reg_results[selected_reg]['MSE']:,.2f}")
    col3.metric("RMSE",     f"${reg_results[selected_reg]['RMSE']:,.2f}")

    scatter_df = pd.DataFrame({"Actual": y_reg_test.values, "Predicted": y_pred_plot})
    fig2 = px.scatter(scatter_df, x="Actual", y="Predicted",
                      title=f"Actual vs Predicted Cost — {selected_reg}",
                      opacity=0.5)
    max_val = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
    fig2.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                   line=dict(color="red", dash="dash"))
    st.plotly_chart(fig2, use_container_width=True)
    st.info("💡 The closer the dots are to the red diagonal line, the more accurate the predictions.")


# ═══════════════════════════════════════════════════════
#  PAGE 5 — CLUSTERING
# ═══════════════════════════════════════════════════════
elif page == "🔵 Clustering":
    st.title("🔵 Clustering — Customer Segmentation")
    st.markdown("**Goal:** Discover natural groups in orders using unsupervised learning — no labels needed.")
    st.markdown("---")

    cluster_features = ["customer_age", "cost_of_the_product", "prior_purchases",
                        "discount_offered", "weight_in_gms", "customer_rating"]
    df_cluster = df[cluster_features].copy()

    cluster_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("sc",  StandardScaler())
    ])
    X_scaled = cluster_pipe.fit_transform(df_cluster)

    st.subheader("📐 Elbow Method — Finding Optimal K")

    @st.cache_data
    def elbow_data():
        inertias, sil_scores = [], []
        for k in range(1, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(X_scaled)
            inertias.append(km.inertia_)
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X_scaled)
            sil_scores.append(round(silhouette_score(X_scaled, labels), 4))
        return inertias, sil_scores

    inertias, sil_scores = elbow_data()

    col1, col2 = st.columns(2)
    with col1:
        fig_elbow = px.line(x=list(range(1,11)), y=inertias, markers=True,
                            title="Elbow Method (Inertia vs K)",
                            labels={"x": "Number of Clusters (k)", "y": "Inertia (WCSS)"})
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        fig_sil = px.line(x=list(range(2,11)), y=sil_scores, markers=True,
                          title="Silhouette Score vs K",
                          labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"})
        st.plotly_chart(fig_sil, use_container_width=True)

    best_k = sil_scores.index(max(sil_scores)) + 2
    st.success(f"🏆 Best K = **{best_k}** (Silhouette Score = **{max(sil_scores):.4f}**)")

    st.markdown("---")
    st.subheader("🔍 Cluster Visualization")
    k_choice = st.slider("Choose number of clusters (k):", min_value=2, max_value=8, value=best_k)

    km_final = KMeans(n_clusters=k_choice, random_state=42, n_init="auto")
    df_cluster["Cluster"] = km_final.fit_predict(X_scaled).astype(str)

    x_axis = st.selectbox("X axis:", cluster_features, index=1)
    y_axis = st.selectbox("Y axis:", cluster_features, index=3)

    fig_c = px.scatter(df_cluster, x=x_axis, y=y_axis, color="Cluster",
                       title=f"Customer Segments (k={k_choice}): {x_axis} vs {y_axis}",
                       opacity=0.7)
    st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Cluster Profiles (Average per Cluster)")
    profile = df_cluster.groupby("Cluster")[cluster_features].mean().round(2)
    st.dataframe(profile, use_container_width=True)

    st.markdown("---")
    st.subheader("⚖️ K-Means vs DBSCAN Comparison")
    dbscan = DBSCAN(eps=0.6, min_samples=5)
    db_labels = dbscan.fit_predict(X_scaled)
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    db_sil = silhouette_score(X_scaled, db_labels) if n_db_clusters >= 2 else -1

    comparison = pd.DataFrame({
        "Model":            [f"K-Means (k={k_choice})", "DBSCAN"],
        "Clusters Found":   [k_choice, n_db_clusters],
        "Silhouette Score": [round(silhouette_score(X_scaled, km_final.labels_), 4), round(db_sil, 4)],
        "Noise Points":     [0, list(db_labels).count(-1)]
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════
#  PAGE 6 — LIVE PREDICTION
# ═══════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Prediction")
    st.markdown("Fill in the order details below and get instant predictions from all three models.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Customer Info**")
        age    = st.slider("Customer Age", 18, 70, 35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        rating = st.slider("Customer Rating", 1, 5, 3)
        prior  = st.slider("Prior Purchases", 1, 7, 3)

    with col2:
        st.markdown("**📦 Order Info**")
        category = st.selectbox("Product Category",
                                ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Toys"])
        priority = st.selectbox("Order Priority", ["Low", "Medium", "High", "Critical"])
        discount = st.slider("Discount Offered (%)", 0, 65, 10)
        cost     = st.number_input("Cost of Product ($)", min_value=30.0, max_value=5000.0, value=250.0)

    with col3:
        st.markdown("**🚚 Shipping Info**")
        warehouse = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "F"])
        shipment  = st.selectbox("Mode of Shipment", ["Ship", "Flight", "Road"])
        weight    = st.slider("Weight (grams)", 1000, 7000, 3500)

    st.markdown("---")

    if st.button("🔮 Predict Now", type="primary", use_container_width=True):

        clf_input = pd.DataFrame([{
            "customer_age": float(age), "gender": gender, "product_category": category,
            "order_priority": priority, "warehouse_block": warehouse, "mode_of_shipment": shipment,
            "customer_rating": float(rating), "cost_of_the_product": float(cost),
            "prior_purchases": int(prior), "discount_offered": float(discount), "weight_in_gms": float(weight)
        }])

        reg_input = pd.DataFrame([{
            "customer_age": float(age), "gender": gender, "product_category": category,
            "order_priority": priority, "warehouse_block": warehouse, "mode_of_shipment": shipment,
            "customer_rating": float(rating), "prior_purchases": int(prior),
            "discount_offered": float(discount), "weight_in_gms": float(weight)
        }])

        clust_input = pd.DataFrame([{
            "customer_age": float(age), "cost_of_the_product": float(cost),
            "prior_purchases": int(prior), "discount_offered": float(discount),
            "weight_in_gms": float(weight), "customer_rating": float(rating)
        }])

        clf_pred  = best_clf.predict(clf_input)[0]
        clf_proba = best_clf.predict_proba(clf_input)[0] if hasattr(best_clf, "predict_proba") else None
        reg_pred  = best_reg.predict(reg_input)[0]
        clust_scaled = cluster_pre.transform(clust_input)
        clust_pred   = cluster_model.predict(clust_scaled)[0]

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("### 🎯 Delivery Prediction")
            if clf_pred == 1:
                st.success("✅ **ON TIME**")
                st.markdown("This order is predicted to be delivered on time.")
            else:
                st.error("❌ **DELAYED**")
                st.markdown("This order is predicted to be delayed.")
            if clf_proba is not None:
                st.markdown(f"**Confidence:** {max(clf_proba)*100:.1f}%")
                fig_prob = go.Figure(go.Bar(
                    x=["Delayed", "On Time"], y=[clf_proba[0], clf_proba[1]],
                    marker_color=["#e74c3c", "#2ecc71"]
                ))
                fig_prob.update_layout(title="Prediction Probability", height=250,
                                       yaxis_range=[0,1], margin=dict(t=40,b=20))
                st.plotly_chart(fig_prob, use_container_width=True)

        with col_b:
            st.markdown("### 💰 Cost Prediction")
            st.info(f"**Predicted Cost:** ${reg_pred:,.2f}")
            st.markdown(f"You entered: **${cost:,.2f}**")
            diff = reg_pred - cost
            if abs(diff) < 50:
                st.success(f"Very close! Difference: ${diff:+.2f}")
            else:
                st.warning(f"Difference from entered cost: ${diff:+.2f}")

        with col_c:
            st.markdown("### 🔵 Customer Segment")
            st.info(f"**Cluster:** {clust_pred}")
            segment_names = {0: "Budget Buyer", 1: "Premium Shopper",
                             2: "Discount Hunter", 3: "Occasional Buyer"}
            seg_name = segment_names.get(clust_pred, f"Segment {clust_pred}")
            st.markdown(f"**Profile:** {seg_name}")
            st.markdown("Customer segments are discovered automatically — no labels needed.")
