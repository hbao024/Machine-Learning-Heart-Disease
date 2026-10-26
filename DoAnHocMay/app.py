"""
🫀 DEMO ĐỒ ÁN: Chẩn Đoán Bệnh Tim Bằng 8 Thuật Toán Học Máy
=====================================================================
Ứng dụng Streamlit tương tác đầy đủ với:
  - Khám phá dữ liệu (EDA)
  - Huấn luyện & so sánh 8 mô hình ML (Kèm 5-Fold Cross-Validation)
  - Đánh giá tầm quan trọng đặc trưng (Feature Importance)
  - Dự đoán / chẩn đoán bệnh nhân mới
=====================================================================
Chạy lệnh:  streamlit run app.py
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib 
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────
# Cấu hình trang Streamlit
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="🫀 Chẩn Đoán Bệnh Tim - ML Demo",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────
# Custom CSS cho giao diện đẹp
# ──────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin: 0;
        font-weight: 300;
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    .metric-card .value {
        font-size: clamp(1.4rem, 2.5vw, 2.2rem);
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        word-break: break-word;
    }
    .metric-card .label {
        font-size: 0.9rem;
        color: #555;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Model result card */
    .model-card {
        background: white;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: transform 0.2s ease;
    }
    .model-card:hover {
        transform: translateX(6px);
    }
    .model-card h3 {
        margin: 0 0 0.8rem 0;
        font-weight: 700;
        font-size: 1.15rem;
    }

    /* Prediction result */
    .pred-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(17,153,142,0.3);
    }
    .pred-danger {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(235,51,73,0.3);
    }
    .pred-safe h2, .pred-danger h2 {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
    }
    .pred-safe p, .pred-danger p {
        font-size: 1.1rem;
        margin: 0;
        opacity: 0.9;
    }

    /* Section divider */
    .section-divider {
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        border: none;
        border-radius: 2px;
        margin: 2rem 0;
    }

    /* Info card / Insight */
    .insight-card {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .insight-card h4 {
        margin: 0 0 0.4rem 0;
        color: #1e293b;
        font-weight: 700;
    }
    .insight-card p {
        margin: 0;
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: white;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: rgba(255,255,255,0.8);
    }

    /* Streamlit default adjustments */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────
# Hàm tiện ích & Cấu hình dữ liệu
# ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Đọc dữ liệu từ file heart.csv (UCI Cleveland chuẩn)"""
    data_path = os.path.join(os.path.dirname(__file__), "heart.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error("❌ Không tìm thấy file heart.csv!")
        return None


FEATURE_DESCRIPTIONS = {
    "age": "Tuổi bệnh nhân (năm)",
    "sex": "Giới tính (1 = Nam, 0 = Nữ)",
    "cp": "Loại đau ngực (0: Điển hình, 1: Không điển hình, 2: Không do tim, 3: Không triệu chứng)",
    "trestbps": "Huyết áp tâm thu lúc nghỉ (mm Hg)",
    "chol": "Cholesterol huyết thanh (mg/dl)",
    "fbs": "Đường huyết lúc đói > 120 mg/dl (1 = Đúng, 0 = Sai)",
    "restecg": "Kết quả điện tâm đồ lúc nghỉ (0: Bình thường, 1: Bất thường ST-T, 2: Phì đại thất trái)",
    "thalach": "Nhịp tim tối đa đạt được khi gắng sức (bpm)",
    "exang": "Đau thắt ngực khi gắng sức (1 = Có, 0 = Không)",
    "oldpeak": "Độ chênh xuống đoạn ST do gắng sức so với lúc nghỉ",
    "slope": "Độ dốc của đoạn ST đỉnh lúc gắng sức (0: Dốc lên, 1: Phẳng, 2: Dốc xuống)",
    "ca": "Số nhánh mạch máu chính được soi thấy (0-3)",
    "thal": "Kết quả xạ hình tưới máu cơ tim (1: Khiếm khuyết cố định, 2: Bình thường, 3: Khuyết có thể hồi phục)",
}

MODEL_COLORS = {
    "Logistic Regression": "#667eea",
    "Decision Tree": "#f093fb",
    "K-Nearest Neighbors": "#4facfe",
    "Support Vector Machine": "#43e97b",
    "Random Forest": "#fa709a",
    "Naive Bayes": "#fee140",
    "Gradient Boosting": "#a18cd1",
    "XGBoost": "#fbc2eb",
}


@st.cache_resource
def train_all_models(X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, feature_names):
    """
    Huấn luyện tất cả 8 mô hình ML kèm 5-Fold Stratified Cross-Validation
    và tính toán Feature Importance / Permutation Importance.
    """
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42, max_iter=1000),
            "scaled": True,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=5, random_state=42),
            "scaled": False,
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(n_neighbors=5),
            "scaled": True,
        },
        "Support Vector Machine": {
            "model": SVC(kernel="rbf", probability=True, random_state=42),
            "scaled": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "scaled": False,
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "scaled": True,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42
            ),
            "scaled": False,
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=4, 
                random_state=42, 
                eval_metric="logloss"
            ),
            "scaled": False,
        },
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, cfg in models_config.items():
        model = cfg["model"]
        X_tr = X_train_sc if cfg["scaled"] else X_train
        X_te = X_test_sc if cfg["scaled"] else X_test

        # 1. 5-Fold Cross-Validation trên tập huấn luyện
        cv_acc = cross_val_score(model, X_tr, y_train, cv=cv, scoring="accuracy")
        cv_auc = cross_val_score(model, X_tr, y_train, cv=cv, scoring="roc_auc")
        cv_f1 = cross_val_score(model, X_tr, y_train, cv=cv, scoring="f1")

        # 2. Huấn luyện mô hình & Đánh giá trên tập kiểm thử (Test Set)
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        # 3. Tính Feature Importance
        feat_imp = None
        if hasattr(model, "feature_importances_"):
            feat_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            # Đối với Logistic Regression: Hệ số chuẩn hóa (Standardized Coefficients)
            feat_imp = pd.Series(model.coef_[0], index=feature_names).sort_values(ascending=False)

        # 4. Permutation Importance (Đo lường tầm quan trọng độc lập cho mọi mô hình)
        try:
            perm = permutation_importance(model, X_te, y_test, n_repeats=5, random_state=42)
            perm_imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
        except Exception:
            perm_imp = None

        results[name] = {
            "model": model,
            "scaled": cfg["scaled"],
            "accuracy": acc,
            "auc": auc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "fpr": fpr,
            "tpr": tpr,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "report": report,
            "cv_scores": cv_acc,
            "cv_mean": cv_acc.mean(),
            "cv_std": cv_acc.std(),
            "cv_auc_mean": cv_auc.mean(),
            "cv_auc_std": cv_auc.std(),
            "cv_f1_mean": cv_f1.mean(),
            "cv_f1_std": cv_f1.std(),
            "feature_importances": feat_imp,
            "perm_importance": perm_imp,
        }

    return results


# ──────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Điều Hướng")
    page = st.radio(
        "Chọn trang:",
        [
            "🏠 Tổng Quan Dự Án",
            "📊 Khám Phá Dữ Liệu (EDA)",
            "🤖 Huấn Luyện & So Sánh Mô Hình",
            "🩺 Chẩn Đoán Bệnh Nhân Mới",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Cấu Hình Mô Hình")
    test_size = st.slider("Tỉ lệ tập Test (%)", 10, 40, 20, 5) / 100
    random_state = st.number_input("Random Seed", 0, 999, 42)

    st.markdown("---")
    st.markdown(
        """
    ### 📌 Thông Tin Đồ Án
    - **Bộ dữ liệu:** UCI Heart Disease (Cleveland - Dữ liệu thực)
    - **Số thuật toán:** 8 mô hình ML
    - **Kỹ thuật thẩm định:** 5-Fold Stratified CV
    - **Ngôn ngữ:** Python 3.x
    - **Framework:** Streamlit, Scikit-Learn, XGBoost
    """
    )


# ──────────────────────────────────────────────────
# Tải & Tiền xử lý dữ liệu (Toàn cục & Caching)
# ──────────────────────────────────────────────────
@st.cache_data
def get_prepared_data(test_size, random_state):
    df = load_data()
    if df is None:
        return None, None, None, None, None, None, None, None
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


# Gọi hàm lấy dữ liệu
df, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = get_prepared_data(test_size, random_state)

if df is None:
    st.stop()

# Đảm bảo các biến toàn cục cho EDA và Feature Importance
feature_cols = [col for col in df.columns if col != "target"]
X = df[feature_cols]
y = df["target"]

# Huấn luyện tất cả mô hình
results = train_all_models(
    X_train.values,
    X_test.values,
    y_train.values,
    y_test.values,
    X_train_scaled,
    X_test_scaled,
    feature_names=tuple(feature_cols),
)


# ══════════════════════════════════════════════════
# TRANG 1: TỔNG QUAN DỰ ÁN
# ══════════════════════════════════════════════════
if page == "🏠 Tổng Quan Dự Án":
    st.markdown(
        """
    <div class="main-header">
        <h1>🫀 Chẩn Đoán Bệnh Tim Bằng Học Máy (Machine Learning)</h1>
        <p>Ứng dụng 8 thuật toán ML phân loại nguy cơ bệnh tim dựa trên bộ dữ liệu y khoa thực tế UCI Cleveland, tích hợp 5-Fold Cross-Validation và phân tích tầm quan trọng đặc trưng (Feature Importance).</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="value">{df.shape[0]}</div>
            <div class="label">Bệnh nhân thực tế (UCI)</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="value">{df.shape[1] - 1}</div>
            <div class="label">Chỉ số sinh hiệu (Features)</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="value">{len(results)}</div>
            <div class="label">Thuật toán ML so sánh</div>
        </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        best_name = max(results, key=lambda k: results[k]["accuracy"])
        best_acc = results[best_name]["accuracy"]
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="value">{best_acc*100:.1f}%</div>
            <div class="label">Test Accuracy ({best_name})</div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Bảng thuật toán
    st.markdown("### 🤖 8 Thuật Toán Học Máy Được Triển Khai")
    algo_data = {
        "STT": [1, 2, 3, 4, 5, 6, 7, 8],
        "Thuật Toán": [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Support Vector Machine",
            "Random Forest",
            "Naive Bayes",
            "Gradient Boosting",
            "XGBoost",
        ],
        "Ký Hiệu": ["LR", "DT", "KNN", "SVM", "RF", "NB", "GB", "XGB"],
        "Nhóm Thuật Toán": [
            "Tuyến tính",
            "Cây quyết định",
            "Khoảng cách (Distance)",
            "Siêu phẳng Kernel RBF",
            "Bagging Ensemble",
            "Xác suất Bayes",
            "Boosting Ensemble",
            "Extreme Gradient Boosting",
        ],
        "StandardScaler": ["✅ Cần", "❌ Không", "✅ Cần", "✅ Cần", "❌ Không", "✅ Cần", "❌ Không", "❌ Không"],
    }
    st.dataframe(
        pd.DataFrame(algo_data).set_index("STT"),
        use_container_width=True,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Quy trình Machine Learning
    st.markdown("### 🔄 Quy Trình Chuẩn Xử Lý Học Máy Y Tế")
    cols = st.columns(6)
    steps = [
        ("📥", "Dữ liệu Y khoa\nUCI Thực tế", "#667eea"),
        ("🔍", "Khám phá EDA\n& Thống kê", "#764ba2"),
        ("⚙️", "Tiền xử lý &\nStandardScaler", "#f093fb"),
        ("🔄", "5-Fold CV &\nCross-Validate", "#fa709a"),
        ("🧠", "Huấn luyện\n8 Mô hình ML", "#4facfe"),
        ("🩺", "Feature Importance\n& Chẩn đoán", "#43e97b"),
    ]
    for i, (icon, label, color) in enumerate(steps):
        with cols[i]:
            st.markdown(
                f"""
            <div style="text-align:center; padding:1rem; background:white;
                 border-radius:14px; border-top:4px solid {color};
                 box-shadow:0 2px 10px rgba(0,0,0,0.06); min-height:120px;
                 display:flex; flex-direction:column; justify-content:center;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="font-size:0.82rem; font-weight:600; color:#333; white-space:pre-line; margin-top:0.4rem;">{label}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Bảng giải thích Features
    st.markdown("### 📋 Bảng Định Nghĩa 13 Chỉ Số Y Tế (Cleveland Clinical Features)")
    feat_df = pd.DataFrame(
        {
            "Tên Cột": FEATURE_DESCRIPTIONS.keys(),
            "Ý Nghĩa Y Khoa": FEATURE_DESCRIPTIONS.values(),
        }
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════
# TRANG 2: KHÁM PHÁ DỮ LIỆU (EDA)
# ══════════════════════════════════════════════════
elif page == "📊 Khám Phá Dữ Liệu (EDA)":
    st.markdown(
        """
    <div class="main-header">
        <h1>📊 Khám Phá & Phân Tích Dữ Liệu Lâm Sàng (EDA)</h1>
        <p>Trực quan hóa cấu trúc, phân bố và mối tương quan sinh lý trong bộ dữ liệu y khoa thực tế UCI Cleveland</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📋 Tổng Quan Dữ Liệu",
            "📊 Phân Bố Biến",
            "🔗 Ma Trận Tương Quan",
            "📈 Phân Tích Chuyên Sâu",
        ]
    )

    # ── Tab 1: Tổng quan ──
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 📑 Dữ Liệu Lâm Sàng Mẫu (10 dòng đầu)")
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.markdown("#### 📐 Thống Kê Mô Tả Dữ Liệu Thật")
            st.dataframe(df.describe().round(2), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📊 Kiểm Tra Dữ Liệu & Giá Trị Khuyết Thiếu")
        info_df = pd.DataFrame(
            {
                "Thuộc tính": df.columns,
                "Kiểu dữ liệu": df.dtypes.values,
                "Số giá trị Null": df.isnull().sum().values,
                "Số giá trị phân biệt": df.nunique().values,
                "Giá trị nhỏ nhất": df.min().values,
                "Giá trị lớn nhất": df.max().values,
            }
        )
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    # ── Tab 2: Phân bố ──
    with tab2:
        st.markdown("#### 🎯 Phân Bố Nhãn Mục Tiêu (Target - Có Bệnh vs Không Bệnh)")
        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#667eea", "#f45c43"]
            target_counts = df["target"].value_counts()
            bars = ax.bar(
                ["0 - Không bệnh", "1 - Có bệnh tim"],
                target_counts.values,
                color=colors,
                edgecolor="white",
                linewidth=2,
                width=0.45,
            )
            for bar, val in zip(bars, target_counts.values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{val} ({val/len(df)*100:.1f}%)",
                    ha="center",
                    fontweight="bold",
                    fontsize=11,
                )
            ax.set_ylabel("Số lượng bệnh nhân", fontsize=11)
            ax.set_title("Phân bố nhãn mục tiêu (Target)", fontsize=13, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(
                target_counts.values,
                labels=["Không bệnh (0)", "Bệnh tim (1)"],
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                explode=(0.04, 0.04),
                textprops={"fontsize": 11, "fontweight": "bold"},
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            ax.set_title(
                "Tỉ Lệ Mắc Bệnh Tim Trong Bộ Dữ Liệu", fontsize=13, fontweight="bold", pad=15
            )
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.markdown("#### 📊 Phân Bố Từng Đặc Trưng Theo Nhóm Target")
        selected_features = st.multiselect(
            "Chọn các đặc trưng để hiển thị phân bố:",
            options=feature_cols,
            default=["age", "chol", "thalach", "oldpeak"],
        )

        if selected_features:
            n_cols = min(len(selected_features), 3)
            n_rows = (len(selected_features) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
            axes = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]

            for i, feat in enumerate(selected_features):
                ax = axes[i]
                for t, color, label in [
                    (0, "#667eea", "Không bệnh (0)"),
                    (1, "#f45c43", "Bệnh tim (1)"),
                ]:
                    subset = df[df["target"] == t][feat]
                    ax.hist(
                        subset,
                        bins=18,
                        alpha=0.65,
                        color=color,
                        label=label,
                        edgecolor="white",
                    )
                ax.set_title(f"{feat}: {FEATURE_DESCRIPTIONS.get(feat, '')[:32]}...", fontsize=10, fontweight="bold")
                ax.legend(fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)

            for j in range(len(selected_features), len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── Tab 3: Correlation ──
    with tab3:
        st.markdown("#### 🔗 Ma Trận Tương Quan (Correlation Heatmap)")
        fig, ax = plt.subplots(figsize=(11, 8.5))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(
            "Ma Trận Tương Quan Giữa 13 Chỉ Số Sinh Hiệu & Bệnh Tim",
            fontsize=13,
            fontweight="bold",
            pad=18,
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown("#### 🎯 Tương Quan Trực Tiếp Với Nhãn Bệnh Tim (Target)")
        target_corr = corr["target"].drop("target").sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_bar = [
            "#f45c43" if v < 0 else "#43e97b" for v in target_corr.values
        ]
        bars = ax.barh(target_corr.index, target_corr.values, color=colors_bar, edgecolor="white", height=0.6)
        ax.set_xlabel("Hệ số tương quan Pearson với Target", fontsize=11)
        ax.set_title(
            "Mức Độ Tương Quan Của Từng Chỉ Số Sinh Hiệu Với Bệnh Tim",
            fontsize=13,
            fontweight="bold",
        )
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, val in zip(bars, target_corr.values):
            ax.text(
                val + (0.015 if val >= 0 else -0.015),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9,
                fontweight="bold",
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 4: Phân tích chuyên sâu ──
    with tab4:
        st.markdown("#### 📈 Biểu Đồ Hộp (Boxplot) So Sánh Chỉ Số Theo Trạng Thái Bệnh")
        selected = st.selectbox(
            "Chọn đặc trưng y tế để xem Boxplot:",
            feature_cols,
            index=0,
        )
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.boxplot(
            data=df,
            x="target",
            y=selected,
            palette=["#667eea", "#f45c43"],
            ax=ax,
            width=0.4,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0 - Không bệnh", "1 - Có bệnh tim"])
        ax.set_title(
            f"Phân Bố Chỉ Số '{selected}' Theo Nhóm Target",
            fontsize=13,
            fontweight="bold",
        )
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown("#### 🔀 Biểu Đồ Cặp Đặc Trưng (Pairplot)")
        pair_features = st.multiselect(
            "Chọn 2-4 đặc trưng để trực quan hóa mối quan hệ kết hợp:",
            feature_cols,
            default=["age", "thalach", "oldpeak"],
            max_selections=4,
        )
        if len(pair_features) >= 2:
            grid = sns.pairplot(
                df[pair_features + ["target"]],
                hue="target",
                palette=["#667eea", "#f45c43"],
                diag_kind="kde",
                plot_kws={"alpha": 0.65, "s": 35},
            )
            grid.fig.suptitle(
                "Pairplot Mối Quan Hệ Giữa Các Đặc Trưng Sinh Hiệu",
                y=1.02,
                fontsize=13,
                fontweight="bold",
            )
            st.pyplot(grid.fig)
            plt.close("all")
        else:
            st.info("⚠️ Vui lòng chọn ít nhất 2 đặc trưng.")


# ══════════════════════════════════════════════════
# TRANG 3: HUẤN LUYỆN & SO SÁNH MÔ HÌNH
# ══════════════════════════════════════════════════
elif page == "🤖 Huấn Luyện & So Sánh Mô Hình":
    st.markdown(
        """
    <div class="main-header">
        <h1>🤖 Huấn Luyện, Đánh Giá & Thẩm Định 8 Mô Hình Học Máy</h1>
        <p>Đánh giá hiệu năng toàn diện trên tập Test Set, kiểm định chéo 5-Fold Stratified Cross-Validation và phân tích tầm quan trọng của đặc trưng (Feature Importance).</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Bảng So Sánh",
            "🔄 Kiểm Định Chéo (Cross-Validation)",
            "⭐ Tầm Quan Trọng Đặc Trưng (Feature Importance)",
            "📈 Đường Cong ROC",
            "🔢 Confusion Matrix",
            "📝 Báo Cáo Chi Tiết",
        ]
    )

    # ── Tab 1: Bảng so sánh ──
    with tab1:
        st.markdown("#### 📊 Bảng Xếp Hạng Toàn Diện 8 Mô Hình ML")

        compare_data = []
        for name, res in results.items():
            compare_data.append(
                {
                    "Mô Hình": name,
                    "Test Accuracy (%)": round(res["accuracy"] * 100, 2),
                    "5-Fold CV Accuracy (%)": f"{res['cv_mean']*100:.2f}% ± {res['cv_std']*100:.2f}%",
                    "CV Mean (%)": round(res["cv_mean"] * 100, 2),
                    "ROC-AUC": round(res["auc"], 4),
                    "Precision (%)": round(res["precision"] * 100, 2),
                    "Recall (%)": round(res["recall"] * 100, 2),
                    "F1-Score (%)": round(res["f1"] * 100, 2),
                }
            )
        compare_df = (
            pd.DataFrame(compare_data)
            .sort_values("Test Accuracy (%)", ascending=False)
            .reset_index(drop=True)
        )
        compare_df.index = [
            f"🥇 1" if i == 0 else f"🥈 2" if i == 1 else f"🥉 3" if i == 2 else f"  {i+1}"
            for i in range(len(compare_df))
        ]
        compare_df.index.name = "Hạng"

        display_df = compare_df.drop(columns=["CV Mean (%)"])

        st.markdown(
            """
            <div style="display: flex; gap: 20px; font-size: 0.88rem; margin-bottom: 8px;">
                <span>🟢 <strong style="color: #2ecc71;">Xanh lá:</strong> Chỉ số cao nhất (Tốt nhất)</span>
                <span>🔴 <strong style="color: #e74c3c;">Đỏ nhạt:</strong> Chỉ số thấp nhất</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            display_df.style
            .format({
                "Test Accuracy (%)": "{:.2f}%",
                "ROC-AUC": "{:.4f}",
                "Precision (%)": "{:.2f}%",
                "Recall (%)": "{:.2f}%",
                "F1-Score (%)": "{:.2f}%",
            })
            .highlight_max(
                subset=["Test Accuracy (%)", "ROC-AUC", "Precision (%)", "Recall (%)", "F1-Score (%)"],
                props="background-color: rgba(46, 204, 113, 0.25); color: #2ecc71; font-weight: bold;",
            )
            .highlight_min(
                subset=["Test Accuracy (%)", "ROC-AUC", "Precision (%)", "Recall (%)", "F1-Score (%)"],
                props="background-color: rgba(231, 76, 60, 0.25); color: #e74c3c; font-weight: bold;",
            ),
            use_container_width=True,
        )

        st.markdown("---")

        # Bar chart so sánh Test Acc vs CV Mean
        st.markdown("#### 📊 So Sánh Test Accuracy vs 5-Fold CV Mean (Kiểm Tra Overfitting)")
        acc_compare_df = compare_df.melt(
            id_vars=["Mô Hình"],
            value_vars=["Test Accuracy (%)", "CV Mean (%)"],
            var_name="Tập Đánh Giá",
            value_name="Độ Chính Xác (%)",
        )
        acc_compare_df["Tập Đánh Giá"] = acc_compare_df["Tập Đánh Giá"].replace(
            {"Test Accuracy (%)": "Test Set Accuracy", "CV Mean (%)": "5-Fold CV Mean Accuracy"}
        )

        fig, ax = plt.subplots(figsize=(13, 5.5))
        chart = sns.barplot(
            data=acc_compare_df,
            x="Mô Hình",
            y="Độ Chính Xác (%)",
            hue="Tập Đánh Giá",
            palette=["#667eea", "#43e97b"],
            ax=ax,
            edgecolor="white",
        )
        ax.set_title(
            "So Sánh Test Set vs 5-Fold Cross-Validation Accuracy Giữa 8 Thuật Toán",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylim(0, 112)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        for p in chart.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.1f}%",
                    (p.get_x() + p.get_width() / 2, h),
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    xytext=(0, 2),
                    textcoords="offset points",
                )
        ax.legend(loc="upper right", fontsize=9.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # AUC bar chart
        st.markdown("#### 🏅 Xếp Hạng Điểm Diện Tích Dưới Đường Cong ROC-AUC")
        auc_df = compare_df.sort_values("ROC-AUC", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_auc = plt.cm.viridis(np.linspace(0.2, 0.9, len(auc_df)))
        bars = ax.barh(
            auc_df["Mô Hình"],
            auc_df["ROC-AUC"],
            color=colors_auc,
            edgecolor="white",
            height=0.6,
        )
        for bar, val in zip(bars, auc_df["ROC-AUC"].values):
            ax.text(
                val + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9.5,
                fontweight="bold",
            )
        ax.set_xlabel("ROC-AUC Score", fontsize=11)
        ax.set_xlim(0, 1.08)
        ax.set_title("Xếp Hạng Khả Năng Phân Biệt Lớp (ROC-AUC)", fontsize=13, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: Cross-Validation ──
    with tab2:
        st.markdown("### 🔄 Kiểm Định Chéo 5-Fold Stratified Cross-Validation")
        st.markdown(
            """
            > **Mục tiêu khoa học:** Đánh giá trên 1 lần chia ngẫu nhiên (Train/Test) có thể gặp may rủi do phân bố mẫu. 
            > **K-Fold Cross-Validation (K=5)** chia tập dữ liệu thành 5 phần bằng nhau, huấn luyện luân phiên trên 4 phần và kiểm thử trên phần còn lại. 
            > Kết quả trung bình và độ lệch chuẩn ($\mu \pm \sigma$) phản ánh tính **khái quát hóa và độ ổn định thực tế** của từng mô hình.
            """
        )

        # Thống kê nhanh CV
        best_cv_name = max(results, key=lambda k: results[k]["cv_mean"])
        best_cv_score = results[best_cv_name]["cv_mean"] * 100
        best_cv_std = results[best_cv_name]["cv_std"] * 100

        most_stable_name = min(results, key=lambda k: results[k]["cv_std"])
        most_stable_std = results[most_stable_name]["cv_std"] * 100

        col_cv1, col_cv2, col_cv3 = st.columns(3)
        with col_cv1:
            st.metric("🏆 Mô hình có CV Mean cao nhất", f"{best_cv_name}", f"{best_cv_score:.2f}% (± {best_cv_std:.2f}%)")
        with col_cv2:
            st.metric("🎯 Mô hình ổn định nhất (Std thấp nhất)", f"{most_stable_name}", f"Độ lệch ± {most_stable_std:.2f}%")
        with col_cv3:
            st.metric("🔢 Số Fold thẩm định", "5 Folds", "Stratified Split")

        st.markdown("---")

        # Boxplot điểm số của 5 fold cho cả 8 mô hình
        st.markdown("#### 📦 Biểu Đồ Hộp (Boxplot) Phân Bố Điểm Số 5 Fold Của 8 Mô Hình")
        cv_scores_dict = {name: res["cv_scores"] * 100 for name, res in results.items()}
        cv_plot_df = pd.DataFrame(cv_scores_dict)

        fig, ax = plt.subplots(figsize=(13, 6))
        palette_list = [MODEL_COLORS.get(col, "#667eea") for col in cv_plot_df.columns]
        sns.boxplot(data=cv_plot_df, palette=palette_list, width=0.45, ax=ax)
        sns.stripplot(data=cv_plot_df, color="black", alpha=0.6, jitter=0.15, size=6, ax=ax)
        ax.set_title("Độ Biến Thiên Của Độ Chính Xác (Accuracy %) Qua 5 Folds", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # Bảng chi tiết từng fold
        st.markdown("#### 📋 Chi Tiết Điểm Số Từng Lần Kiểm Định (Fold 1 -> Fold 5)")
        fold_table = []
        for name, res in results.items():
            scores = res["cv_scores"] * 100
            fold_table.append(
                {
                    "Mô Hình": name,
                    "Fold 1 (%)": round(scores[0], 2),
                    "Fold 2 (%)": round(scores[1], 2),
                    "Fold 3 (%)": round(scores[2], 2),
                    "Fold 4 (%)": round(scores[3], 2),
                    "Fold 5 (%)": round(scores[4], 2),
                    "CV Mean (%)": round(res["cv_mean"] * 100, 2),
                    "Độ lệch chuẩn Std (±)": round(res["cv_std"] * 100, 2),
                    "CV ROC-AUC": round(res["cv_auc_mean"], 4),
                    "CV F1-Score (%)": round(res["cv_f1_mean"] * 100, 2),
                }
            )
        fold_df = pd.DataFrame(fold_table).sort_values("CV Mean (%)", ascending=False).reset_index(drop=True)
        fold_df.index = [
            f"🥇 1" if i == 0 else f"🥈 2" if i == 1 else f"🥉 3" if i == 2 else f"  {i+1}"
            for i in range(len(fold_df))
        ]
        fold_df.index.name = "Hạng"
        st.dataframe(
            fold_df.style
            .format({
                "Fold 1 (%)": "{:.2f}%",
                "Fold 2 (%)": "{:.2f}%",
                "Fold 3 (%)": "{:.2f}%",
                "Fold 4 (%)": "{:.2f}%",
                "Fold 5 (%)": "{:.2f}%",
                "CV Mean (%)": "{:.2f}%",
                "Độ lệch chuẩn Std (±)": "± {:.2f}%",
                "CV ROC-AUC": "{:.4f}",
                "CV F1-Score (%)": "{:.2f}%",
            })
            .highlight_max(
                subset=["CV Mean (%)", "CV ROC-AUC", "CV F1-Score (%)"],
                props="background-color: rgba(46, 204, 113, 0.25); color: #2ecc71; font-weight: bold;",
            )
            .highlight_min(
                subset=["CV Mean (%)", "CV ROC-AUC", "CV F1-Score (%)"],
                props="background-color: rgba(231, 76, 60, 0.25); color: #e74c3c; font-weight: bold;",
            ),
            use_container_width=True,
        )

    # ── Tab 3: Feature Importance ──
    with tab3:
        st.markdown("### ⭐ Tầm Quan Trọng Của Các Chỉ Số Y Tế (Feature Importance)")
        st.markdown(
            """
            > **Ý nghĩa lâm sàng:** Trong chẩn đoán y tế, mô hình Machine Learning không chỉ cần độ chính xác cao mà còn cần **tính minh bạch và giải thích được (Explainable AI - XAI)**.
            > Tính năng này bóc tách trọng số quyết định của từng chỉ số tim mạch đối với kết luận chẩn đoán.
            """
        )

        # 1. Consensus Feature Importance từ các mô hình cây kết hợp (Ensemble Consensus)
        tree_models = ["Random Forest", "XGBoost", "Gradient Boosting"]
        ensemble_imp = pd.DataFrame(index=feature_cols)
        for tm in tree_models:
            if tm in results and results[tm]["feature_importances"] is not None:
                ensemble_imp[tm] = results[tm]["feature_importances"]

        if not ensemble_imp.empty:
            ensemble_imp["Consensus Mean"] = ensemble_imp.mean(axis=1)
            ensemble_imp = ensemble_imp.sort_values("Consensus Mean", ascending=True)

            st.markdown("#### 🌟 1. Đồng Thuận Trọng Số Từ Các Mô Hình Ensemble (RF + GB + XGB)")
            fig, ax = plt.subplots(figsize=(11, 6))
            colors_fi = plt.cm.magma(np.linspace(0.3, 0.85, len(ensemble_imp)))
            bars = ax.barh(
                [f"{col} ({FEATURE_DESCRIPTIONS.get(col, '')[:20]}...)" for col in ensemble_imp.index],
                ensemble_imp["Consensus Mean"] * 100,
                color=colors_fi,
                edgecolor="white",
                height=0.6,
            )
            for bar, val in zip(bars, ensemble_imp["Consensus Mean"].values * 100):
                ax.text(
                    val + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                )
            ax.set_xlabel("Mức độ quan trọng trung bình (%)", fontsize=11)
            ax.set_title("Top Chỉ Số Tim Mạch Có Ảnh Hưởng Lớn Nhất Đến Kết Quả Chẩn Đoán", fontsize=13, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # 2. Chi tiết theo từng mô hình cụ thể
        st.markdown("#### 🔍 2. Phân Tích Feature Importance Theo Từng Thuật Toán Riêng Biệt")
        selected_model_fi = st.selectbox(
            "Chọn mô hình để xem trọng số đặc trưng:",
            [
                "Random Forest (Gini Importance)",
                "XGBoost (Gain Importance)",
                "Gradient Boosting (Feature Importance)",
                "Decision Tree (Split Importance)",
                "Logistic Regression (Hệ số tác động Coefficients)",
            ],
        )

        model_key_map = {
            "Random Forest (Gini Importance)": "Random Forest",
            "XGBoost (Gain Importance)": "XGBoost",
            "Gradient Boosting (Feature Importance)": "Gradient Boosting",
            "Decision Tree (Split Importance)": "Decision Tree",
            "Logistic Regression (Hệ số tác động Coefficients)": "Logistic Regression",
        }
        chosen_key = model_key_map[selected_model_fi]
        fi_series = results[chosen_key]["feature_importances"]

        if fi_series is not None:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            if chosen_key == "Logistic Regression":
                # Đối với Logistic Regression: Hệ số âm (giảm nguy cơ) và dương (tăng nguy cơ)
                fi_sorted = fi_series.sort_values(ascending=True)
                colors_lr = ["#11998e" if v < 0 else "#eb3349" for v in fi_sorted.values]
                bars = ax.barh(fi_sorted.index, fi_sorted.values, color=colors_lr, edgecolor="white", height=0.6)
                ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
                ax.set_xlabel("Hệ số hồi quy chuẩn hóa (Standardized Coefficient)", fontsize=11)
                ax.set_title(f"Hệ Số Tác Động Tuyến Tính - {chosen_key} (Đỏ: Tăng nguy cơ, Xanh: Giảm nguy cơ)", fontsize=12, fontweight="bold")
                for bar, val in zip(bars, fi_sorted.values):
                    ax.text(
                        val + (0.04 if val >= 0 else -0.04),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}",
                        va="center",
                        ha="left" if val >= 0 else "right",
                        fontsize=9,
                        fontweight="bold",
                    )
            else:
                fi_sorted = fi_series.sort_values(ascending=True)
                bars = ax.barh(fi_sorted.index, fi_sorted.values * 100, color=MODEL_COLORS.get(chosen_key, "#667eea"), edgecolor="white", height=0.6)
                ax.set_xlabel("Tỷ lệ đóng góp (%)", fontsize=11)
                ax.set_title(f"Tầm Quan Trọng Đặc Trưng - {chosen_key}", fontsize=12, fontweight="bold")
                for bar, val in zip(bars, fi_sorted.values * 100):
                    ax.text(
                        val + 0.3,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%",
                        va="center",
                        fontsize=9,
                        fontweight="bold",
                    )
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # 3. Phân tích ý nghĩa y khoa (Medical Clinical Insights)
        st.markdown("#### 🩺 3. Giải Thích Ý Nghĩa Y Khoa Của Các Chỉ Số Hàng Đầu")
        st.markdown(
            """
            <div class="insight-card">
                <h4>💔 1. cp (Loại đau ngực - Chest Pain Type)</h4>
                <p>Là chỉ báo lâm sàng quan trọng hàng đầu. Trong y khoa thực tế, cơn đau thắt ngực (Angina) xuất hiện do cơ tim bị thiếu máu cục bộ. Những bệnh nhân đau thắt ngực không triệu chứng hoặc điển hình có tỷ lệ tắc nghẽn động mạch vành rất cao.</p>
            </div>
            <div class="insight-card">
                <h4>❤️ 2. thalach (Nhịp tim tối đa khi gắng sức - Max Heart Rate)</h4>
                <p>Khả năng tăng nhịp tim khi gắng sức phản ánh chức năng co bóp và sức bền của cơ tim. Bệnh nhân có nhịp tim tối đa thấp khi gắng sức thường có liên quan chặt chẽ tới suy giảm chức năng tim và thiếu máu cơ tim.</p>
            </div>
            <div class="insight-card">
                <h4>📐 3. oldpeak (ST Depression lúc gắng sức)</h4>
                <p>Độ chênh xuống của đoạn ST trên điện tâm đồ (ECG) khi nghiệm pháp gắng sức là tiêu chuẩn vàng phi xâm lấn để chẩn đoán thiếu máu dưới nội tâm mạc.</p>
            </div>
            <div class="insight-card">
                <h4>🩸 4. ca (Số nhánh mạch máu chính soi thấy - Major Vessels Colored)</h4>
                <p>Được xác định qua chụp mạch huỳnh quang. Số lượng mạch máu bị cản quang phản ánh trực tiếp mức độ xơ vữa động mạch vành của bệnh nhân.</p>
            </div>
            <div class="insight-card">
                <h4>🔬 5. thal (Kết quả xạ hình tưới máu Thallium)</h4>
                <p>Phản ánh khuyết tật tưới máu cơ tim có thể hồi phục (Reversible Defect) hoặc khuyết tật cố định do sẹo nhồi máu cũ.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Tab 4: ROC Curves ──
    with tab4:
        st.markdown("#### 📈 Đường Cong ROC (ROC Curves) - So Sánh Cả 8 Mô Hình")
        fig, ax = plt.subplots(figsize=(10.5, 7.5))
        for name, res in results.items():
            color = MODEL_COLORS.get(name, "#333")
            ax.plot(
                res["fpr"],
                res["tpr"],
                label=f'{name} (AUC = {res["auc"]:.3f})',
                color=color,
                linewidth=2.2,
            )
        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            alpha=0.4,
            linewidth=1.5,
            label="Random Guess (AUC = 0.500)",
        )
        ax.fill_between([0, 1], [0, 1], alpha=0.03, color="gray")
        ax.set_xlabel("False Positive Rate (FPR - Tỷ lệ dương tính giả)", fontsize=11)
        ax.set_ylabel("True Positive Rate (TPR - Độ nhạy / Recall)", fontsize=11)
        ax.set_title(
            "Đường Cong ROC - Khả Năng Phân Biệt Nguy Cơ Bệnh Tim Của 8 Thuật Toán",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.legend(loc="lower right", fontsize=9.2, framealpha=0.9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 5: Confusion Matrix ──
    with tab5:
        st.markdown("#### 🔢 Ma Trận Nhầm Lẫn (Confusion Matrix) Của Cả 8 Mô Hình")
        model_names = list(results.keys())
        n_models = len(model_names)
        n_cols_cm = 4
        n_rows_cm = (n_models + n_cols_cm - 1) // n_cols_cm

        fig, axes = plt.subplots(
            n_rows_cm, n_cols_cm, figsize=(4.2 * n_cols_cm, 3.8 * n_rows_cm)
        )
        axes = axes.flatten()

        for i, name in enumerate(model_names):
            ax = axes[i]
            cm = results[name]["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["0", "1"],
                yticklabels=["0", "1"],
                cbar=False,
                annot_kws={"size": 13, "fontweight": "bold"},
            )
            acc = results[name]["accuracy"]
            ax.set_title(
                f"{name}\nAcc: {acc*100:.1f}%",
                fontsize=9.5,
                fontweight="bold",
            )
            ax.set_xlabel("Dự đoán", fontsize=8.5)
            ax.set_ylabel("Thực tế", fontsize=8.5)

        for j in range(n_models, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            "Confusion Matrix Trên Tập Kiểm Thử (0: Không bệnh, 1: Có bệnh tim)",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Tab 6: Báo cáo chi tiết ──
    with tab6:
        st.markdown("#### 📝 Báo Cáo Phân Lớp (Classification Report) Chi Tiết")
        selected_model = st.selectbox(
            "Chọn mô hình để xem hồ sơ chi tiết:",
            list(results.keys()),
        )

        res = results[selected_model]
        color = MODEL_COLORS.get(selected_model, "#667eea")

        col1, col2, col3, col4, col5 = st.columns(5)
        metrics_list = [
            ("Test Accuracy", res["accuracy"]),
            ("ROC-AUC", res["auc"]),
            ("Precision", res["precision"]),
            ("Recall", res["recall"]),
            ("F1-Score", res["f1"]),
        ]
        for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics_list):
            with col:
                st.metric(label, f"{value*100:.2f}%")

        st.markdown(
            f"**5-Fold Cross-Validation:** Accuracy trung bình = **{res['cv_mean']*100:.2f}% (± {res['cv_std']*100:.2f}%)** | ROC-AUC trung bình = **{res['cv_auc_mean']:.4f}**"
        )
        st.markdown("---")

        report = res["report"]
        report_df = pd.DataFrame(report).transpose()
        st.markdown(f"**Classification Report - {selected_model}:**")
        st.dataframe(report_df.round(4), use_container_width=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**Confusion Matrix - {selected_model}:**")
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            sns.heatmap(
                res["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Âm tính (0)", "Dương tính (1)"],
                yticklabels=["Âm tính (0)", "Dương tính (1)"],
                annot_kws={"size": 15, "fontweight": "bold"},
            )
            ax.set_xlabel("Dự đoán (Predicted)", fontsize=10)
            ax.set_ylabel("Thực tế (Actual)", fontsize=10)
            ax.set_title(selected_model, fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown(f"**ROC Curve - {selected_model}:**")
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.plot(
                res["fpr"],
                res["tpr"],
                color=color,
                linewidth=2.5,
                label=f'AUC = {res["auc"]:.3f}',
            )
            ax.fill_between(res["fpr"], res["tpr"], alpha=0.15, color=color)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set_xlabel("FPR", fontsize=10)
            ax.set_ylabel("TPR", fontsize=10)
            ax.set_title(selected_model, fontsize=12, fontweight="bold")
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ══════════════════════════════════════════════════
# TRANG 4: CHẨN ĐOÁN BỆNH NHÂN MỚI
# ══════════════════════════════════════════════════
elif page == "🩺 Chẩn Đoán Bệnh Nhân Mới":
    st.markdown(
        """
    <div class="main-header">
        <h1>🩺 Chẩn Đoán Nguy Cơ Bệnh Tim Cho Bệnh Nhân Mới</h1>
        <p>Nhập 13 chỉ số sinh hiệu y khoa của bệnh nhân để dự đoán nguy cơ mắc bệnh tim bằng 8 thuật toán ML và biểu quyết đa số (Majority Voting Ensemble).</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Nhập Chỉ Số Sinh Hiệu Bệnh Nhân")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("🎂 Tuổi", min_value=1, max_value=120, value=55, step=1)
        sex = st.selectbox("👤 Giới tính", options=[("Nam", 1), ("Nữ", 0)], format_func=lambda x: x[0])
        cp = st.selectbox(
            "💔 Loại đau ngực (cp)",
            options=[
                ("0 - Đau thắt ngực điển hình (Typical Angina)", 0),
                ("1 - Đau thắt ngực không điển hình (Atypical Angina)", 1),
                ("2 - Đau không do tim (Non-anginal Pain)", 2),
                ("3 - Không có triệu chứng (Asymptomatic)", 3),
            ],
            format_func=lambda x: x[0],
        )
        trestbps = st.slider("💉 Huyết áp lúc nghỉ (mm Hg)", 80, 220, 130)
        chol = st.slider("🧪 Cholesterol (mg/dl)", 100, 580, 240)

    with col2:
        fbs = st.selectbox(
            "🍬 Đường huyết lúc đói > 120 mg/dl",
            options=[("Không", 0), ("Có", 1)],
            format_func=lambda x: x[0],
        )
        restecg = st.selectbox(
            "📉 Điện tâm đồ lúc nghỉ (restecg)",
            options=[
                ("0 - Bình thường", 0),
                ("1 - Bất thường sóng ST-T", 1),
                ("2 - Phì đại thất trái", 2),
            ],
            format_func=lambda x: x[0],
        )
        thalach = st.slider("❤️ Nhịp tim tối đa khi gắng sức (bpm)", 60, 220, 150)
        exang = st.selectbox(
            "🏃 Đau thắt ngực khi gắng sức",
            options=[("Không", 0), ("Có", 1)],
            format_func=lambda x: x[0],
        )

    with col3:
        oldpeak = st.slider("📐 ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
        slope = st.selectbox(
            "📊 Độ dốc đoạn ST (slope)",
            options=[
                ("0 - Dốc lên (Upsloping)", 0),
                ("1 - Phẳng (Flat)", 1),
                ("2 - Dốc xuống (Downsloping)", 2),
            ],
            format_func=lambda x: x[0],
        )
        ca = st.selectbox("🩸 Số mạch máu chính được soi thấy (ca)", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "🔬 Xạ hình tưới máu tim (thal)",
            options=[
                ("1 - Khiếm khuyết cố định (Fixed Defect)", 1),
                ("2 - Bình thường (Normal)", 2),
                ("3 - Khiếm khuyết có thể hồi phục (Reversible Defect)", 3),
            ],
            format_func=lambda x: x[0],
            index=1,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Nút chẩn đoán
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        diagnose = st.button(
            "🔍  CHẨN ĐOÁN NGAY VỚI 8 MÔ HÌNH",
            use_container_width=True,
            type="primary",
        )

    if diagnose:
        patient_data = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex[1],
                    "cp": cp[1],
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": fbs[1],
                    "restecg": restecg[1],
                    "thalach": thalach,
                    "exang": exang[1],
                    "oldpeak": oldpeak,
                    "slope": slope[1],
                    "ca": ca,
                    "thal": thal[1],
                }
            ]
        )

        # Cảnh báo an toàn dữ liệu đầu vào (Input Boundary Check)
        warn_msgs = []
        if chol > 350:
            warn_msgs.append(f"⚠️️ **Cholesterol rất cao ({chol} mg/dl):** Vượt ngưỡng bình thường (< 200 mg/dl).")
        if trestbps > 180:
            warn_msgs.append(f"⚠️ **Huyết áp lúc nghỉ rất cao ({trestbps} mm Hg):** Nguy cơ cơn tăng huyết áp cấp cứu.")
        if thalach > 205:
            warn_msgs.append(f"⚠️ **Nhịp tim tối đa rất cao ({thalach} bpm):** Cần theo dõi nhịp nhanh.")

        if warn_msgs:
            st.warning("### 🩺 Cảnh Báo Chỉ Số Sinh Hiệu Bất Thường")
            for w in warn_msgs:
                st.write(w)
            st.info("💡 *Lưu ý: Chỉ số trên vẫn được đưa vào mô hình để suy luận, nhưng người bệnh cần được kiểm tra trực tiếp bởi bác sĩ.*")

        patient_scaled = scaler.transform(patient_data)

        st.markdown("### 🔬 Kết Quả Chẩn Đoán Từ 8 Thuật Toán ML")

        # Tổng hợp kết quả từ tất cả mô hình
        all_predictions = {}
        for name, res in results.items():
            model = res["model"]
            if res["scaled"]:
                pred = model.predict(patient_scaled)[0]
                prob = model.predict_proba(patient_scaled)[0][1]
            else:
                pred = model.predict(patient_data)[0]
                prob = model.predict_proba(patient_data)[0][1]
            all_predictions[name] = {"prediction": pred, "probability": prob}

        # Hiển thị cards kết quả
        cols = st.columns(4)
        for i, (name, pred_info) in enumerate(all_predictions.items()):
            with cols[i % 4]:
                prob = pred_info["probability"]
                is_sick = pred_info["prediction"] == 1
                color = MODEL_COLORS.get(name, "#667eea")
                icon = "⚠️" if is_sick else "✅"
                status = "Nguy cơ" if is_sick else "An toàn"

                st.markdown(
                    f"""
                <div class="model-card" style="border-left-color: {color};">
                    <h3 style="color: {color};">{icon} {name}</h3>
                    <div style="font-size: 1.6rem; font-weight: 800; color: {'#eb3349' if is_sick else '#11998e'};">
                        {prob*100:.1f}%
                    </div>
                    <div style="font-size: 0.85rem; color: #777; margin-top: 0.3rem;">
                        Kết luận: <strong>{status}</strong>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Kết luận biểu quyết đa số (Majority Voting)
        positive_count = sum(1 for p in all_predictions.values() if p["prediction"] == 1)
        avg_prob = np.mean([p["probability"] for p in all_predictions.values()])
        majority_positive = positive_count > len(all_predictions) / 2

        col1, col2 = st.columns([2, 1])

        with col1:
            if majority_positive:
                st.markdown(
                    f"""
                <div class="pred-danger">
                    <h2>⚠️ CẢNH BÁO: CÓ NGUY CƠ MẮC BỆNH TIM</h2>
                    <p>Xác suất trung bình: <strong>{avg_prob*100:.1f}%</strong> | Biểu quyết: <strong>{positive_count}/{len(all_predictions)}</strong> mô hình chẩn đoán Dương tính</p>
                    <p style="margin-top: 0.8rem; font-size: 0.92rem;">⚕️ Khuyến nghị: Bệnh nhân nên đến cơ sở y tế chuyên khoa tim mạch để thực hiện chụp chiếu và can thiệp kịp thời.</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="pred-safe">
                    <h2>✅ AN TOÀN: NGUY CƠ BỆNH TIM THẤP</h2>
                    <p>Xác suất trung bình: <strong>{avg_prob*100:.1f}%</strong> | Biểu quyết: <strong>{positive_count}/{len(all_predictions)}</strong> mô hình chẩn đoán Dương tính</p>
                    <p style="margin-top: 0.8rem; font-size: 0.92rem;">💚 Khuyến nghị: Duy trì chế độ dinh dưỡng lành mạnh, vận động thường xuyên và kiểm tra sức khỏe định kỳ.</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4.2))
            probs = [p["probability"] * 100 for p in all_predictions.values()]
            names = [n.replace(" Classifier", "").replace(" Regression", "")[:12] for n in all_predictions.keys()]
            colors_prob = ["#eb3349" if p > 50 else "#43e97b" for p in probs]
            bars = ax.barh(names, probs, color=colors_prob, edgecolor="white", height=0.6)
            ax.set_xlim(0, 105)
            ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
            ax.set_xlabel("Xác suất bệnh tim (%)", fontsize=9.5)
            ax.set_title("So Sánh Dự Đoán", fontsize=11, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            for bar, val in zip(bars, probs):
                ax.text(
                    val + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.markdown("#### 📋 Hồ Sơ Sinh Hiệu Bệnh Nhân Đã Nhập")
        patient_display = patient_data.copy()
        patient_display.columns = [
            f"{col} ({FEATURE_DESCRIPTIONS.get(col, '')[:18]}...)" for col in patient_display.columns
        ]
        st.dataframe(patient_display, use_container_width=True, hide_index=True)

        st.info(
            "⚠️ **Lưu ý y khoa:** Ứng dụng này phục vụ mục đích nghiên cứu học thuật và hỗ trợ chẩn đoán tham khảo dựa trên thuật toán Học máy. "
            "Kết quả không thay thế chẩn đoán lâm sàng của bác sĩ chuyên khoa tim mạch."
        )
