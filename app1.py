import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="House Price Predictor", page_icon="🏠",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0F172A;color:#F1F5F9;}
[data-testid="stSidebar"]{background:#1E293B!important;border-right:1px solid #334155;}
[data-testid="stSidebar"] *{color:#F1F5F9!important;}
.block-container{padding:2rem 2.5rem;}
.card{background:#1E293B;border:1px solid #334155;border-radius:12px;padding:1.5rem;margin-bottom:1.25rem;}
.mcard{background:#0F172A;border:1px solid #334155;border-radius:10px;padding:1rem 1.25rem;text-align:center;}
.mlabel{font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;}
.mvalue{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:600;color:#F1F5F9;}
.mvalue.teal{color:#1D9E75;} .mvalue.blue{color:#60A5FA;} .mvalue.amber{color:#F59E0B;} .mvalue.red{color:#EF4444;}
.sec-title{font-size:18px;font-weight:600;color:#F1F5F9;margin-bottom:4px;}
.sec-sub{font-size:13px;color:#64748B;margin-bottom:1.25rem;}
.step-item{display:flex;align-items:center;gap:12px;padding:10px 14px;border-radius:8px;margin-bottom:4px;}
.step-item.active{background:#0F172A;border:1px solid #1D9E75;}
.step-item.done{background:#0F172A;border:1px solid #334155;}
.step-item.idle{background:transparent;border:1px solid transparent;}
.sdot{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;flex-shrink:0;}
.sdot.active{background:#1D9E75;color:#fff;} .sdot.done{background:#134E3A;color:#1D9E75;} .sdot.idle{background:#1E293B;color:#64748B;border:1px solid #334155;}
.slabel{font-size:13px;font-weight:500;}
.slabel.active{color:#F1F5F9;} .slabel.done{color:#94A3B8;} .slabel.idle{color:#475569;}
.badge{display:inline-block;font-size:11px;font-weight:500;padding:2px 9px;border-radius:20px;}
.badge-blue{background:#1E3A5F;color:#60A5FA;} .badge-amber{background:#3B2A0F;color:#F59E0B;}
.badge-red{background:#3B0F0F;color:#F87171;} .badge-green{background:#0F3B27;color:#34D399;} .badge-gray{background:#1E293B;color:#94A3B8;border:1px solid #334155;}
.atable{width:100%;border-collapse:collapse;font-size:13px;}
.atable th{text-align:left;padding:10px 12px;color:#64748B;font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #334155;}
.atable td{padding:10px 12px;border-bottom:1px solid #1E293B;color:#CBD5E1;vertical-align:middle;}
.atable tr.issue td:first-child{border-left:3px solid #EF4444;}
.log-box{background:#0F172A;border:1px solid #334155;border-radius:8px;padding:1rem 1.25rem;font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.9;max-height:220px;overflow-y:auto;}
.log-ok{color:#34D399;} .log-warn{color:#F59E0B;} .log-info{color:#94A3B8;}
.price-result{background:#0F3B27;border:1px solid #1D9E75;border-radius:14px;padding:2rem;text-align:center;margin-top:1.25rem;}
.plabel{font-size:13px;color:#94A3B8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;}
.pvalue{font-family:'JetBrains Mono',monospace;font-size:42px;font-weight:700;color:#1D9E75;}
.prange{font-size:13px;color:#64748B;margin-top:6px;}
.stButton>button{background:#1D9E75!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:.55rem 1.4rem!important;}
.stButton>button:hover{opacity:.88!important;}
[data-testid="stFileUploader"]{background:#1E293B;border:2px dashed #334155;border-radius:12px;}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
DEFAULTS_STATE = {
    "step": 1, "df_raw": None, "df_clean": None,
    "pipeline": None, "feat_imp": None, "eval_data": None,
    "feat_cols": None, "clean_log": [], "log_transform": True, "target": "price_usd",
}
for k, v in DEFAULTS_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────────────────────
def classify_cols(df, target):
    num_cols, cat_cols = [], []
    for c in df.columns:
        if c == target: continue
        if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8:
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols

def mcard(label, value, cls=""):
    return f'<div class="mcard"><div class="mlabel">{label}</div><div class="mvalue {cls}">{value}</div></div>'

def badge(text, kind="gray"):
    return f'<span class="badge badge-{kind}">{text}</span>'

BOOL_COLS  = {"has_basement","has_pool","renovated"}
INT_COLS   = {"bedrooms","bathrooms","floors","garage_cars","year_built",
              "house_age_years","condition_score","quality_grade","basement_sqft","hoa_monthly_usd"}
INT_RANGES = {
    "bedrooms":(1,20,1),"bathrooms":(1,15,1),"floors":(1,10,1),
    "garage_cars":(0,10,1),"year_built":(1900,2025,1),"house_age_years":(0,125,1),
    "condition_score":(1,5,1),"quality_grade":(1,10,1),
    "basement_sqft":(0,5000,1),"hoa_monthly_usd":(0,2000,1),
}
# (min, max, step, format)
FLOAT_FIELDS = {
    "sqft_living":           (100.0,   20000.0,  10.0,  "%.0f"),
    "sqft_lot":              (500.0,   200000.0, 100.0, "%.0f"),
    "school_rating":         (0.0,     10.0,     0.1,   "%.1f"),
    "crime_rate_per_1000":   (0.0,     200.0,    0.5,   "%.1f"),
    "dist_city_center_km":   (0.0,     200.0,    0.5,   "%.1f"),
    "dist_school_km":        (0.0,     50.0,     0.1,   "%.1f"),
    "dist_hospital_km":      (0.0,     100.0,    0.5,   "%.1f"),
    "property_tax_rate_pct": (0.0,     10.0,     0.05,  "%.2f"),
    "market_trend":          (-1.0,    1.0,      0.005, "%.3f"),
}
PREDICT_DEFAULTS = {
    "sqft_living":1800,"sqft_lot":7500,"bedrooms":3,"bathrooms":2,"floors":1,
    "has_basement":0,"basement_sqft":0,"garage_cars":2,"has_pool":0,
    "year_built":2005,"house_age_years":19,"renovated":0,"condition_score":3,
    "quality_grade":7,"school_rating":7.0,"crime_rate_per_1000":15.0,
    "dist_city_center_km":10.0,"dist_school_km":1.5,"dist_hospital_km":5.0,
    "property_tax_rate_pct":1.2,"hoa_monthly_usd":100,"market_trend":0.02,
}

# ── Sidebar stepper ────────────────────────────────────────────────────────────
STEPS = ["Upload CSV","Data Audit","Clean Data","Train Model","Predict"]
with st.sidebar:
    st.markdown('<div style="padding:1rem 0 1.5rem"><div style="font-size:20px;font-weight:700;color:#F1F5F9">🏠 HouseML</div><div style="font-size:12px;color:#64748B;margin-top:2px">Price Prediction Pipeline</div></div>', unsafe_allow_html=True)
    for i, label in enumerate(STEPS, 1):
        s = "active" if i == st.session_state.step else "done" if i < st.session_state.step else "idle"
        dot = "✓" if s == "done" else str(i)
        st.markdown(f'<div class="step-item {s}"><div class="sdot {s}">{dot}</div><div class="slabel {s}">{label}</div></div>', unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:1px solid #334155;margin:1.5rem 0">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#475569;padding:0 4px">Linear Regression · scikit-learn</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown('<div class="sec-title">Upload your dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Upload a CSV file with house data to get started</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state.df_raw = df
        st.markdown(f'<div class="card" style="padding:.75rem 1.25rem"><div style="display:flex;gap:2rem;align-items:center"><span style="color:#94A3B8;font-size:13px">File</span><span style="font-weight:500;font-size:13px">{uploaded.name}</span><span style="color:#94A3B8;font-size:13px">Rows</span><span style="font-weight:600;color:#1D9E75;font-size:13px">{df.shape[0]:,}</span><span style="color:#94A3B8;font-size:13px">Columns</span><span style="font-weight:600;color:#1D9E75;font-size:13px">{df.shape[1]}</span></div></div>', unsafe_allow_html=True)
        st.dataframe(df.head(8), use_container_width=True, height=260)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue to Audit →"):
            st.session_state.step = 2
            st.rerun()
    else:
        st.markdown('<div class="card" style="text-align:center;padding:3rem"><div style="font-size:32px;margin-bottom:12px">📂</div><div style="font-size:15px;font-weight:500;margin-bottom:6px">Drop your CSV file here</div><div style="font-size:13px;color:#64748B">or use the uploader above</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — AUDIT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    df = st.session_state.df_raw
    n = len(df)
    st.markdown('<div class="sec-title">Data Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Review your raw dataset before cleaning</div>', unsafe_allow_html=True)

    total_missing = int(df.isna().sum().sum())
    dups = int(df.duplicated().sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(mcard("Total Rows", f"{n:,}", "teal"), unsafe_allow_html=True)
    c2.markdown(mcard("Columns", str(df.shape[1]), "blue"), unsafe_allow_html=True)
    c3.markdown(mcard("Missing Cells", f"{total_missing:,}", "amber" if total_missing else "teal"), unsafe_allow_html=True)
    c4.markdown(mcard("Duplicate Rows", str(dups), "red" if dups else "teal"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    report, rows_html = {}, ""
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        pct = missing / n * 100
        uniq = int(s.nunique())
        is_num = pd.api.types.is_numeric_dtype(s)
        outliers = 0
        if is_num and s.dropna().shape[0]:
            q1,q3 = s.quantile(.25), s.quantile(.75)
            outliers = int(((s < q1-1.5*(q3-q1)) | (s > q3+1.5*(q3-q1))).sum())
        report[col] = {"missing": missing, "outliers": outliers, "is_num": is_num}
        has_issue = missing > 0 or outliers > 0
        tb = badge("numeric","blue") if is_num else badge("categorical","amber")
        mb = badge(f"{missing} ({pct:.1f}%)","red" if missing else "green")
        ob = badge(str(outliers),"amber" if outliers else "gray")
        rows_html += f'<tr class="{"issue" if has_issue else ""}"><td style="font-weight:500;color:#F1F5F9">{col}</td><td>{tb}</td><td>{mb}</td><td style="font-family:monospace">{uniq:,}</td><td>{ob}</td></tr>'
    st.session_state.audit_report = report

    st.markdown(f'<div class="card" style="padding:0;overflow:hidden"><table class="atable"><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Outliers</th></tr></thead><tbody>{rows_html}</tbody></table></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2, _ = st.columns([1,1,5])
    with b1:
        if st.button("← Back"):
            st.session_state.step = 1; st.rerun()
    with b2:
        if st.button("Continue to Clean →"):
            st.session_state.step = 3; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CLEAN
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    df = st.session_state.df_raw
    st.markdown('<div class="sec-title">Clean Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Configure cleaning strategies, then run the pipeline</div>', unsafe_allow_html=True)

    # All widgets at top level — NO nesting inside columns for key settings
    target_opts = list(df.columns)
    default_target_idx = target_opts.index("price_usd") if "price_usd" in target_opts else 0

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Missing values**")
    cl1, cl2 = st.columns(2)
    num_strat    = cl1.selectbox("Numeric columns",      ["Median fill","Mean fill","Zero fill","Drop rows"])
    cat_strat    = cl2.selectbox("Categorical columns",  ["Mode fill","Fill 'Unknown'","Drop rows"])
    st.markdown("**Outlier handling**")
    outlier_method = st.selectbox("Method", ["IQR clipping (1.5×)","Z-score clipping (±3σ)","No handling"])
    st.markdown("**Target column**")
    cl3, cl4 = st.columns(2)
    target       = cl3.selectbox("Price column", target_opts, index=default_target_idx)
    log_transform = cl4.selectbox("Log-transform target", ["Yes — log1p (recommended)","No"], index=0) == "Yes — log1p (recommended)"
    st.markdown("**Drop columns** (optional)")
    drop_cols    = st.multiselect("Columns to exclude from training", [c for c in df.columns if c != target], default=[])
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("▶  Run Cleaning"):
        logs, df2 = [], df.copy()
        orig = len(df2)

        if drop_cols:
            df2.drop(columns=drop_cols, inplace=True)
            logs.append(("info", f"Dropped columns: {', '.join(drop_cols)}"))

        before = len(df2); df2 = df2[df2[target].notna()]
        if before-len(df2): logs.append(("warn", f"Dropped {before-len(df2):,} rows — missing target"))

        before = len(df2); df2.drop_duplicates(inplace=True)
        if before-len(df2): logs.append(("warn", f"Removed {before-len(df2):,} duplicate rows"))

        num_cols, cat_cols = classify_cols(df2, target)
        for c in num_cols: df2[c] = pd.to_numeric(df2[c], errors="coerce")
        logs.append(("info", f"Detected {len(num_cols)} numeric, {len(cat_cols)} categorical features"))

        ns = {"Median fill":"median","Mean fill":"mean","Zero fill":"zero","Drop rows":"drop"}[num_strat]
        fixed = 0
        if ns == "drop":
            before = len(df2); df2.dropna(subset=num_cols, inplace=True)
            logs.append(("warn", f"Dropped {before-len(df2):,} rows (missing numerics)"))
        else:
            for c in num_cols:
                n = df2[c].isna().sum()
                if n:
                    fill = df2[c].median() if ns=="median" else df2[c].mean() if ns=="mean" else 0
                    df2[c].fillna(fill, inplace=True); fixed += n
            if fixed: logs.append(("ok", f"Imputed {fixed:,} numeric values with {ns}"))

        cs = {"Mode fill":"mode","Fill 'Unknown'":"unknown","Drop rows":"drop"}[cat_strat]
        fixed = 0
        if cs == "drop":
            before = len(df2); df2.dropna(subset=cat_cols, inplace=True)
            logs.append(("warn", f"Dropped {before-len(df2):,} rows (missing categoricals)"))
        else:
            for c in cat_cols:
                n = df2[c].isna().sum()
                if n:
                    fill = df2[c].mode()[0] if cs=="mode" and not df2[c].mode().empty else "Unknown"
                    df2[c].fillna(fill, inplace=True); fixed += n
            if fixed: logs.append(("ok", f"Imputed {fixed:,} categorical values with {cs}"))

        om = {"IQR clipping (1.5×)":"iqr","Z-score clipping (±3σ)":"zscore","No handling":"none"}[outlier_method]
        clipped = 0
        if om != "none":
            for c in num_cols:
                if c == target: continue
                if om == "iqr":
                    q1,q3 = df2[c].quantile(.25), df2[c].quantile(.75); iqr = q3-q1
                    lo,hi = q1-1.5*iqr, q3+1.5*iqr
                else:
                    mu,sig = df2[c].mean(), df2[c].std() or 1
                    lo,hi = mu-3*sig, mu+3*sig
                if ((df2[c]<lo)|(df2[c]>hi)).sum():
                    df2[c] = df2[c].clip(lo,hi); clipped += 1
            logs.append(("ok", f"Clipped outliers in {clipped} columns ({om.upper()})"))

        if log_transform:
            df2[target] = np.log1p(df2[target])
            logs.append(("ok", f"Applied log1p to '{target}'"))

        logs.append(("ok", f"✓ Clean complete — {len(df2):,} rows ready (removed {orig-len(df2):,})"))

        st.session_state.df_clean    = df2
        st.session_state.clean_log   = logs
        st.session_state.log_transform = log_transform
        st.session_state.target      = target
        st.session_state.step        = 4
        st.rerun()

    b1, _, b3 = st.columns([1,5,1])
    with b1:
        if st.button("← Back"):
            st.session_state.step = 2; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    df           = st.session_state.df_clean
    target       = st.session_state.target
    log_transform = st.session_state.log_transform

    st.markdown('<div class="sec-title">Train Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Fit a linear regression on the cleaned dataset</div>', unsafe_allow_html=True)

    # Clean log
    logs = st.session_state.clean_log
    if logs:
        html = "".join(f'<div class="log-{"ok" if t=="ok" else "warn" if t=="warn" else "info"}">{"✓" if t=="ok" else "⚠" if t=="warn" else "·"} {msg}</div>' for t,msg in logs)
        st.markdown(f'<div class="log-box">{html}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    nc, catc = classify_cols(df, target)
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(mcard("Clean Rows",            f"{len(df):,}",  "teal"),  unsafe_allow_html=True)
    c2.markdown(mcard("Numeric Features",      str(len(nc)),    "blue"),  unsafe_allow_html=True)
    c3.markdown(mcard("Categorical Features",  str(len(catc)),  "amber"), unsafe_allow_html=True)
    c4.markdown(mcard("Log Transform",         "Yes" if log_transform else "No", "teal" if log_transform else ""), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶  Train Linear Regression"):
        with st.spinner("Training…"):
            X = df.drop(columns=[target])
            y = df[target]
            num_cols = X.select_dtypes(include="number").columns.tolist()
            cat_cols = X.select_dtypes(exclude="number").columns.tolist()

            num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
            cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                 ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
            pre  = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")
            pipe = Pipeline([("pre", pre), ("reg", LinearRegression())])

            X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)

            y_te_r = np.expm1(y_te)   if log_transform else y_te
            y_pr_r = np.expm1(y_pred) if log_transform else y_pred

            r2   = r2_score(y_te_r, y_pr_r)
            rmse = float(np.sqrt(mean_squared_error(y_te_r, y_pr_r)))
            mae  = float(mean_absolute_error(y_te_r, y_pr_r))
            cv   = cross_val_score(pipe, X, y, cv=5, scoring="r2")

            ohe_feats = pipe["pre"].named_transformers_["cat"]["enc"].get_feature_names_out(cat_cols).tolist() if cat_cols else []
            feat_imp  = pd.Series(np.abs(pipe["reg"].coef_), index=num_cols+ohe_feats).sort_values(ascending=False)

            st.session_state.pipeline  = pipe
            st.session_state.feat_imp  = feat_imp
            st.session_state.eval_data = (y_te_r.values if hasattr(y_te_r,"values") else y_te_r,
                                          y_pr_r, r2, rmse, mae, cv)
            st.session_state.feat_cols = X.columns.tolist()
            st.rerun()

    if st.session_state.eval_data is not None:
        y_te_r, y_pr_r, r2, rmse, mae, cv = st.session_state.eval_data
        feat_imp = st.session_state.feat_imp

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(mcard("R² Score",      f"{r2:.4f}",                         "teal" if r2>.7 else "amber"), unsafe_allow_html=True)
        c2.markdown(mcard("RMSE",          f"${rmse:,.0f}",                     "blue"), unsafe_allow_html=True)
        c3.markdown(mcard("MAE",           f"${mae:,.0f}",                      "blue"), unsafe_allow_html=True)
        c4.markdown(mcard("CV R² (5-fold)",f"{cv.mean():.4f} ±{cv.std():.4f}", "teal" if cv.mean()>.7 else "amber"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        with ch1:
            top = feat_imp.head(10)
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#1E293B")
            ax.set_facecolor("#1E293B")
            colors = ["#1D9E75" if i<3 else "#60A5FA" if i<6 else "#64748B" for i in range(len(top))]
            ax.barh(range(len(top)), top.values, color=colors, height=0.6)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels([t[:24] for t in top.index], fontsize=8, color="#94A3B8")
            ax.invert_yaxis()
            ax.set_xlabel("|Coefficient|", fontsize=9, color="#64748B")
            for sp in ax.spines.values(): sp.set_color("#334155"); sp.set_linewidth(0.5)
            ax.tick_params(colors="#64748B", labelsize=8)
            plt.tight_layout()
            st.markdown('<div style="font-size:13px;font-weight:500;color:#94A3B8;margin-bottom:8px">Top 10 features</div>', unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=True); plt.close()

        with ch2:
            fig2, ax2 = plt.subplots(figsize=(5, 3.5), facecolor="#1E293B")
            ax2.set_facecolor("#1E293B")
            ax2.scatter(y_te_r/1e6, y_pr_r/1e6, alpha=0.4, s=14, color="#1D9E75", edgecolors="none")
            mn = min(y_te_r.min(), y_pr_r.min())/1e6
            mx = max(y_te_r.max(), y_pr_r.max())/1e6
            ax2.plot([mn,mx],[mn,mx],"--",color="#F59E0B",lw=1.4,label="Perfect fit")
            ax2.set_xlabel("Actual ($M)", fontsize=9, color="#64748B")
            ax2.set_ylabel("Predicted ($M)", fontsize=9, color="#64748B")
            ax2.legend(fontsize=8, labelcolor="#94A3B8", facecolor="#1E293B", edgecolor="#334155")
            for sp in ax2.spines.values(): sp.set_color("#334155"); sp.set_linewidth(0.5)
            ax2.tick_params(colors="#64748B", labelsize=8)
            ax2.text(.06,.92,f"R²={r2:.3f}",transform=ax2.transAxes,fontsize=9,color="#1D9E75",fontweight="bold")
            plt.tight_layout()
            st.markdown('<div style="font-size:13px;font-weight:500;color:#94A3B8;margin-bottom:8px">Actual vs predicted</div>', unsafe_allow_html=True)
            st.pyplot(fig2, use_container_width=True); plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue to Predict →"):
            st.session_state.step = 5; st.rerun()

    b1, _ = st.columns([1,6])
    with b1:
        if st.button("← Back"):
            st.session_state.step = 3; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    pipe          = st.session_state.pipeline
    feat_cols     = st.session_state.feat_cols
    log_transform = st.session_state.log_transform
    df_clean      = st.session_state.df_clean
    target        = st.session_state.target

    st.markdown('<div class="sec-title">Predict Price</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Enter house details to get an estimated price</div>', unsafe_allow_html=True)

    num_cols = df_clean.drop(columns=[target]).select_dtypes(include="number").columns.tolist()
    cat_cols = df_clean.drop(columns=[target]).select_dtypes(exclude="number").columns.tolist()

    bool_present  = [c for c in num_cols if c in BOOL_COLS]
    int_present   = [c for c in num_cols if c in INT_COLS and c not in BOOL_COLS]
    float_present = [c for c in num_cols if c not in BOOL_COLS and c not in INT_COLS]

    sample = {}

    # ── Section 1: Size & Structure ───────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Size & structure**")
    size_cols = ([c for c in float_present if c in ("sqft_living","sqft_lot")] +
                 [c for c in int_present   if c in ("bedrooms","bathrooms","floors","garage_cars","basement_sqft")])
    if size_cols:
        cols = st.columns(3)
        for i, c in enumerate(size_cols):
            w = cols[i % 3]
            lbl = c.replace("_"," ").title()
            if c in INT_COLS:
                mn,mx,step = INT_RANGES.get(c,(0,9999,1))
                sample[c] = w.number_input(lbl, min_value=mn, max_value=mx,
                                            value=int(PREDICT_DEFAULTS.get(c,mn)), step=step, key=f"p_{c}")
            else:
                mn,mx,step,fmt = FLOAT_FIELDS.get(c,(0.0,99999.0,1.0,"%.2f"))
                sample[c] = w.number_input(lbl, min_value=float(mn), max_value=float(mx),
                                            value=float(PREDICT_DEFAULTS.get(c,mn)),
                                            step=float(step), format=fmt, key=f"p_{c}")

    # ── Section 2: Property Features (Yes/No) ─────────────────────────────────
    if bool_present:
        st.markdown("---")
        st.markdown("**Property features**")
        cols = st.columns(3)
        for i, c in enumerate(bool_present):
            default_yn = "Yes" if PREDICT_DEFAULTS.get(c, 0) == 1 else "No"
            choice = cols[i%3].selectbox(c.replace("_"," ").title(), ["No","Yes"],
                                          index=["No","Yes"].index(default_yn), key=f"p_{c}")
            sample[c] = 1 if choice == "Yes" else 0

    # ── Section 3: Condition & Quality ────────────────────────────────────────
    cond_cols = [c for c in int_present if c in
                 ("year_built","house_age_years","condition_score","quality_grade","hoa_monthly_usd")]
    if cond_cols:
        st.markdown("---")
        st.markdown("**Condition & quality**")
        cols = st.columns(3)
        for i, c in enumerate(cond_cols):
            mn,mx,step = INT_RANGES.get(c,(0,9999,1))
            sample[c] = cols[i%3].number_input(c.replace("_"," ").title(), min_value=mn, max_value=mx,
                                                value=int(PREDICT_DEFAULTS.get(c,mn)), step=step, key=f"p_{c}")

    # ── Section 4: Location & Rates ───────────────────────────────────────────
    loc_cols = [c for c in float_present if c in
                ("school_rating","crime_rate_per_1000","dist_city_center_km",
                 "dist_school_km","dist_hospital_km","property_tax_rate_pct","market_trend")]
    if loc_cols:
        st.markdown("---")
        st.markdown("**Location & rates**")
        cols = st.columns(3)
        for i, c in enumerate(loc_cols):
            mn,mx,step,fmt = FLOAT_FIELDS.get(c,(0.0,99999.0,1.0,"%.2f"))
            sample[c] = cols[i%3].number_input(c.replace("_"," ").title(),
                                                min_value=float(mn), max_value=float(mx),
                                                value=float(PREDICT_DEFAULTS.get(c,mn)),
                                                step=float(step), format=fmt, key=f"p_{c}")

    # ── Section 5: Any remaining ──────────────────────────────────────────────
    shown = set(size_cols + bool_present + cond_cols + loc_cols)
    other = [c for c in num_cols if c not in shown]
    if other:
        st.markdown("---")
        st.markdown("**Other**")
        cols = st.columns(3)
        for i, c in enumerate(other):
            if c in INT_COLS:
                mn,mx,step = INT_RANGES.get(c,(0,99999,1))
                sample[c] = cols[i%3].number_input(c.replace("_"," ").title(), min_value=mn, max_value=mx,
                                                    value=int(PREDICT_DEFAULTS.get(c,mn)), step=step, key=f"p_{c}")
            else:
                mn,mx,step,fmt = FLOAT_FIELDS.get(c,(0.0,99999.0,1.0,"%.2f"))
                sample[c] = cols[i%3].number_input(c.replace("_"," ").title(),
                                                    min_value=float(mn), max_value=float(mx),
                                                    value=float(PREDICT_DEFAULTS.get(c,mn)),
                                                    step=float(step), format=fmt, key=f"p_{c}")

    # ── Categoricals ──────────────────────────────────────────────────────────
    if cat_cols:
        st.markdown("---")
        st.markdown("**Neighborhood & categories**")
        cols = st.columns(min(len(cat_cols), 3))
        for i, c in enumerate(cat_cols):
            opts = sorted(df_clean[c].dropna().unique().tolist())
            sample[c] = cols[i%3].selectbox(c.replace("_"," ").title(), opts, key=f"p_cat_{c}")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🏠  Predict Price"):
        df_s = pd.DataFrame([{c: sample.get(c, np.nan) for c in feat_cols}])
        raw  = pipe.predict(df_s)[0]
        price = np.expm1(raw) if log_transform else raw
        _, y_te_r, _, rmse, _, _ = st.session_state.eval_data
        lo, hi = max(0, price-rmse), price+rmse
        st.markdown(f'<div class="price-result"><div class="plabel">Estimated Market Price</div><div class="pvalue">${price:,.0f}</div><div class="prange">Confidence range · ${lo:,.0f} – ${hi:,.0f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    b1, _ = st.columns([1,6])
    with b1:
        if st.button("← Back"):
            st.session_state.step = 4; st.rerun()
