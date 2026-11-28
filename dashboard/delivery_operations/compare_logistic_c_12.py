"""Compare Logistic Regression regularization strength C (0.01, 0.1, 1, 10, 100) for Session 12."""

import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREV11_RESULTS_DIR = BASE_DIR / "11회" / "results"
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
INTERACTIVE_DIR = GRAPHS_DIR / "interactive"

FEATURES_CSV = DATA_DIR / "delivery_features_09.csv"
REF_JSON = DATA_DIR / "delivery_model_reference.json"
PREV11_PRED_CSV = PREV11_RESULTS_DIR / "delivery_predictions_11.csv"
BASE_LOGISTIC_JOBLIB = PREV11_RESULTS_DIR / "delivery_logistic_11.joblib"

OUT_PRED_CSV = RESULTS_DIR / "delivery_predictions_12.csv"
OUT_COMP_CSV = RESULTS_DIR / "logistic_c_comparison_12.csv"

OUT_PNG_METRICS = GRAPHS_DIR / "01_로지스틱_C별_평가성능_지표_비교.png"
OUT_PNG_CM = GRAPHS_DIR / "02_로지스틱_C별_혼동행렬_비교.png"
OUT_PNG_SHEET = GRAPHS_DIR / "00_전체_결과물_한눈에보기.png"

OUT_HTML_METRICS = INTERACTIVE_DIR / "01_logistic_c_metrics_12.html"
OUT_HTML_CM = INTERACTIVE_DIR / "02_logistic_c_confusion_12.html"

THRESHOLD = 0.50

C_SPECS = [
    {"c": 0.01, "tag": "c0_01", "label": "C=0.01", "sub": "강한 규제", "col_prefix": "로지스틱_C0.01", "color": "#64748B"},
    {"c": 0.1,  "tag": "c0_1",  "label": "C=0.1",  "sub": "중간 규제", "col_prefix": "로지스틱_C0.1",  "color": "#0EA5E9"},
    {"c": 1.0,  "tag": "c1",    "label": "C=1 (로지스틱 기본)", "sub": "기본값 재사용", "col_prefix": "로지스틱_C1", "color": "#2563EB"},
    {"c": 10.0, "tag": "c10",   "label": "C=10",   "sub": "약한 규제", "col_prefix": "로지스틱_C10",   "color": "#7C3AED"},
    {"c": 100.0,"tag": "c100",  "label": "C=100",  "sub": "매우 약한 규제", "col_prefix": "로지스틱_C100", "color": "#DB2777"},
]


def safe_div(num: float, den: float):
    """Divide safely, returning None when denominator is zero."""
    return (num / den) if den > 0 else None


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """Compute binary classification metrics with Positive=1 (delayed) and raw 0~1 ratios."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    total = len(y_true)
    inspect = tp + fp

    acc = safe_div(tp + tn, total)
    prec = safe_div(tp, inspect)
    rec = safe_div(tp, tp + fn)
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
    return {
        "전체주문": total,
        "실제지연": tp + fn,
        "실제정상": fp + tn,
        "임계값": threshold,
        "점검대상": inspect,
        "점검비율": safe_div(inspect, total),
        "찾아낸_실제지연(TP)": tp,
        "점검대상_실제정상(FP)": fp,
        "놓친_실제지연(FN)": fn,
        "점검제외_실제정상(TN)": tn,
        "정확도": acc,
        "정밀도": prec,
        "재현율": rec,
        "F1": f1,
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
    }


def load_splits_and_base_pipeline():
    """Load dataset splits and existing 11th-session '로지스틱 기본' pipeline."""
    with open(REF_JSON, encoding="utf-8") as f:
        ref = json.load(f)

    raw_df = pd.read_csv(FEATURES_CSV)
    dt = pd.to_datetime(raw_df[ref["split_column"]])
    train_df = raw_df[(dt >= "2026-01-01") & (dt < "2026-07-01")].copy()
    eval_df = raw_df[(dt >= "2026-07-01") & (dt < "2026-09-01")].copy()

    input_cols = ref["numeric_inputs"] + ref["categorical_inputs"]
    target_col = ref["target"]

    base_pipe = joblib.load(BASE_LOGISTIC_JOBLIB)
    return ref, train_df, eval_df, input_cols, target_col, base_pipe


def train_and_evaluate_c_models(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    input_cols: list[str],
    target_col: str,
    base_pipe: Pipeline,
):
    """Reuse C=1 pipeline and train C=0.01, 0.1, 10, 100 with identical preprocessing & settings."""
    ohe = base_pipe.named_steps["ohe_preprocessor"]
    scaler = base_pipe.named_steps["numeric_scaler"]
    base_lr = base_pipe.named_steps["model"]
    base_params = base_lr.get_params()

    X_train_scaled = scaler.transform(ohe.transform(train_df[input_cols]))
    X_eval_scaled = scaler.transform(ohe.transform(eval_df[input_cols]))
    y_train = train_df[target_col].to_numpy()
    y_eval = eval_df[target_col].to_numpy()

    prev_pred_df = pd.read_csv(PREV11_PRED_CSV)
    out_pred_df = prev_pred_df.copy()

    rows = []
    prob_parity_checked = False

    for spec in C_SPECS:
        c_val = spec["c"]
        if c_val == 1.0:
            pipe = base_pipe
            clf = base_lr
            reused = True
            joblib_name = "11회/results/delivery_logistic_11.joblib (재사용)"
        else:
            params = dict(base_params)
            params["C"] = float(c_val)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                clf = LogisticRegression(**params)
                clf.fit(X_train_scaled, y_train)
                assert len(w) == 0, f"Unexpected warning for C={c_val}: {[str(x.message) for x in w]}"

            pipe = Pipeline([
                ("ohe_preprocessor", ohe),
                ("numeric_scaler", scaler),
                ("model", clf),
            ])
            reused = False
            joblib_file = RESULTS_DIR / f"delivery_logistic_{spec['tag']}_12.joblib"
            joblib.dump(pipe, joblib_file)
            joblib_name = joblib_file.name

        p_train = pipe.predict_proba(train_df[input_cols])[:, 1]
        p_eval = pipe.predict_proba(eval_df[input_cols])[:, 1]

        if c_val == 1.0:
            max_diff = float(np.max(np.abs(p_eval - prev_pred_df["로지스틱회귀_지연확률"].to_numpy())))
            assert max_diff < 1e-12, f"C=1 mismatch vs delivery_predictions_11.csv: {max_diff}"
            prob_parity_checked = True

        # Align by 주문ID into delivery_predictions_12.csv
        c_df = pd.DataFrame({
            "주문ID": eval_df["주문ID"].values,
            f"{spec['col_prefix']}_지연확률": p_eval,
            f"{spec['col_prefix']}_지연예측": (p_eval >= THRESHOLD).astype(int),
        })
        for col in [f"{spec['col_prefix']}_지연확률", f"{spec['col_prefix']}_지연예측"]:
            if col in out_pred_df.columns:
                out_pred_df = out_pred_df.drop(columns=[col])
        out_pred_df = out_pred_df.merge(c_df, on="주문ID", how="left", validate="one_to_one")
        assert out_pred_df[f"{spec['col_prefix']}_지연확률"].isna().sum() == 0

        coef_l2 = float(np.linalg.norm(clf.coef_))
        n_iter = int(clf.n_iter_[0])

        for split_label, period_label, y_true, probs in [
            ("학습 (1~6월)", "2026-01~06", y_train, p_train),
            ("평가 (7~8월)", "2026-07~08", y_eval, p_eval),
        ]:
            m = compute_metrics(y_true, probs, THRESHOLD)
            rows.append({
                "모델": spec["label"],
                "C": c_val,
                "구분": split_label,
                "기간": period_label,
                "기존모델재사용": reused,
                "반복수": n_iter,
                "계수_L2노름": coef_l2,
                "모델파일": joblib_name,
                **m,
            })

    assert prob_parity_checked
    out_pred_df.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUT_COMP_CSV, index=False, encoding="utf-8-sig")
    return comp_df, out_pred_df


def save_metrics_png(comp_df: pd.DataFrame) -> None:
    """Create a 2x2 static figure comparing Accuracy, Precision, Recall, and F1 across C values."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    tr_df = comp_df[comp_df["구분"] == "학습 (1~6월)"].reset_index(drop=True)
    ev_df = comp_df[comp_df["구분"] == "평가 (7~8월)"].reset_index(drop=True)

    metrics = [
        ("정확도", "① 정확도 (Accuracy)"),
        ("정밀도", "② 정밀도 (Precision · 점검 적중률)"),
        ("재현율", "③ 재현율 (Recall · 지연 포착률)"),
        ("F1", "④ F1 Score (정밀도·재현율 균형)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.6), sharey=True)
    x = np.arange(len(C_SPECS))
    w = 0.34
    x_labels = [
        f"C={s['c']:g}\n({s['sub']})" if s["c"] != 1.0 else "C=1 (기본)\n(재사용)"
        for s in C_SPECS
    ]

    for ax, (col, title) in zip(axes.flatten(), metrics):
        tr_vals = [float(r[col]) * 100 for _, r in tr_df.iterrows()]
        ev_vals = [float(r[col]) * 100 for _, r in ev_df.iterrows()]
        ev_colors = [s["color"] for s in C_SPECS]

        b1 = ax.bar(x - w / 2, tr_vals, w, label="학습 (1~6월 · 7,200건)", color="#CBD5E1", edgecolor="#64748B", linewidth=0.6)
        b2 = ax.bar(x + w / 2, ev_vals, w, label="평가 (7~8월 · 2,400건)", color=ev_colors, edgecolor="#0F172A", linewidth=0.7)

        for bar, v in zip(b1, tr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.0, color="#475569", fontweight="bold")
        for bar, v in zip(b2, ev_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.4, color="#0F172A", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9.3, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_ylabel("지표 값 (%)", fontsize=10)
        ax.set_title(title, fontsize=11.8, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle="--", alpha=0.38)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#CBD5E1", edgecolor="#64748B", label="학습 (1~6월 · 7,200건)"),
        Patch(facecolor="#2563EB", edgecolor="#0F172A", label="평가 (7~8월 · 2,400건, 막대 색=C값별)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=True, fontsize=10)
    fig.suptitle(
        "로지스틱 회귀 C(규제 강도) 비교 — 학습(1~6월) vs 평가(7~8월) 성능 지표 (50% 기준)",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG_METRICS, dpi=180)
    plt.close(fig)


def save_confusion_png(comp_df: pd.DataFrame) -> None:
    """Create a 2x3 figure showing 5 confusion matrices (rows=Pred, cols=Actual) and a summary panel."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    ev_df = comp_df[comp_df["구분"] == "평가 (7~8월)"].reset_index(drop=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.8))
    axes_flat = axes.flatten()

    for ax, (_, row), spec in zip(axes_flat[:5], ev_df.iterrows(), C_SPECS):
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.axis("off")

        tp = int(row["찾아낸_실제지연(TP)"])
        fp = int(row["점검대상_실제정상(FP)"])
        fn = int(row["놓친_실제지연(FN)"])
        tn = int(row["점검제외_실제정상(TN)"])
        tot = int(row["전체주문"])

        # Unified layout: rows=Prediction (top=Normal, bottom=Delay), cols=Actual (left=Normal, right=Delay)
        cells = [
            (0, 1, tn, "TN · 정상 예측(제외)", "#EFF6FF", "#3B82F6", "#1D4ED8"),
            (1, 1, fn, "FN · 놓친 지연", "#FEF2F2", "#EF4444", "#B91C1C"),
            (0, 0, fp, "FP · 과탐 정상(점검)", "#FFFBEB", "#F59E0B", "#B45309"),
            (1, 0, tp, "TP · 찾아낸 지연", "#ECFDF5", "#10B981", "#047857"),
        ]
        pad = 0.05
        for x0, y0, cnt, label, bg, edge, tcol in cells:
            rect = mpatches.FancyBboxPatch(
                (x0 + pad, y0 + pad),
                1 - 2 * pad,
                1 - 2 * pad,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=bg,
                edgecolor=edge,
                linewidth=1.8,
            )
            ax.add_patch(rect)
            ax.text(x0 + 0.5, y0 + 0.74, label, ha="center", va="center", fontsize=8.8, fontweight="bold", color=tcol)
            ax.text(x0 + 0.5, y0 + 0.44, f"{cnt:,}건", ha="center", va="center", fontsize=15.0, fontweight="bold", color="#0F172A")
            ax.text(x0 + 0.5, y0 + 0.20, f"({cnt / tot * 100:.2f}%)", ha="center", va="center", fontsize=8.5, color="#475569")

        ax.text(0.5, 2.04, "실제 정상", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#334155")
        ax.text(1.5, 2.04, "실제 지연", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#334155")
        ax.text(-0.04, 1.5, "정상 예측\n(제외)", ha="right", va="center", fontsize=8.4, fontweight="bold", color="#334155")
        ax.text(-0.04, 0.5, "지연 예측\n(점검)", ha="right", va="center", fontsize=8.4, fontweight="bold", color="#334155")

        ax.set_title(
            f"{spec['label']} (점검 {int(row['점검대상']):,}건 · {row['점검비율']*100:.2f}%)\n"
            f"정확도 {row['정확도']*100:.2f}% | 정밀도 {row['정밀도']*100:.2f}% | 재현율 {row['재현율']*100:.2f}% | F1 {row['F1']*100:.2f}%",
            fontsize=10.0,
            fontweight="bold",
            pad=22,
        )

    # Panel 6: Summary trend of Precision, Recall, F1 and Inspection count across C
    ax6 = axes_flat[5]
    c_labels = [f"C={s['c']:g}" for s in C_SPECS]
    x_idx = np.arange(len(C_SPECS))
    prec_v = ev_df["정밀도"].to_numpy() * 100
    rec_v = ev_df["재현율"].to_numpy() * 100
    f1_v = ev_df["F1"].to_numpy() * 100
    insp_v = ev_df["점검대상"].to_numpy()

    ax6_r = ax6.twinx()
    ax6_r.bar(x_idx, insp_v, width=0.46, color="#E2E8F0", edgecolor="#94A3B8", alpha=0.72, label="점검 대상 건수(우측)")
    for xi, cnt in zip(x_idx, insp_v):
        ax6_r.text(xi, cnt + 5, f"{int(cnt)}건", ha="center", va="bottom", fontsize=8.3, color="#334155", fontweight="bold")
    ax6_r.set_ylim(0, 305)
    ax6_r.set_ylabel("점검 대상 건수 (건)", fontsize=9.2, color="#475569")

    ax6.plot(x_idx, prec_v, marker="o", linewidth=2.2, color="#0D9488", label="정밀도 (%)")
    ax6.plot(x_idx, f1_v, marker="s", linewidth=2.2, color="#2563EB", label="F1 Score (%)")
    ax6.plot(x_idx, rec_v, marker="^", linewidth=2.2, color="#EF4444", label="재현율 (%)")
    for xi, pv, fv, rv in zip(x_idx, prec_v, f1_v, rec_v):
        ax6.text(xi, pv + 2.2, f"{pv:.2f}%", ha="center", va="bottom", fontsize=7.8, color="#0F766E", fontweight="bold")
        ax6.text(xi, fv + 2.2, f"{fv:.2f}%", ha="center", va="bottom", fontsize=7.8, color="#1D4ED8", fontweight="bold")
        ax6.text(xi, rv - 2.6, f"{rv:.2f}%", ha="center", va="top", fontsize=7.8, color="#B91C1C", fontweight="bold")

    ax6.set_zorder(ax6_r.get_zorder() + 1)
    ax6.patch.set_visible(False)
    ax6.set_xticks(x_idx)
    ax6.set_xticklabels(c_labels, fontsize=9.2, fontweight="bold")
    ax6.set_ylim(10, 98)
    ax6.set_ylabel("성능 지표 (%)", fontsize=9.2)
    ax6.set_title("⑥ C 변화에 따른 지표·점검물량 추이\n(C≥1 구간 성능 수렴 확인)", fontsize=10.3, fontweight="bold", pad=12)
    ax6.grid(axis="y", linestyle="--", alpha=0.35)
    ax6.spines["top"].set_visible(False)
    ax6_r.spines["top"].set_visible(False)
    h1, l1 = ax6.get_legend_handles_labels()
    h2, l2 = ax6_r.get_legend_handles_labels()
    ax6.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2, fontsize=7.5, frameon=True)

    fig.suptitle(
        "로지스틱 회귀 C별 혼동행렬 및 지표 변화 (7~8월 평가 2,400건 · 50% 기준 · 행=예측 / 열=실제)",
        fontsize=14.2,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(left=0.055, right=0.945, top=0.87, bottom=0.07, wspace=0.32, hspace=0.34)
    fig.savefig(OUT_PNG_CM, dpi=180)
    plt.close(fig)


def save_interactive_htmls(comp_df: pd.DataFrame) -> None:
    """Create interactive Plotly HTML charts for C comparison."""
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    tr_df = comp_df[comp_df["구분"] == "학습 (1~6월)"].reset_index(drop=True)
    ev_df = comp_df[comp_df["구분"] == "평가 (7~8월)"].reset_index(drop=True)

    # 1. Interactive metrics comparison
    metrics = [
        ("정확도", "① 정확도 (Accuracy)"),
        ("정밀도", "② 정밀도 (Precision)"),
        ("재현율", "③ 재현율 (Recall)"),
        ("F1", "④ F1 Score"),
    ]
    pfig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"<b>{t}</b>" for _, t in metrics],
        shared_yaxes=True,
        horizontal_spacing=0.07,
        vertical_spacing=0.16,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    x_names = [s["label"] for s in C_SPECS]

    for idx, ((r_i, c_i), (col, title)) in enumerate(zip(positions, metrics)):
        pfig.add_trace(
            go.Bar(
                name="학습 (1~6월 · 7,200건)",
                x=x_names,
                y=[r[col] * 100 for _, r in tr_df.iterrows()],
                marker_color="#CBD5E1",
                text=[f"{r[col]*100:.2f}%" for _, r in tr_df.iterrows()],
                textposition="outside",
                showlegend=(idx == 0),
                customdata=[[int(r["점검대상"]), int(r["찾아낸_실제지연(TP)"]), int(r["점검대상_실제정상(FP)"]), int(r["놓친_실제지연(FN)"]), r["ROC_AUC"]] for _, r in tr_df.iterrows()],
                hovertemplate="<b>%{x}</b> (학습)<br>" + title + ": %{y:.2f}%<br>점검 %{customdata[0]:,}건 (TP %{customdata[1]:,} · FP %{customdata[2]:,} · FN %{customdata[3]:,})<br>ROC-AUC: %{customdata[4]:.4f}<extra></extra>",
            ),
            row=r_i, col=c_i,
        )
        pfig.add_trace(
            go.Bar(
                name="평가 (7~8월 · 2,400건)",
                x=x_names,
                y=[r[col] * 100 for _, r in ev_df.iterrows()],
                marker_color=[s["color"] for s in C_SPECS],
                text=[f"{r[col]*100:.2f}%" for _, r in ev_df.iterrows()],
                textposition="outside",
                showlegend=(idx == 0),
                customdata=[[int(r["점검대상"]), int(r["찾아낸_실제지연(TP)"]), int(r["점검대상_실제정상(FP)"]), int(r["놓친_실제지연(FN)"]), r["ROC_AUC"]] for _, r in ev_df.iterrows()],
                hovertemplate="<b>%{x}</b> (평가)<br>" + title + ": %{y:.2f}%<br>점검 %{customdata[0]:,}건 (TP %{customdata[1]:,} · FP %{customdata[2]:,} · FN %{customdata[3]:,})<br>ROC-AUC: %{customdata[4]:.4f}<extra></extra>",
            ),
            row=r_i, col=c_i,
        )

    pfig.update_yaxes(range=[0, 102], title_text="지표 값 (%)", row=1, col=1)
    pfig.update_yaxes(range=[0, 102], title_text="지표 값 (%)", row=2, col=1)
    pfig.update_layout(
        barmode="group",
        title=dict(text="<b>로지스틱 회귀 C별 학습(1~6월) vs 평가(7~8월) 성능 지표 비교 (50% 기준)</b>", x=0.5, xanchor="center"),
        template="plotly_white",
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
        margin=dict(t=105, b=80, l=65, r=35),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    pfig.write_html(str(OUT_HTML_METRICS), include_plotlyjs=True)

    # 2. Interactive confusion matrix heatmaps + summary trend
    cm_fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"<b>{s['label']}</b> (점검 {int(r['점검대상']):,}건)" for s, (_, r) in zip(C_SPECS, ev_df.iterrows())] + ["<b>⑥ C별 지표 추이 (평가)</b>"],
        horizontal_spacing=0.09,
        vertical_spacing=0.18,
    )
    grid_pos = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    for (r_i, c_i), (_, row), spec in zip(grid_pos, ev_df.iterrows(), C_SPECS):
        tp = int(row["찾아낸_실제지연(TP)"])
        fp = int(row["점검대상_실제정상(FP)"])
        fn = int(row["놓친_실제지연(FN)"])
        tn = int(row["점검제외_실제정상(TN)"])
        tot = int(row["전체주문"])
        z = [[fp, tp], [tn, fn]]
        txt = [
            [f"<b>{fp:,}건</b><br>FP · {fp/tot*100:.2f}%", f"<b>{tp:,}건</b><br>TP · {tp/tot*100:.2f}%"],
            [f"<b>{tn:,}건</b><br>TN · {tn/tot*100:.2f}%", f"<b>{fn:,}건</b><br>FN · {fn/tot*100:.2f}%"],
        ]
        cm_fig.add_trace(
            go.Heatmap(
                z=z,
                x=["실제 정상", "실제 지연"],
                y=["지연 예측(점검)", "정상 예측(제외)"],
                text=txt,
                texttemplate="%{text}",
                textfont=dict(size=12, color="#0F172A"),
                colorscale=[[0.0, "#F8FAFC"], [0.2, "#FEF3C7"], [0.45, "#DBEAFE"], [1.0, "#93C5FD"]],
                showscale=False,
                xgap=4,
                ygap=4,
                hovertemplate=f"<b>{spec['label']}</b><br>%{{y}} × %{{x}}<br>건수: %{{z:,}}건<extra></extra>",
            ),
            row=r_i, col=c_i,
        )

    c_lbls = [f"C={s['c']:g}" for s in C_SPECS]
    for name, col, color in [
        ("정밀도 (%)", "정밀도", "#0D9488"),
        ("F1 Score (%)", "F1", "#2563EB"),
        ("재현율 (%)", "재현율", "#EF4444"),
    ]:
        cm_fig.add_trace(
            go.Scatter(
                x=c_lbls,
                y=ev_df[col] * 100,
                mode="lines+markers+text",
                name=name,
                line=dict(color=color, width=2.4),
                text=[f"{v*100:.2f}%" for v in ev_df[col]],
                textposition="top center",
                hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.2f}}%<extra></extra>",
            ),
            row=2, col=3,
        )
    cm_fig.update_yaxes(range=[0, 92], title_text="비율 (%)", row=2, col=3)
    cm_fig.update_layout(
        title=dict(text="<b>로지스틱 회귀 C별 혼동행렬 및 지표 변화 (7~8월 평가 2,400건 · 50% 기준)</b>", x=0.5, xanchor="center"),
        template="plotly_white",
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        margin=dict(t=95, b=85, l=65, r=35),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    cm_fig.write_html(str(OUT_HTML_CM), include_plotlyjs=True)


def make_contact_sheet() -> None:
    """Generate 00_전체_결과물_한눈에보기.png thumbnail sheet covering current 12th-session static PNGs."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    items = [
        (OUT_PNG_METRICS.name, "01. 로지스틱 C별 학습·평가 지표 비교"),
        (OUT_PNG_CM.name, "02. 로지스틱 C별 혼동행렬 및 추이"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.8))
    for ax, (fname, caption) in zip(axes, items):
        img = mpimg.imread(GRAPHS_DIR / fname)
        ax.imshow(img)
        ax.set_title(caption, fontsize=11.0, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#CBD5E1")
            sp.set_linewidth(1.2)
    fig.suptitle("[2026-11-28] 12회차 배송 지연 예측 결과물 한눈에 보기 (01 ~ 02)", fontsize=14.0, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_PNG_SHEET, dpi=170)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    _, train_df, eval_df, input_cols, target_col, base_pipe = load_splits_and_base_pipeline()
    comp_df, out_pred_df = train_and_evaluate_c_models(train_df, eval_df, input_cols, target_col, base_pipe)
    save_metrics_png(comp_df)
    save_confusion_png(comp_df)
    save_interactive_htmls(comp_df)
    make_contact_sheet()

    ev_df = comp_df[comp_df["구분"] == "평가 (7~8월)"]
    print("=== 평가 기간(7~8월 2,400건 · 50% 기준) C별 결과 ===")
    for _, r in ev_df.iterrows():
        print(
            f"{r['모델']} | iter={int(r['반복수'])}, L2={r['계수_L2노름']:.4f}, "
            f"점검={int(r['점검대상'])}건({r['점검비율']*100:.2f}%), "
            f"TP={int(r['찾아낸_실제지연(TP)'])}, FP={int(r['점검대상_실제정상(FP)'])}, "
            f"FN={int(r['놓친_실제지연(FN)'])}, TN={int(r['점검제외_실제정상(TN)'])} | "
            f"Acc={r['정확도']*100:.2f}%, Prec={r['정밀도']*100:.2f}%, "
            f"Rec={r['재현율']*100:.2f}%, F1={r['F1']*100:.2f}%, "
            f"ROC-AUC={r['ROC_AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}"
        )
    print(f"\nSaved predictions CSV: {OUT_PRED_CSV} (shape={out_pred_df.shape})")
    print(f"Saved comparison CSV:  {OUT_COMP_CSV}")


if __name__ == "__main__":
    main()
