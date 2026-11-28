"""Compare Logistic Regression class_weight (1:1, 1:2, 1:3, 1:4, 1:5) at fixed C=1 for Session 12."""

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
BASE_LOGISTIC_JOBLIB = PREV11_RESULTS_DIR / "delivery_logistic_11.joblib"

PRED_12_CSV = RESULTS_DIR / "delivery_predictions_12.csv"
OUT_COMP_CSV = RESULTS_DIR / "logistic_weight_comparison_12.csv"

OUT_PNG_METRICS = GRAPHS_DIR / "03_로지스틱_가중치별_평가성능_지표_비교.png"
OUT_PNG_CM = GRAPHS_DIR / "04_로지스틱_가중치별_혼동행렬_및_물량_비교.png"
OUT_PNG_SHEET = GRAPHS_DIR / "00_전체_결과물_한눈에보기.png"

OUT_HTML_METRICS = INTERACTIVE_DIR / "03_logistic_weight_metrics_12.html"
OUT_HTML_CM = INTERACTIVE_DIR / "04_logistic_weight_confusion_12.html"

THRESHOLD = 0.50

WEIGHT_SPECS = [
    {"ratio": 1, "weight": None,         "tag": "w1_1", "label": "1:1 (로지스틱 기본)", "short": "1:1\n(기본·재사용)", "col_prefix": "로지스틱_가중치1대1", "color": "#64748B"},
    {"ratio": 2, "weight": {0: 1, 1: 2}, "tag": "w1_2", "label": "1:2 ({0:1, 1:2})",  "short": "1:2\n(지연 2배)",   "col_prefix": "로지스틱_가중치1대2", "color": "#0EA5E9"},
    {"ratio": 3, "weight": {0: 1, 1: 3}, "tag": "w1_3", "label": "1:3 ({0:1, 1:3})",  "short": "1:3\n(지연 3배)",   "col_prefix": "로지스틱_가중치1대3", "color": "#2563EB"},
    {"ratio": 4, "weight": {0: 1, 1: 4}, "tag": "w1_4", "label": "1:4 ({0:1, 1:4})",  "short": "1:4\n(지연 4배)",   "col_prefix": "로지스틱_가중치1대4", "color": "#7C3AED"},
    {"ratio": 5, "weight": {0: 1, 1: 5}, "tag": "w1_5", "label": "1:5 ({0:1, 1:5})",  "short": "1:5\n(지연 5배)",   "col_prefix": "로지스틱_가중치1대5", "color": "#DB2777"},
]


def safe_div(num: float, den: float):
    """Divide safely, returning None when denominator is zero."""
    return (num / den) if den > 0 else None


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """Compute binary classification metrics (Positive=1 delayed) with raw 0~1 ratios."""
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


def train_and_evaluate_weights():
    """Reuse 1:1 '로지스틱 기본' (C=1) and fit 1:2..1:5 with identical preprocessing & settings."""
    with open(REF_JSON, encoding="utf-8") as f:
        ref = json.load(f)

    raw_df = pd.read_csv(FEATURES_CSV)
    dt = pd.to_datetime(raw_df[ref["split_column"]])
    train_df = raw_df[(dt >= "2026-01-01") & (dt < "2026-07-01")].copy()
    eval_df = raw_df[(dt >= "2026-07-01") & (dt < "2026-09-01")].copy()

    input_cols = ref["numeric_inputs"] + ref["categorical_inputs"]
    target_col = ref["target"]

    base_pipe = joblib.load(BASE_LOGISTIC_JOBLIB)
    ohe = base_pipe.named_steps["ohe_preprocessor"]
    scaler = base_pipe.named_steps["numeric_scaler"]
    base_lr = base_pipe.named_steps["model"]
    base_params = base_lr.get_params()
    assert float(base_params["C"]) == 1.0

    X_train_scaled = scaler.transform(ohe.transform(train_df[input_cols]))
    y_train = train_df[target_col].to_numpy()
    y_eval = eval_df[target_col].to_numpy()

    pred_df = pd.read_csv(PRED_12_CSV)
    rows = []

    for spec in WEIGHT_SPECS:
        if spec["ratio"] == 1:
            pipe = base_pipe
            clf = base_lr
            reused = True
            joblib_name = "11회/results/delivery_logistic_11.joblib (재사용)"
        else:
            params = dict(base_params)
            params["C"] = 1.0
            params["class_weight"] = spec["weight"]
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                clf = LogisticRegression(**params)
                clf.fit(X_train_scaled, y_train)
                assert len(w) == 0, f"Unexpected warning for 1:{spec['ratio']}: {[str(x.message) for x in w]}"

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

        if spec["ratio"] == 1:
            max_diff = float(np.max(np.abs(p_eval - pred_df["로지스틱_C1_지연확률"].to_numpy())))
            assert max_diff < 1e-12, f"1:1 parity mismatch: {max_diff}"

        w_df = pd.DataFrame({
            "주문ID": eval_df["주문ID"].values,
            f"{spec['col_prefix']}_지연확률": p_eval,
            f"{spec['col_prefix']}_지연예측": (p_eval >= THRESHOLD).astype(int),
        })
        for col in [f"{spec['col_prefix']}_지연확률", f"{spec['col_prefix']}_지연예측"]:
            if col in pred_df.columns:
                pred_df = pred_df.drop(columns=[col])
        pred_df = pred_df.merge(w_df, on="주문ID", how="left", validate="one_to_one")
        assert pred_df[f"{spec['col_prefix']}_지연확률"].isna().sum() == 0

        n_iter = int(clf.n_iter_[0])
        for split_label, period_label, y_true, probs in [
            ("학습 (1~6월)", "2026-01~06", y_train, p_train),
            ("평가 (7~8월)", "2026-07~08", y_eval, p_eval),
        ]:
            m = compute_metrics(y_true, probs, THRESHOLD)
            rows.append({
                "모델": spec["label"],
                "정상_지연_가중치": f"1:{spec['ratio']}",
                "지연가중치": spec["ratio"],
                "C": 1.0,
                "구분": split_label,
                "기간": period_label,
                "기존모델재사용": reused,
                "반복수": n_iter,
                "모델파일": joblib_name,
                **m,
            })

    pred_df.to_csv(PRED_12_CSV, index=False, encoding="utf-8-sig")
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUT_COMP_CSV, index=False, encoding="utf-8-sig")
    return comp_df, pred_df


def save_metrics_png(comp_df: pd.DataFrame) -> None:
    """Create 2x2 bar chart comparing Accuracy, Precision, Recall, F1 across class_weight ratios."""
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
    x = np.arange(len(WEIGHT_SPECS))
    w = 0.34
    x_labels = [s["short"] for s in WEIGHT_SPECS]

    for ax, (col, title) in zip(axes.flatten(), metrics):
        tr_vals = [float(r[col]) * 100 for _, r in tr_df.iterrows()]
        ev_vals = [float(r[col]) * 100 for _, r in ev_df.iterrows()]
        ev_colors = [s["color"] for s in WEIGHT_SPECS]

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
        Patch(facecolor="#2563EB", edgecolor="#0F172A", label="평가 (7~8월 · 2,400건, 막대 색=가중치별)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=True, fontsize=10)
    fig.suptitle(
        "로지스틱 회귀 (C=1 고정) 정상:지연 가중치별 성능 비교 (50% 기준)",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(OUT_PNG_METRICS, dpi=180)
    plt.close(fig)


def save_confusion_workload_png(comp_df: pd.DataFrame) -> None:
    """Create 2x3 figure with 5 confusion matrices (rows=Pred, cols=Actual) + TP/FP/FN workload breakdown."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    ev_df = comp_df[comp_df["구분"] == "평가 (7~8월)"].reset_index(drop=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.8))
    axes_flat = axes.flatten()

    for ax, (_, row), spec in zip(axes_flat[:5], ev_df.iterrows(), WEIGHT_SPECS):
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.axis("off")

        tp = int(row["찾아낸_실제지연(TP)"])
        fp = int(row["점검대상_실제정상(FP)"])
        fn = int(row["놓친_실제지연(FN)"])
        tn = int(row["점검제외_실제정상(TN)"])
        tot = int(row["전체주문"])

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

    # Panel 6: Clean stacked inspection workload (TP + FP) vs Missed Delays (FN)
    ax6 = axes_flat[5]
    x_idx = np.arange(len(WEIGHT_SPECS))
    tp_v = ev_df["찾아낸_실제지연(TP)"].to_numpy()
    fp_v = ev_df["점검대상_실제정상(FP)"].to_numpy()
    fn_v = ev_df["놓친_실제지연(FN)"].to_numpy()

    bw = 0.40
    ax6.bar(x_idx - 0.14, tp_v, bw, label="찾아낸 지연 (TP)", color="#2563EB", edgecolor="#1E3A8A", linewidth=0.5)
    ax6.bar(x_idx - 0.14, fp_v, bw, bottom=tp_v, label="과탐 정상 (FP)", color="#F59E0B", alpha=0.88, edgecolor="#92400E", linewidth=0.5)
    ax6.bar(x_idx + 0.23, fn_v, 0.24, label="놓친 지연 (FN)", color="#EF4444", alpha=0.88, edgecolor="#991B1B", linewidth=0.5)

    for xi, tp, fp, fn in zip(x_idx, tp_v, fp_v, fn_v):
        tot = tp + fp
        ax6.text(xi - 0.14, tp / 2, f"{tp}", ha="center", va="center", fontsize=8.2, color="white", fontweight="bold")
        if fp >= 120:
            ax6.text(xi - 0.14, tp + fp / 2, f"{fp}", ha="center", va="center", fontsize=8.2, color="#1F2937", fontweight="bold")
            top_lbl = f"{tot}건"
        else:
            top_lbl = f"{tot}건\n(FP {fp})"
        ax6.text(xi - 0.14, tot + 18, top_lbl, ha="center", va="bottom", fontsize=8.1, color="#0F172A", fontweight="bold")
        ax6.text(xi + 0.23, fn + 16, f"{fn}", ha="center", va="bottom", fontsize=8.1, color="#991B1B", fontweight="bold")

    ax6.set_xticks(x_idx)
    ax6.set_xticklabels([f"1:{s['ratio']}" for s in WEIGHT_SPECS], fontsize=9.5, fontweight="bold")
    ax6.set_ylim(0, 1080)
    ax6.set_ylabel("주문 건수 (건)", fontsize=9.5)
    ax6.set_title("⑥ 가중치별 점검 물량(TP+FP) vs 놓친 지연(FN)\n(1:3에서 최고 F1 50.05% · 이후 FP 급증)", fontsize=10.2, fontweight="bold", pad=12)
    ax6.grid(axis="y", linestyle="--", alpha=0.35)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)
    ax6.legend(loc="upper left", fontsize=8.2, frameon=True)

    fig.suptitle(
        "로지스틱 회귀 (C=1) 가중치별 혼동행렬 및 점검 물량 변화 (7~8월 평가 2,400건 · 50% 기준)",
        fontsize=14.2,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(left=0.055, right=0.965, top=0.87, bottom=0.07, wspace=0.28, hspace=0.34)
    fig.savefig(OUT_PNG_CM, dpi=180)
    plt.close(fig)


def save_interactive_htmls(comp_df: pd.DataFrame) -> None:
    """Create interactive Plotly HTML charts for class_weight comparison."""
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    tr_df = comp_df[comp_df["구분"] == "학습 (1~6월)"].reset_index(drop=True)
    ev_df = comp_df[comp_df["구분"] == "평가 (7~8월)"].reset_index(drop=True)

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
    x_names = [s["label"] for s in WEIGHT_SPECS]

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
                marker_color=[s["color"] for s in WEIGHT_SPECS],
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
        title=dict(text="<b>로지스틱 회귀 (C=1) 정상:지연 가중치별 성능 비교 (50% 기준)</b>", x=0.5, xanchor="center"),
        template="plotly_white",
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
        margin=dict(t=105, b=80, l=65, r=35),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    pfig.write_html(str(OUT_HTML_METRICS), include_plotlyjs=True)

    cm_fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"<b>{s['label']}</b> (점검 {int(r['점검대상']):,}건)" for s, (_, r) in zip(WEIGHT_SPECS, ev_df.iterrows())] + ["<b>⑥ 점검 물량 분해 (TP·FP vs FN)</b>"],
        horizontal_spacing=0.09,
        vertical_spacing=0.18,
    )
    grid_pos = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    for (r_i, c_i), (_, row), spec in zip(grid_pos, ev_df.iterrows(), WEIGHT_SPECS):
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

    w_lbls = [f"1:{s['ratio']}" for s in WEIGHT_SPECS]
    cm_fig.add_trace(go.Bar(name="찾아낸 지연(TP)", x=w_lbls, y=ev_df["찾아낸_실제지연(TP)"], marker_color="#2563EB", text=ev_df["찾아낸_실제지연(TP)"], textposition="inside"), row=2, col=3)
    cm_fig.add_trace(go.Bar(name="과탐 정상(FP)", x=w_lbls, y=ev_df["점검대상_실제정상(FP)"], marker_color="#F59E0B", text=ev_df["점검대상_실제정상(FP)"], textposition="inside"), row=2, col=3)
    cm_fig.add_trace(go.Bar(name="놓친 지연(FN)", x=w_lbls, y=ev_df["놓친_실제지연(FN)"], marker_color="#EF4444", text=ev_df["놓친_실제지연(FN)"], textposition="outside"), row=2, col=3)

    cm_fig.update_yaxes(title_text="건수 (건)", row=2, col=3)
    cm_fig.update_layout(
        barmode="group",
        title=dict(text="<b>로지스틱 회귀 (C=1) 가중치별 혼동행렬 및 점검 물량 (7~8월 2,400건 · 50% 기준)</b>", x=0.5, xanchor="center"),
        template="plotly_white",
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        margin=dict(t=95, b=85, l=65, r=35),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    cm_fig.write_html(str(OUT_HTML_CM), include_plotlyjs=True)


def make_contact_sheet() -> None:
    """Update 00_전체_결과물_한눈에보기.png covering all 01~04 PNGs."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    items = [
        ("01_로지스틱_C별_평가성능_지표_비교.png", "01. 로지스틱 C별 학습·평가 지표"),
        ("02_로지스틱_C별_혼동행렬_비교.png", "02. 로지스틱 C별 혼동행렬·추이"),
        ("03_로지스틱_가중치별_평가성능_지표_비교.png", "03. 가중치별(1:1~1:5) 성능 지표"),
        ("04_로지스틱_가중치별_혼동행렬_및_물량_비교.png", "04. 가중치별 혼동행렬·물량"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17.6, 13.0))
    for ax, (fname, caption) in zip(axes.flatten(), items):
        img = mpimg.imread(GRAPHS_DIR / fname)
        ax.imshow(img)
        ax.set_title(caption, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#CBD5E1")
            sp.set_linewidth(1.2)
    fig.suptitle("[2026-11-28] 12회차 배송 지연 예측 결과물 한눈에 보기 (01 ~ 04)", fontsize=15.0, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.95], h_pad=2.2, w_pad=1.6)
    fig.savefig(OUT_PNG_SHEET, dpi=170)
    plt.close(fig)


def main() -> None:
    comp_df, pred_df = train_and_evaluate_weights()
    save_metrics_png(comp_df)
    save_confusion_workload_png(comp_df)
    save_interactive_htmls(comp_df)
    make_contact_sheet()

    ev_df = comp_df[comp_df["구분"] == "평가 (7~8월)"]
    print("=== 평가 기간(7~8월 2,400건 · C=1 · 50% 기준) 가중치별 결과 ===")
    for _, r in ev_df.iterrows():
        print(
            f"{r['정상_지연_가중치']} | iter={int(r['반복수'])}, "
            f"점검={int(r['점검대상'])}건({r['점검비율']*100:.2f}%), "
            f"TP={int(r['찾아낸_실제지연(TP)'])}, FP={int(r['점검대상_실제정상(FP)'])}, "
            f"FN={int(r['놓친_실제지연(FN)'])}, TN={int(r['점검제외_실제정상(TN)'])} | "
            f"Acc={r['정확도']*100:.2f}%, Prec={r['정밀도']*100:.2f}%, "
            f"Rec={r['재현율']*100:.2f}%, F1={r['F1']*100:.2f}%, "
            f"ROC-AUC={r['ROC_AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}"
        )
    print(f"\nUpdated predictions CSV: {PRED_12_CSV} (shape={pred_df.shape})")
    print(f"Saved weight comparison CSV: {OUT_COMP_CSV}")


if __name__ == "__main__":
    main()
