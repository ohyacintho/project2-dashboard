"""Evaluate and compare all 6 models (including 6th Logistic Regression C=1, 1:3) for Session 12."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREV11_RESULTS_DIR = BASE_DIR / "11회" / "results"
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
INTERACTIVE_DIR = GRAPHS_DIR / "interactive"

FEATURES_CSV = DATA_DIR / "delivery_features_09.csv"
REF_JSON = DATA_DIR / "delivery_model_reference.json"
PRED_12_CSV = RESULTS_DIR / "delivery_predictions_12.csv"

OUT_COMP_CSV = RESULTS_DIR / "model_comparison_12.csv"
OUT_TE_CSV = RESULTS_DIR / "model_train_eval_comparison_12.csv"
OUT_THRESH_CSV = RESULTS_DIR / "logistic_w13_thresholds_12.csv"

OUT_PNG_METRICS = GRAPHS_DIR / "10_6개모델_평가성능_지표_비교.png"
OUT_PNG_CM = GRAPHS_DIR / "11_6개모델_혼동행렬_비교.png"
OUT_PNG_ROC = GRAPHS_DIR / "12_6개모델_ROC곡선_및_AUC_비교.png"

OUT_HTML_METRICS = INTERACTIVE_DIR / "10_6models_metrics_12.html"
OUT_HTML_CM = INTERACTIVE_DIR / "11_6models_confusion_12.html"
OUT_HTML_ROC = INTERACTIVE_DIR / "model_roc_12.html"

THRESHOLD = 0.50

SIX_MODELS = [
    {
        "key": "tree4",
        "name": "의사결정나무 · 깊이 4",
        "label_2line": "의사결정나무\n(깊이 4)",
        "joblib_path": PREV11_RESULTS_DIR / "delivery_tree_depth4_10.joblib",
        "prob_col": "의사결정나무_지연확률",
        "color": "#64748B",
        "ls": "--",
        "lw": 1.8,
    },
    {
        "key": "tree6",
        "name": "의사결정나무 · 깊이 6",
        "label_2line": "의사결정나무\n(깊이 6)",
        "joblib_path": PREV11_RESULTS_DIR / "delivery_tree_depth6_10.joblib",
        "prob_col": "의사결정나무_깊이6_지연확률",
        "color": "#0EA5E9",
        "ls": "--",
        "lw": 1.8,
    },
    {
        "key": "rf_none",
        "name": "랜덤포레스트 (None)",
        "label_2line": "랜덤포레스트\n(None)",
        "joblib_path": PREV11_RESULTS_DIR / "delivery_forest_10.joblib",
        "prob_col": "랜덤포레스트_지연확률",
        "color": "#0D9488",
        "ls": "-.",
        "lw": 2.0,
    },
    {
        "key": "rf_bal",
        "name": "가중치 적용 랜덤포레스트 (balanced)",
        "label_2line": "가중치 RF\n(balanced)",
        "joblib_path": PREV11_RESULTS_DIR / "delivery_forest_balanced_10.joblib",
        "prob_col": "가중치_적용_랜덤포레스트_지연확률",
        "color": "#F59E0B",
        "ls": "-",
        "lw": 2.2,
    },
    {
        "key": "lr_base",
        "name": "로지스틱 기본 (C=1, 1:1)",
        "label_2line": "로지스틱 기본\n(C=1, 1:1)",
        "joblib_path": PREV11_RESULTS_DIR / "delivery_logistic_11.joblib",
        "prob_col": "로지스틱_C1_지연확률",
        "color": "#7C3AED",
        "ls": ":",
        "lw": 2.2,
    },
    {
        "key": "lr_w13",
        "name": "로지스틱 가중치 (C=1, 1:3)",
        "label_2line": "로지스틱 가중치\n(C=1, 1:3 ★)",
        "joblib_path": RESULTS_DIR / "delivery_logistic_w1_3_12.joblib",
        "prob_col": "로지스틱_가중치1대3_지연확률",
        "color": "#DC2626",
        "ls": "-",
        "lw": 2.7,
    },
]


def safe_div(num: float, den: float):
    """Divide safely, returning None when denominator is zero."""
    return (num / den) if den > 0 else None


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> dict:
    """Compute binary classification metrics (Positive=1 delayed) with raw 0~1 values."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    total = len(y_true)
    inspect = tp + fp
    actual_pos = tp + fn
    actual_neg = fp + tn

    acc = safe_div(tp + tn, total)
    prec = safe_div(tp, inspect)
    rec = safe_div(tp, actual_pos)
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
    fpr = safe_div(fp, actual_neg)
    return {
        "전체주문": total,
        "실제지연": actual_pos,
        "실제정상": actual_neg,
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
        "FPR": fpr,
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
    }


def evaluate_six_models():
    """Load saved 6 model pipelines, verify parity vs delivery_predictions_12.csv, and compute metrics."""
    with open(REF_JSON, encoding="utf-8") as f:
        ref = json.load(f)

    raw_df = pd.read_csv(FEATURES_CSV)
    dt = pd.to_datetime(raw_df[ref["split_column"]])
    train_df = raw_df[(dt >= "2026-01-01") & (dt < "2026-07-01")].copy()
    eval_df = raw_df[(dt >= "2026-07-01") & (dt < "2026-09-01")].copy()

    input_cols = ref["numeric_inputs"] + ref["categorical_inputs"]
    target_col = ref["target"]
    y_train = train_df[target_col].to_numpy()
    y_eval = eval_df[target_col].to_numpy()

    pred_df = pd.read_csv(PRED_12_CSV)
    assert np.array_equal(pred_df[target_col].to_numpy(), y_eval)

    eval_rows = []
    te_rows = []
    eval_probs = {}

    for spec in SIX_MODELS:
        pipe = joblib.load(spec["joblib_path"])
        p_train = pipe.predict_proba(train_df[input_cols])[:, 1]
        p_eval = pipe.predict_proba(eval_df[input_cols])[:, 1]
        max_diff = float(np.max(np.abs(p_eval - pred_df[spec["prob_col"]].to_numpy())))
        assert max_diff < 1e-12, f"Parity mismatch for {spec['name']}: {max_diff}"
        eval_probs[spec["key"]] = p_eval

        m_tr = compute_metrics(y_train, p_train, THRESHOLD)
        m_ev = compute_metrics(y_eval, p_eval, THRESHOLD)
        eval_rows.append({"모델": spec["name"], "평가기간": "2026-07~08", "모델파일": spec["joblib_path"].name, **m_ev})
        te_rows.append({"모델": spec["name"], "구분": "학습 (1~6월)", "기간": "2026-01~06", **m_tr})
        te_rows.append({"모델": spec["name"], "구분": "평가 (7~8월)", "기간": "2026-07~08", **m_ev})

    comp_df = pd.DataFrame(eval_rows)
    te_df = pd.DataFrame(te_rows)
    comp_df.to_csv(OUT_COMP_CSV, index=False, encoding="utf-8-sig")
    te_df.to_csv(OUT_TE_CSV, index=False, encoding="utf-8-sig")

    # 6th model ROC threshold table (50%, 30%, 20%)
    p6 = eval_probs["lr_w13"]
    thr_rows = []
    for thr in [0.50, 0.30, 0.20]:
        mt = compute_metrics(y_eval, p6, thr)
        thr_rows.append({"모델": "로지스틱 가중치 (C=1, 1:3)", "임계값표기": f"{int(thr*100)}%", **mt})
    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(OUT_THRESH_CSV, index=False, encoding="utf-8-sig")

    return comp_df, te_df, thr_df, y_eval, eval_probs


def save_metrics_png(te_df: pd.DataFrame) -> None:
    """Create 10_6개모델_평가성능_지표_비교.png (2x2 train vs eval bar charts)."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    metrics = [
        ("정확도", "① 정확도 (Accuracy)"),
        ("정밀도", "② 정밀도 (Precision · 점검 적중률)"),
        ("재현율", "③ 재현율 (Recall · 지연 포착률)"),
        ("F1", "④ F1 Score (정밀도·재현율 균형)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16.4, 9.8), sharey=True)
    x = np.arange(len(SIX_MODELS))
    w = 0.35
    x_labels = [m["label_2line"] for m in SIX_MODELS]

    for ax, (col, title) in zip(axes.flatten(), metrics):
        tr_vals = [te_df[(te_df["모델"] == m["name"]) & (te_df["구분"] == "학습 (1~6월)")][col].iloc[0] * 100 for m in SIX_MODELS]
        ev_vals = [te_df[(te_df["모델"] == m["name"]) & (te_df["구분"] == "평가 (7~8월)")][col].iloc[0] * 100 for m in SIX_MODELS]
        ev_colors = [m["color"] for m in SIX_MODELS]

        b1 = ax.bar(x - w / 2, tr_vals, w, label="학습 (1~6월 · 7,200건)", color="#CBD5E1", edgecolor="#64748B", linewidth=0.5)
        b2 = ax.bar(x + w / 2, ev_vals, w, label="평가 (7~8월 · 2,400건)", color=ev_colors, edgecolor="#0F172A", linewidth=0.6)

        for bar, v in zip(b1, tr_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.1, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.6, color="#475569", fontweight="bold")
        for bar, v in zip(b2, ev_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.1, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.9, color="#0F172A", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8.8, fontweight="bold")
        ax.set_ylim(0, 102)
        ax.set_ylabel("지표 값 (%)", fontsize=10)
        ax.set_title(title, fontsize=11.8, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle="--", alpha=0.38)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#CBD5E1", edgecolor="#64748B", label="학습 (1~6월 · 7,200건)"),
        Patch(facecolor="#DC2626", edgecolor="#0F172A", label="평가 (7~8월 · 2,400건, 6번째 신규 모델=빨강 강조)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, frameon=True, fontsize=9.8)
    fig.suptitle(
        "6개 모델 학습(1~6월) vs 평가(7~8월) 성능 지표 비교 (50% 임계값 · 6번째: 로지스틱 가중치 C=1, 1:3)",
        fontsize=14.2,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(OUT_PNG_METRICS, dpi=180)
    plt.close(fig)


def save_confusion_png(comp_df: pd.DataFrame) -> None:
    """Create 11_6개모델_혼동행렬_비교.png (2x3 grid, rows=Pred, cols=Actual)."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.8))
    for ax, (_, row), spec in zip(axes.flatten(), comp_df.iterrows(), SIX_MODELS):
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
                (x0 + pad, y0 + pad), 1 - 2 * pad, 1 - 2 * pad,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=bg, edgecolor=edge, linewidth=2.1 if spec["key"] == "lr_w13" else 1.7,
            )
            ax.add_patch(rect)
            ax.text(x0 + 0.5, y0 + 0.74, label, ha="center", va="center", fontsize=8.8, fontweight="bold", color=tcol)
            ax.text(x0 + 0.5, y0 + 0.44, f"{cnt:,}건", ha="center", va="center", fontsize=15.0, fontweight="bold", color="#0F172A")
            ax.text(x0 + 0.5, y0 + 0.20, f"({cnt / tot * 100:.2f}%)", ha="center", va="center", fontsize=8.5, color="#475569")

        ax.text(0.5, 2.04, "실제 정상", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#334155")
        ax.text(1.5, 2.04, "실제 지연", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#334155")
        ax.text(-0.04, 1.5, "정상 예측\n(제외)", ha="right", va="center", fontsize=8.4, fontweight="bold", color="#334155")
        ax.text(-0.04, 0.5, "지연 예측\n(점검)", ha="right", va="center", fontsize=8.4, fontweight="bold", color="#334155")

        prefix = "★ 6번째 · " if spec["key"] == "lr_w13" else ""
        t_color = "#B91C1C" if spec["key"] == "lr_w13" else "#0F172A"
        ax.set_title(
            f"{prefix}{spec['name']} (점검 {int(row['점검대상']):,}건 · {row['점검비율']*100:.2f}%)\n"
            f"정확도 {row['정확도']*100:.2f}% | 정밀도 {row['정밀도']*100:.2f}% | 재현율 {row['재현율']*100:.2f}% | F1 {row['F1']*100:.2f}%",
            fontsize=9.8,
            fontweight="bold",
            color=t_color,
            pad=22,
        )

    fig.suptitle(
        "6개 모델 평가 기간(7~8월 2,400건 · 50% 기준) 혼동행렬 비교 (행=예측 / 열=실제)",
        fontsize=14.4,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(left=0.055, right=0.975, top=0.87, bottom=0.05, wspace=0.25, hspace=0.36)
    fig.savefig(OUT_PNG_CM, dpi=180)
    plt.close(fig)


def save_roc_png(comp_df: pd.DataFrame, thr_df: pd.DataFrame, y_eval: np.ndarray, eval_probs: dict) -> None:
    """Create 12_6개모델_ROC곡선_및_AUC_비교.png (left: 6-model ROC, right: 6th model thresholds & AUC)."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.2, 6.8), dpi=180)

    # Left: 6-model ROC curves + 50% operating dots
    for spec, (_, row) in zip(SIX_MODELS, comp_df.iterrows()):
        probs = eval_probs[spec["key"]]
        fpr, tpr, _ = roc_curve(y_eval, probs)
        auc_v = float(row["ROC_AUC"])
        ax1.plot(fpr, tpr, color=spec["color"], linestyle=spec["ls"], linewidth=spec["lw"],
                 label=f"{spec['name']} (AUC {auc_v:.4f})")
        ax1.scatter([row["FPR"]], [row["재현율"]], color=spec["color"], s=58 if spec["key"] == "lr_w13" else 42,
                    edgecolor="#0F172A", linewidth=0.9, zorder=6)

    ax1.plot([0, 1], [0, 1], color="#94A3B8", linestyle=":", linewidth=1.3, label="무작위 기준선 (AUC 0.5000)")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("FPR = 실제 정상 중 잘못 지연 예측 비율 (FP / 1,984)", fontsize=9.8)
    ax1.set_ylabel("TPR = Recall = 실제 지연 포착 비율 (TP / 416)", fontsize=9.8)
    ax1.set_title("① 6개 모델 ROC 곡선 및 50% 작동 위치 (점=50% 기준)", fontsize=11.6, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", fontsize=8.4, frameon=True)

    # Right: 6th model (C=1, 1:3) ROC with 50%, 30%, 20% threshold callouts
    p6 = eval_probs["lr_w13"]
    fpr6, tpr6, _ = roc_curve(y_eval, p6)
    auc6 = float(comp_df.loc[comp_df["모델"] == "로지스틱 가중치 (C=1, 1:3)", "ROC_AUC"].iloc[0])
    ax2.plot(fpr6, tpr6, color="#DC2626", linewidth=2.6, label=f"로지스틱 가중치 (C=1, 1:3) · AUC {auc6:.4f}")
    ax2.fill_between(fpr6, 0, tpr6, color="#FEE2E2", alpha=0.35)
    ax2.plot([0, 1], [0, 1], color="#94A3B8", linestyle=":", linewidth=1.3, label="무작위 기준선 (AUC 0.5000)")

    thr_offsets = {
        "50%": (0.28, 0.44),
        "30%": (0.48, 0.67),
        "20%": (0.60, 0.80),
    }
    thr_colors = {"50%": "#1D4ED8", "30%": "#D97706", "20%": "#059669"}
    for _, tr in thr_df.iterrows():
        lbl = tr["임계값표기"]
        fx, ty = float(tr["FPR"]), float(tr["재현율"])
        ox, oy = thr_offsets[lbl]
        col = thr_colors[lbl]
        ax2.scatter([fx], [ty], color=col, s=80, edgecolor="#0F172A", linewidth=1.1, zorder=7)
        ax2.annotate(
            f"{lbl} 기준 (점검 {int(tr['점검대상']):,}건)\nTPR {ty*100:.2f}% · FPR {fx*100:.2f}%",
            xy=(fx, ty),
            xytext=(ox, oy),
            fontsize=8.6,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor=col, linewidth=1.3),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
            zorder=8,
        )

    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("FPR (실제 정상 오탐 비율)", fontsize=9.8)
    ax2.set_ylabel("TPR / Recall (실제 지연 포착 비율)", fontsize=9.8)
    ax2.set_title("② 6번째 모델(C=1, 1:3) 임계값(50%·30%·20%)별 ROC 이동", fontsize=11.6, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=8.8, frameon=True)

    fig.suptitle(
        "6개 모델 ROC-AUC 비교 (7~8월 평가 2,400건 · 상위 4개 모델 AUC 0.80대 근접 vs 50% 작동점 차이)",
        fontsize=14.0,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(left=0.06, right=0.975, top=0.86, bottom=0.11, wspace=0.22)
    fig.savefig(OUT_PNG_ROC, dpi=180)
    plt.close(fig)


def save_interactive_htmls(comp_df: pd.DataFrame, te_df: pd.DataFrame, thr_df: pd.DataFrame, y_eval: np.ndarray, eval_probs: dict) -> None:
    """Create interactive Plotly HTMLs for 6-model metrics, confusion matrices, and ROC curves."""
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Interactive 6-model metrics
    metrics = [("정확도", "① 정확도"), ("정밀도", "② 정밀도"), ("재현율", "③ 재현율"), ("F1", "④ F1 Score")]
    pfig = make_subplots(rows=2, cols=2, subplot_titles=[f"<b>{t}</b>" for _, t in metrics], shared_yaxes=True, horizontal_spacing=0.07, vertical_spacing=0.17)
    x_names = [m["name"] for m in SIX_MODELS]
    for idx, ((r_i, c_i), (col, title)) in enumerate(zip([(1, 1), (1, 2), (2, 1), (2, 2)], metrics)):
        tr_sub = [te_df[(te_df["모델"] == m["name"]) & (te_df["구분"] == "학습 (1~6월)")].iloc[0] for m in SIX_MODELS]
        ev_sub = [te_df[(te_df["모델"] == m["name"]) & (te_df["구분"] == "평가 (7~8월)")].iloc[0] for m in SIX_MODELS]
        pfig.add_trace(go.Bar(name="학습 (1~6월)", x=x_names, y=[r[col]*100 for r in tr_sub], marker_color="#CBD5E1", text=[f"{r[col]*100:.2f}%" for r in tr_sub], textposition="outside", showlegend=(idx == 0)), row=r_i, col=c_i)
        pfig.add_trace(go.Bar(name="평가 (7~8월)", x=x_names, y=[r[col]*100 for r in ev_sub], marker_color=[m["color"] for m in SIX_MODELS], text=[f"{r[col]*100:.2f}%" for r in ev_sub], textposition="outside", showlegend=(idx == 0)), row=r_i, col=c_i)
    pfig.update_yaxes(range=[0, 106], title_text="비율 (%)", row=1, col=1)
    pfig.update_yaxes(range=[0, 106], title_text="비율 (%)", row=2, col=1)
    pfig.update_layout(barmode="group", title=dict(text="<b>6개 모델 학습 vs 평가 성능 지표 비교 (50% 기준)</b>", x=0.5, xanchor="center"), template="plotly_white", height=760, legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5), margin=dict(t=105, b=95, l=65, r=35), font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
    pfig.write_html(str(OUT_HTML_METRICS), include_plotlyjs=True)

    # 2. Interactive 6-model confusion matrices
    cm_fig = make_subplots(rows=2, cols=3, subplot_titles=[f"<b>{r['모델']}</b><br>(점검 {int(r['점검대상']):,}건 · F1 {r['F1']*100:.2f}%)" for _, r in comp_df.iterrows()], horizontal_spacing=0.09, vertical_spacing=0.19)
    for (r_i, c_i), (_, row) in zip([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)], comp_df.iterrows()):
        tp, fp, fn, tn, tot = int(row["찾아낸_실제지연(TP)"]), int(row["점검대상_실제정상(FP)"]), int(row["놓친_실제지연(FN)"]), int(row["점검제외_실제정상(TN)"]), int(row["전체주문"])
        z = [[fp, tp], [tn, fn]]
        txt = [[f"<b>{fp:,}건</b><br>FP · {fp/tot*100:.2f}%", f"<b>{tp:,}건</b><br>TP · {tp/tot*100:.2f}%"], [f"<b>{tn:,}건</b><br>TN · {tn/tot*100:.2f}%", f"<b>{fn:,}건</b><br>FN · {fn/tot*100:.2f}%"]]
        cm_fig.add_trace(go.Heatmap(z=z, x=["실제 정상", "실제 지연"], y=["지연 예측(점검)", "정상 예측(제외)"], text=txt, texttemplate="%{text}", textfont=dict(size=11.5, color="#0F172A"), colorscale=[[0.0, "#F8FAFC"], [0.2, "#FEF3C7"], [0.45, "#DBEAFE"], [1.0, "#93C5FD"]], showscale=False, xgap=4, ygap=4), row=r_i, col=c_i)
    cm_fig.update_layout(title=dict(text="<b>6개 모델 평가(7~8월 2,400건) 혼동행렬 비교 (50% 기준)</b>", x=0.5, xanchor="center"), template="plotly_white", height=760, margin=dict(t=105, b=50, l=75, r=35), font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
    cm_fig.write_html(str(OUT_HTML_CM), include_plotlyjs=True)

    # 3. Interactive ROC curves (model_roc_12.html — toggleable per model + 50%/30%/20% threshold points)
    rfig = make_subplots(rows=1, cols=2, subplot_titles=("<b>① 6개 모델 ROC 곡선 (범례 클릭으로 개별 선택)</b>", "<b>② 6번째 모델(C=1, 1:3) 임계값별 ROC 위치</b>"), horizontal_spacing=0.10)
    for spec, (_, row) in zip(SIX_MODELS, comp_df.iterrows()):
        probs = eval_probs[spec["key"]]
        fpr, tpr, _ = roc_curve(y_eval, probs)
        dash = "solid" if spec["ls"] == "-" else ("dash" if spec["ls"] == "--" else ("dot" if spec["ls"] == ":" else "dashdot"))
        rfig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{spec['name']} (AUC {row['ROC_AUC']:.4f})",
                line=dict(color=spec["color"], width=spec["lw"] + 0.5, dash=dash),
                legendgroup=spec["key"],
                hovertemplate=f"<b>{spec['name']}</b><br>FPR: %{{x:.4f}}<br>TPR(Recall): %{{y:.4f}}<extra></extra>",
            ),
            row=1, col=1,
        )
        rfig.add_trace(
            go.Scatter(
                x=[row["FPR"]], y=[row["재현율"]], mode="markers",
                marker=dict(color=spec["color"], size=10, line=dict(color="#0F172A", width=1.2)),
                legendgroup=spec["key"], showlegend=False,
                hovertemplate=f"<b>{spec['name']} (50% 기준)</b><br>점검: {int(row['점검대상']):,}건<br>TPR: {row['재현율']*100:.2f}% | FPR: {row['FPR']*100:.2f}%<br>Precision: {row['정밀도']*100:.2f}% | F1: {row['F1']*100:.2f}%<extra></extra>",
            ),
            row=1, col=1,
        )
    rfig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="무작위 기준선 (0.5000)", line=dict(color="#94A3B8", dash="dot")), row=1, col=1)

    p6 = eval_probs["lr_w13"]
    fpr6, tpr6, _ = roc_curve(y_eval, p6)
    rfig.add_trace(go.Scatter(x=fpr6, y=tpr6, mode="lines", line=dict(color="#DC2626", width=2.8), showlegend=False, hovertemplate="FPR: %{x:.4f}<br>TPR: %{y:.4f}<extra></extra>"), row=1, col=2)
    rfig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#94A3B8", dash="dot"), showlegend=False), row=1, col=2)
    for _, tr in thr_df.iterrows():
        rfig.add_trace(
            go.Scatter(
                x=[tr["FPR"]], y=[tr["재현율"]], mode="markers+text",
                marker=dict(size=11, color="#1D4ED8" if tr["임계값표기"] == "50%" else ("#D97706" if tr["임계값표기"] == "30%" else "#059669"), line=dict(color="#0F172A", width=1.2)),
                text=[f"{tr['임계값표기']} ({int(tr['점검대상']):,}건)"],
                textposition="bottom right",
                showlegend=False,
                hovertemplate=f"<b>{tr['임계값표기']} 기준</b><br>점검: {int(tr['점검대상']):,}건<br>TPR(Recall): {tr['재현율']*100:.2f}%<br>FPR: {tr['FPR']*100:.2f}%<br>Precision: {tr['정밀도']*100:.2f}%<extra></extra>",
            ),
            row=1, col=2,
        )

    rfig.update_xaxes(title_text="FPR (실제 정상 오탐 비율)", range=[-0.02, 1.02], row=1, col=1)
    rfig.update_xaxes(title_text="FPR (실제 정상 오탐 비율)", range=[-0.02, 1.02], row=1, col=2)
    rfig.update_yaxes(title_text="TPR / Recall (실제 지연 포착 비율)", range=[-0.02, 1.02], row=1, col=1)
    rfig.update_yaxes(title_text="TPR / Recall", range=[-0.02, 1.02], row=1, col=2)
    rfig.update_layout(
        title=dict(text="<b>6개 모델 ROC-AUC 비교 및 6번째 모델(C=1, 1:3) 임계값 이동 (7~8월 2,400건)</b>", x=0.5, xanchor="center"),
        template="plotly_white",
        height=580,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(t=85, b=120, l=65, r=30),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    rfig.write_html(str(OUT_HTML_ROC), include_plotlyjs=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    comp_df, te_df, thr_df, y_eval, eval_probs = evaluate_six_models()
    save_metrics_png(te_df)
    save_confusion_png(comp_df)
    save_roc_png(comp_df, thr_df, y_eval, eval_probs)
    save_interactive_htmls(comp_df, te_df, thr_df, y_eval, eval_probs)

    print("=== 6개 모델 7~8월 평가 결과 (50% 기준) ===")
    for _, r in comp_df.iterrows():
        print(
            f"{r['모델']} | 점검={int(r['점검대상'])}건({r['점검비율']*100:.2f}%), "
            f"TP={int(r['찾아낸_실제지연(TP)'])}, FP={int(r['점검대상_실제정상(FP)'])}, "
            f"FN={int(r['놓친_실제지연(FN)'])}, TN={int(r['점검제외_실제정상(TN)'])} | "
            f"Acc={r['정확도']*100:.2f}%, Prec={r['정밀도']*100:.2f}%, "
            f"Rec={r['재현율']*100:.2f}%, F1={r['F1']*100:.2f}%, "
            f"FPR={r['FPR']*100:.2f}%, ROC-AUC={r['ROC_AUC']:.4f}"
        )
    print(f"\nSaved: {OUT_COMP_CSV}")
    print(f"Saved: {OUT_TE_CSV}")
    print(f"Saved: {OUT_THRESH_CSV}")


if __name__ == "__main__":
    main()
