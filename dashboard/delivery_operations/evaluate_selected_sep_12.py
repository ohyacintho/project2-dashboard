"""Confirm selected model (Random Forest balanced, 50% threshold) and run September final evaluation for Session 12."""

import json
import shutil
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
SOURCE_JOBLIB = PREV11_RESULTS_DIR / "delivery_forest_balanced_10.joblib"
COMP_12_CSV = RESULTS_DIR / "model_comparison_12.csv"

OUT_SELECTED_JOBLIB = RESULTS_DIR / "selected_model_12.joblib"
OUT_SELECTED_JSON = RESULTS_DIR / "selected_model_12.json"
OUT_SEP_EVAL_CSV = RESULTS_DIR / "delivery_evaluation_sep_12.csv"
OUT_SEP_SUMMARY_CSV = RESULTS_DIR / "sep_evaluation_summary_12.csv"

OUT_PNG = GRAPHS_DIR / "13_선정모델_가중치RF_7~8월vs9월_최종평가_비교.png"
OUT_HTML = INTERACTIVE_DIR / "13_selected_rf_sep_evaluation_12.html"

MODEL_NAME = "가중치 적용 랜덤포레스트 (balanced)"
THRESHOLD = 0.50


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
    actual_pos = tp + fn
    actual_neg = fp + tn
    inspect = tp + fp

    acc = safe_div(tp + tn, total)
    prec = safe_div(tp, inspect)
    rec = safe_div(tp, actual_pos)
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
    fpr = safe_div(fp, actual_neg)
    return {
        "전체주문": total,
        "실제지연": actual_pos,
        "실제지연율": safe_div(actual_pos, total),
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


def save_selected_and_evaluate_sep():
    """Lock in selected_model_12.joblib/.json (from 7~8월 comparison) and evaluate 9월 data."""
    with open(REF_JSON, encoding="utf-8") as f:
        ref = json.load(f)

    shutil.copy2(SOURCE_JOBLIB, OUT_SELECTED_JOBLIB)
    pipe = joblib.load(OUT_SELECTED_JOBLIB)

    raw_df = pd.read_csv(FEATURES_CSV)
    dt = pd.to_datetime(raw_df[ref["split_column"]])
    eval_df = raw_df[(dt >= "2026-07-01") & (dt < "2026-09-01")].copy()
    sep_df = raw_df[(dt >= "2026-09-01") & (dt < "2026-10-01")].copy()

    input_cols = ref["numeric_inputs"] + ref["categorical_inputs"]
    target_col = ref["target"]

    # Verify parity on 7~8월 vs model_comparison_12.csv
    p_eval = pipe.predict_proba(eval_df[input_cols])[:, 1]
    y_eval = eval_df[target_col].to_numpy()
    m_eval = compute_metrics(y_eval, p_eval, THRESHOLD)

    comp12 = pd.read_csv(COMP_12_CSV)
    r_prev = comp12[comp12["모델"] == MODEL_NAME].iloc[0]
    assert abs(float(r_prev["ROC_AUC"]) - m_eval["ROC_AUC"]) < 1e-12
    assert int(r_prev["찾아낸_실제지연(TP)"]) == m_eval["찾아낸_실제지연(TP)"]

    # Save selected_model_12.json (7~8월 comparison basis only; 9월 not used for selection)
    est = pipe.named_steps["model"]
    est_params = {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v)) for k, v in est.get_params().items()}
    sel_payload = {
        "model_name": MODEL_NAME,
        "estimator": type(est).__name__,
        "source_joblib": "11회/results/delivery_forest_balanced_10.joblib",
        "selected_joblib": OUT_SELECTED_JOBLIB.name,
        "numeric_inputs": ref["numeric_inputs"],
        "categorical_inputs": ref["categorical_inputs"],
        "actual_parameters": est_params,
        "train_period": ["2026-01-01", "2026-06-30"],
        "comparison_period": ["2026-07-01", "2026-08-31"],
        "final_evaluation_period": ["2026-09-01", "2026-09-30"],
        "operations_period_unused_outcomes": ["2026-10-01", "2026-10-20"],
        "threshold": THRESHOLD,
        "positive_label": 1,
        "selection_reason": "7~8월 비교에서 실제 지연 416건 중 315건(재현율 75.72%)을 사전에 찾아내어 후보 모델 중 가장 높은 지연 포착 성능을 보였기 때문입니다.",
        "tradeoff": "50% 기준 점검 대상이 873건(36.38%)으로 가장 많고 그중 558건이 실제 정상 주문(정밀도 36.08%)이어서 높은 점검 물량과 오탐 부담을 감수해야 합니다.",
        "selected_at": "2026-11-28",
    }
    OUT_SELECTED_JSON.write_text(json.dumps(sel_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Predict September final evaluation without retraining or altering threshold
    p_sep = pipe.predict_proba(sep_df[input_cols])[:, 1]
    y_sep = sep_df[target_col].to_numpy()
    m_sep = compute_metrics(y_sep, p_sep, THRESHOLD)

    sep_out = sep_df[["주문ID", "점검기준일시", "배송여유일수", "재고확보상태", "물류센터", "배송지연여부"]].copy()
    sep_out["지연예측확률"] = p_sep.astype(float)
    sep_out["위험여부"] = (p_sep >= THRESHOLD).astype(int)
    sep_out = sep_out.sort_values(["점검기준일시", "주문ID"]).reset_index(drop=True)
    sep_out.to_csv(OUT_SEP_EVAL_CSV, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame([
        {"기간구분": "7~8월 (선정 비교)", "기간": "2026-07-01 ~ 2026-08-20", "모델": MODEL_NAME, **m_eval},
        {"기간구분": "9월 (최종 평가)", "기간": "2026-09-01 ~ 2026-09-30", "모델": MODEL_NAME, **m_sep},
    ])
    summary_df.to_csv(OUT_SEP_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    return summary_df, y_eval, p_eval, y_sep, p_sep


def save_static_png(summary_df: pd.DataFrame, y_eval: np.ndarray, p_eval: np.ndarray, y_sep: np.ndarray, p_sep: np.ndarray) -> None:
    """Create 13_선정모델_가중치RF_7~8월vs9월_최종평가_비교.png (2x2 panels)."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    r_ev = summary_df.iloc[0]
    r_sep = summary_df.iloc[1]

    fig = plt.figure(figsize=(16.4, 10.2), dpi=180)
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.25)

    # Panel 1: Metrics bar comparison (7~8월 vs 9월)
    ax1 = fig.add_subplot(gs[0, 0])
    m_cols = ["정확도", "정밀도", "재현율", "F1"]
    m_labels = ["정확도\n(Accuracy)", "정밀도\n(Precision)", "재현율\n(Recall)", "F1 Score"]
    x = np.arange(len(m_cols))
    w = 0.34
    v_ev = [float(r_ev[c]) * 100 for c in m_cols]
    v_sep = [float(r_sep[c]) * 100 for c in m_cols]

    b1 = ax1.bar(x - w / 2, v_ev, w, label="7~8월 (2,400건 · 지연 416건)", color="#94A3B8", edgecolor="#475569", linewidth=0.6)
    b2 = ax1.bar(x + w / 2, v_sep, w, label="9월 최종 평가 (1,455건 · 지연 223건)", color="#F59E0B", edgecolor="#92400E", linewidth=0.7)
    for bar, v in zip(b1, v_ev):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.3, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#334155")
    for bar, v in zip(b2, v_sep):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.3, f"{v:.2f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#92400E")

    ax1.set_xticks(x)
    ax1.set_xticklabels(m_labels, fontsize=9.6, fontweight="bold")
    ax1.set_ylim(0, 96)
    ax1.set_ylabel("지표 값 (%)", fontsize=9.8)
    ax1.set_title("① 7~8월 vs 9월 핵심 성능 지표 비교 (50% 기준)", fontsize=11.6, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Panel 2: ROC Curve & AUC comparison (7~8월 vs 9월)
    ax2 = fig.add_subplot(gs[0, 1])
    fpr_ev, tpr_ev, _ = roc_curve(y_eval, p_eval)
    fpr_sep, tpr_sep, _ = roc_curve(y_sep, p_sep)

    ax2.plot(fpr_ev, tpr_ev, color="#64748B", linestyle="--", linewidth=2.0, label=f"7~8월 ROC (AUC {r_ev['ROC_AUC']:.4f})")
    ax2.plot(fpr_sep, tpr_sep, color="#D97706", linestyle="-", linewidth=2.6, label=f"9월 최종 평가 ROC (AUC {r_sep['ROC_AUC']:.4f})")
    ax2.plot([0, 1], [0, 1], color="#CBD5E1", linestyle=":", linewidth=1.3, label="무작위 기준선 (AUC 0.5000)")

    ax2.scatter([r_ev["FPR"]], [r_ev["재현율"]], color="#64748B", s=65, edgecolor="#0F172A", linewidth=1.0, zorder=6)
    ax2.scatter([r_sep["FPR"]], [r_sep["재현율"]], color="#D97706", s=85, edgecolor="#0F172A", linewidth=1.1, zorder=7)
    ax2.annotate(
        f"7~8월 50%: TPR {r_ev['재현율']*100:.2f}%, FPR {r_ev['FPR']*100:.2f}%",
        xy=(r_ev["FPR"], r_ev["재현율"]), xytext=(0.40, 0.72),
        fontsize=8.3, fontweight="bold", color="#334155",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor="#64748B"),
        arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.3),
    )
    ax2.annotate(
        f"9월 50%: TPR {r_sep['재현율']*100:.2f}%, FPR {r_sep['FPR']*100:.2f}%",
        xy=(r_sep["FPR"], r_sep["재현율"]), xytext=(0.40, 0.52),
        fontsize=8.5, fontweight="bold", color="#92400E",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB", edgecolor="#D97706"),
        arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.5),
    )
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlabel("FPR (실제 정상 오탐 비율)", fontsize=9.6)
    ax2.set_ylabel("TPR / Recall (실제 지연 포착 비율)", fontsize=9.6)
    ax2.set_title("② 7~8월 vs 9월 ROC 곡선 및 AUC 비교", fontsize=11.6, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=8.6, frameon=True)

    # Panel 3 & 4: Confusion matrices (7~8월 vs 9월, rows=Pred, cols=Actual)
    for ax, row, title_cm, is_sep in [
        (fig.add_subplot(gs[1, 0]), r_ev, "③ 7~8월 선정 비교 혼동행렬 (2,400건 · 점검 873건 / 36.38%)", False),
        (fig.add_subplot(gs[1, 1]), r_sep, "④ 9월 최종 평가 혼동행렬 (1,455건 · 점검 501건 / 34.43%)", True),
    ]:
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
                facecolor=bg, edgecolor=edge, linewidth=2.0 if is_sep else 1.6,
            )
            ax.add_patch(rect)
            ax.text(x0 + 0.5, y0 + 0.74, label, ha="center", va="center", fontsize=9.2, fontweight="bold", color=tcol)
            ax.text(x0 + 0.5, y0 + 0.44, f"{cnt:,}건", ha="center", va="center", fontsize=16.0, fontweight="bold", color="#0F172A")
            ax.text(x0 + 0.5, y0 + 0.20, f"({cnt / tot * 100:.2f}%)", ha="center", va="center", fontsize=9.0, color="#475569")

        ax.text(0.5, 2.04, "실제 정상", ha="center", va="bottom", fontsize=9.6, fontweight="bold", color="#334155")
        ax.text(1.5, 2.04, "실제 지연", ha="center", va="bottom", fontsize=9.6, fontweight="bold", color="#334155")
        ax.text(-0.04, 1.5, "정상 예측\n(제외)", ha="right", va="center", fontsize=8.8, fontweight="bold", color="#334155")
        ax.text(-0.04, 0.5, "지연 예측\n(점검)", ha="right", va="center", fontsize=8.8, fontweight="bold", color="#334155")
        ax.set_title(title_cm, fontsize=11.0, fontweight="bold", color="#92400E" if is_sep else "#0F172A", pad=22)

    fig.suptitle(
        "최종 선정 모델 [가중치 적용 랜덤포레스트 (balanced) · 50% 기준] — 7~8월 vs 9월 최종 평가 비교",
        fontsize=14.4,
        fontweight="bold",
        y=0.975,
    )
    plt.subplots_adjust(left=0.075, right=0.975, top=0.89, bottom=0.06)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)


def save_interactive_html(summary_df: pd.DataFrame, y_eval: np.ndarray, p_eval: np.ndarray, y_sep: np.ndarray, p_sep: np.ndarray) -> None:
    """Create interactive Plotly HTML comparing 7~8월 and 9월 evaluation."""
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    r_ev = summary_df.iloc[0]
    r_sep = summary_df.iloc[1]

    pfig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>① 7~8월 vs 9월 성능 지표 (%)</b>", "<b>② 7~8월 vs 9월 ROC 곡선 및 AUC</b>"),
        horizontal_spacing=0.10,
    )
    m_cols = ["정확도", "정밀도", "재현율", "F1"]
    m_names = ["정확도", "정밀도", "재현율", "F1 Score"]
    pfig.add_trace(go.Bar(name="7~8월 (2,400건)", x=m_names, y=[float(r_ev[c])*100 for c in m_cols], marker_color="#94A3B8", text=[f"{float(r_ev[c])*100:.2f}%" for c in m_cols], textposition="outside"), row=1, col=1)
    pfig.add_trace(go.Bar(name="9월 최종 평가 (1,455건)", x=m_names, y=[float(r_sep[c])*100 for c in m_cols], marker_color="#F59E0B", text=[f"{float(r_sep[c])*100:.2f}%" for c in m_cols], textposition="outside"), row=1, col=1)

    fpr_ev, tpr_ev, _ = roc_curve(y_eval, p_eval)
    fpr_sep, tpr_sep, _ = roc_curve(y_sep, p_sep)
    pfig.add_trace(go.Scatter(x=fpr_ev, y=tpr_ev, mode="lines", name=f"7~8월 ROC (AUC {r_ev['ROC_AUC']:.4f})", line=dict(color="#64748B", width=2.2, dash="dash")), row=1, col=2)
    pfig.add_trace(go.Scatter(x=fpr_sep, y=tpr_sep, mode="lines", name=f"9월 ROC (AUC {r_sep['ROC_AUC']:.4f})", line=dict(color="#D97706", width=2.8)), row=1, col=2)
    pfig.add_trace(go.Scatter(x=[r_ev["FPR"], r_sep["FPR"]], y=[r_ev["재현율"], r_sep["재현율"]], mode="markers", name="50% 기준 작동점", marker=dict(size=10, color=["#64748B", "#D97706"], line=dict(color="#0F172A", width=1.2))), row=1, col=2)
    pfig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="무작위 기준선 (0.5000)", line=dict(color="#CBD5E1", dash="dot")), row=1, col=2)

    pfig.update_yaxes(range=[0, 96], title_text="비율 (%)", row=1, col=1)
    pfig.update_xaxes(title_text="FPR (정상 오탐 비율)", range=[-0.02, 1.02], row=1, col=2)
    pfig.update_yaxes(title_text="TPR / Recall", range=[-0.02, 1.02], row=1, col=2)
    pfig.update_layout(
        barmode="group",
        title=dict(text="<b>최종 선정 모델 [가중치 적용 랜덤포레스트 (balanced)] 9월 최종 평가</b>", x=0.5, xanchor="center"),
        template="plotly_white",
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
        margin=dict(t=80, b=110, l=65, r=30),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    pfig.write_html(str(OUT_HTML), include_plotlyjs=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    summary_df, y_eval, p_eval, y_sep, p_sep = save_selected_and_evaluate_sep()
    save_static_png(summary_df, y_eval, p_eval, y_sep, p_sep)
    save_interactive_html(summary_df, y_eval, p_eval, y_sep, p_sep)

    print("=== 선정 모델 (가중치 적용 랜덤포레스트 balanced · 50% 기준) 평가 비교 ===")
    for _, r in summary_df.iterrows():
        print(
            f"{r['기간구분']} ({r['기간']}) | 전체={int(r['전체주문'])}건, 실제지연={int(r['실제지연'])}건({r['실제지연율']*100:.2f}%), "
            f"점검={int(r['점검대상'])}건({r['점검비율']*100:.2f}%), TP={int(r['찾아낸_실제지연(TP)'])}, "
            f"FP={int(r['점검대상_실제정상(FP)'])}, FN={int(r['놓친_실제지연(FN)'])}, TN={int(r['점검제외_실제정상(TN)'])} | "
            f"Acc={r['정확도']*100:.2f}%, Prec={r['정밀도']*100:.2f}%, Rec={r['재현율']*100:.2f}%, "
            f"F1={r['F1']*100:.2f}%, FPR={r['FPR']*100:.2f}%, ROC-AUC={r['ROC_AUC']:.4f}, PR-AUC={r['PR_AUC']:.4f}"
        )
    print(f"\nSaved model: {OUT_SELECTED_JOBLIB} & {OUT_SELECTED_JSON}")
    print(f"Saved Sep predictions CSV: {OUT_SEP_EVAL_CSV}")
    print(f"Saved Sep summary CSV:     {OUT_SEP_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
