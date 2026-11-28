"""Analyze numeric feature scaling (Z-score vs Min-Max) reflecting the Session 12 PDF on real shipping data."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREV11_RESULTS_DIR = BASE_DIR / "11회" / "results"
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
INTERACTIVE_DIR = GRAPHS_DIR / "interactive"

FEATURES_CSV = DATA_DIR / "delivery_features_09.csv"
REF_JSON = DATA_DIR / "delivery_model_reference.json"
BASE_LOGISTIC_JOBLIB = PREV11_RESULTS_DIR / "delivery_logistic_11.joblib"

OUT_STATS_CSV = RESULTS_DIR / "pdf_숫자형_변수_스케일링_기준통계.csv"
OUT_ORDERS_CSV = RESULTS_DIR / "pdf_숫자형_변수_스케일링_실제주문5건.csv"

OUT_PNG_DIST = GRAPHS_DIR / "05_PDF_숫자형_변수_스케일링_원리_및_분포비교.png"
OUT_PNG_ORDERS = GRAPHS_DIR / "06_PDF_로지스틱_표준화_실제주문5건_계산과정.png"

OUT_HTML_DIST = INTERACTIVE_DIR / "05_pdf_scaling_zscore_minmax_12.html"
OUT_HTML_ORDERS = INTERACTIVE_DIR / "06_pdf_scaling_5orders_12.html"
OUT_REPORT_HTML = RESULTS_DIR / "PDF_숫자형_변수_스케일링_분석.html"


def compute_scaling_tables():
    """Reuse existing fitted scaler/model in delivery_logistic_11.joblib (no re-fitting)."""
    with open(REF_JSON, encoding="utf-8") as f:
        ref = json.load(f)

    raw_df = pd.read_csv(FEATURES_CSV)
    dt = pd.to_datetime(raw_df[ref["split_column"]])
    train_df = raw_df[(dt >= "2026-01-01") & (dt < "2026-07-01")].copy()
    eval_df = raw_df[(dt >= "2026-07-01") & (dt < "2026-09-01")].copy()
    sep_df = raw_df[(dt >= "2026-09-01") & (dt < "2026-10-01")].copy()

    num_cols = ref["numeric_inputs"]
    pipe = joblib.load(BASE_LOGISTIC_JOBLIB)
    scaler = pipe.named_steps["numeric_scaler"].named_transformers_["num_scale"]
    lr = pipe.named_steps["model"]

    stats_rows = []
    for i, col in enumerate(num_cols):
        tr_vals = train_df[col].to_numpy(dtype=float)
        ev_vals = eval_df[col].to_numpy(dtype=float)
        sep_vals = sep_df[col].to_numpy(dtype=float)

        tr_mean = float(scaler.mean_[i])
        tr_std = float(scaler.scale_[i])
        assert abs(tr_mean - float(tr_vals.mean())) < 1e-9
        assert abs(tr_std - float(tr_vals.std(ddof=0))) < 1e-9

        tr_min = float(tr_vals.min())
        tr_max = float(tr_vals.max())
        tr_span = tr_max - tr_min

        ev_mm = (ev_vals - tr_min) / tr_span
        sep_mm = (sep_vals - tr_min) / tr_span
        ev_out = int(((ev_vals < tr_min) | (ev_vals > tr_max)).sum())
        sep_out = int(((sep_vals < tr_min) | (sep_vals > tr_max)).sum())

        std_coef = float(lr.coef_[0][i])
        raw_coef = std_coef / tr_std

        stats_rows.append({
            "변수명": col,
            "학습_평균(saved)": tr_mean,
            "학습_표준편차(saved)": tr_std,
            "학습_최솟값": tr_min,
            "학습_최댓값": tr_max,
            "평가_최솟값": float(ev_vals.min()),
            "평가_최댓값": float(ev_vals.max()),
            "평가_MinMax_최소": float(ev_mm.min()),
            "평가_MinMax_최대": float(ev_mm.max()),
            "평가_MinMax_범위이탈건수": ev_out,
            "9월_MinMax_범위이탈건수": sep_out,
            "표준화전_원단위1당_계수": raw_coef,
            "표준화후_Zscore_계수": std_coef,
            "표준화후_계수절댓값": abs(std_coef),
        })

    stats_df = pd.DataFrame(stats_rows).sort_values("표준화후_계수절댓값", ascending=False).reset_index(drop=True)
    stats_df.to_csv(OUT_STATS_CSV, index=False, encoding="utf-8-sig")

    # Actual 5 orders (first 5 rows of 7~8월 evaluation period)
    sample5 = eval_df.head(5).copy().reset_index(drop=True)
    ohe = pipe.named_steps["ohe_preprocessor"]
    scaler_full = pipe.named_steps["numeric_scaler"]
    input_cols = ref["numeric_inputs"] + ref["categorical_inputs"]
    X_sample_scaled = scaler_full.transform(ohe.transform(sample5[input_cols]))

    dist_idx = num_cols.index("배송거리_km")
    wait_idx = num_cols.index("센터출고대기건수")
    dist_min, dist_max = float(train_df["배송거리_km"].min()), float(train_df["배송거리_km"].max())
    wait_min, wait_max = float(train_df["센터출고대기건수"].min()), float(train_df["센터출고대기건수"].max())

    order_rows = []
    for row_idx, r in sample5.iterrows():
        d_raw = float(r["배송거리_km"])
        w_raw = float(r["센터출고대기건수"])
        d_z = float(X_sample_scaled[row_idx, dist_idx])
        w_z = float(X_sample_scaled[row_idx, wait_idx])

        # Verify manual formula matches saved scaler exactly
        d_z_calc = (d_raw - float(scaler.mean_[dist_idx])) / float(scaler.scale_[dist_idx])
        w_z_calc = (w_raw - float(scaler.mean_[wait_idx])) / float(scaler.scale_[wait_idx])
        assert abs(d_z - d_z_calc) < 1e-12
        assert abs(w_z - w_z_calc) < 1e-12

        order_rows.append({
            "주문ID": str(r["주문ID"]),
            "점검기준일시": str(r["점검기준일시"]),
            "물류센터": str(r["물류센터"]),
            "배송거리_km_원본": d_raw,
            "배송거리_Zscore": d_z,
            "배송거리_MinMax": (d_raw - dist_min) / (dist_max - dist_min),
            "센터출고대기건수_원본": w_raw,
            "센터출고대기건수_Zscore": w_z,
            "센터출고대기건수_MinMax": (w_raw - wait_min) / (wait_max - wait_min),
        })

    orders_df = pd.DataFrame(order_rows)
    orders_df.to_csv(OUT_ORDERS_CSV, index=False, encoding="utf-8-sig")

    # Also capture out-of-range Min-Max orders for empirical Page-3 check
    w_min, w_max = float(train_df["총중량_kg"].min()), float(train_df["총중량_kg"].max())
    w_idx = num_cols.index("총중량_kg")
    oor_rows = []
    for split_name, sub_df in [("평가 (7~8월)", eval_df), ("최종평가 (9월)", sep_df)]:
        sub_oor = sub_df[(sub_df["총중량_kg"] < w_min) | (sub_df["총중량_kg"] > w_max)]
        for _, rr in sub_oor.iterrows():
            val = float(rr["총중량_kg"])
            oor_rows.append({
                "기간": split_name,
                "주문ID": str(rr["주문ID"]),
                "점검기준일시": str(rr["점검기준일시"]),
                "변수명": "총중량_kg",
                "원본값": val,
                "학습범위": f"[{w_min:.2f}, {w_max:.2f}]",
                "MinMax값": (val - w_min) / (w_max - w_min),
                "Zscore값": (val - float(scaler.mean_[w_idx])) / float(scaler.scale_[w_idx]),
            })
    oor_df = pd.DataFrame(oor_rows)

    return train_df, eval_df, stats_df, orders_df, oor_df, scaler


def save_static_png_overview(train_df, eval_df, stats_df, orders_df, oor_df, scaler) -> None:
    """Create 05_PDF_숫자형_변수_스케일링_원리_및_분포비교.png (4 panels covering PDF pp.1-4)."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16.6, 10.6), dpi=180)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.28)

    # Panel 1 (PDF p.1): Standardized vs Unstandardized Coefficients
    ax1 = fig.add_subplot(gs[0, 0])
    s_sorted = stats_df.sort_values("표준화후_계수절댓값", ascending=True)
    y_pos = np.arange(len(s_sorted))
    std_vals = s_sorted["표준화후_Zscore_계수"].to_numpy()
    raw_vals = s_sorted["표준화전_원단위1당_계수"].to_numpy()
    cols_c = ["#EF4444" if v > 0 else "#2563EB" for v in std_vals]

    bars = ax1.barh(y_pos, std_vals, color=cols_c, height=0.60, edgecolor="#1E293B", linewidth=0.5)
    ax1.axvline(0, color="#0F172A", linewidth=1.0)
    for idx, (bar, sv, rv) in enumerate(zip(bars, std_vals, raw_vals)):
        if sv >= 0:
            ax1.text(sv + 0.03, bar.get_y() + bar.get_height() / 2, f"{sv:+.4f}  (원단위 {rv:+.4f})", va="center", ha="left", fontsize=8.0, fontweight="bold", color="#0F172A")
        else:
            ax1.text(sv - 0.03, bar.get_y() + bar.get_height() / 2, f"(원단위 {rv:+.4f})  {sv:+.4f}", va="center", ha="right", fontsize=8.0, fontweight="bold", color="#0F172A")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(s_sorted["변수명"], fontsize=9.0, fontweight="bold")
    ax1.set_xlim(-1.95, 1.42)
    ax1.set_xlabel("로지스틱 기본(C=1) 계수 (Z-score 1표준편차 기준 vs 괄호: 원단위 1당)", fontsize=9.0)
    ax1.set_title("① [PDF p.1] 표준화로 단위 차이 해소 및 계수 공정 비교", fontsize=11.4, fontweight="bold", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2 (PDF p.2): Distribution shape preserved — 배송거리_km (Raw vs Z-score vs Min-Max)
    # Use inset / twin or side-by-side clean 3-tick x-axis or 2 stacked mini-histograms
    ax2 = fig.add_subplot(gs[0, 1])
    ev_dist = eval_df["배송거리_km"].to_numpy(dtype=float)
    tr_dist_mean = float(stats_df.loc[stats_df["변수명"] == "배송거리_km", "학습_평균(saved)"].iloc[0])
    tr_dist_std = float(stats_df.loc[stats_df["변수명"] == "배송거리_km", "학습_표준편차(saved)"].iloc[0])
    tr_dist_min = float(stats_df.loc[stats_df["변수명"] == "배송거리_km", "학습_최솟값"].iloc[0])
    tr_dist_max = float(stats_df.loc[stats_df["변수명"] == "배송거리_km", "학습_최댓값"].iloc[0])
    ev_dist_z = (ev_dist - tr_dist_mean) / tr_dist_std

    counts, edges, patches = ax2.hist(ev_dist_z, bins=18, color="#3B82F6", alpha=0.78, edgecolor="#FFFFFF")
    ax2.axvline(0, color="#0F172A", linestyle="--", linewidth=1.6, label=f"학습 평균 = 0 ({tr_dist_mean:.2f} km)")
    ax2.axvline(1, color="#D97706", linestyle=":", linewidth=1.6, label=f"+1 표준편차 ({tr_dist_mean + tr_dist_std:.2f} km)")
    # Highlight DLV007201
    d1_z = float(orders_df.loc[0, "배송거리_Zscore"])
    d1_raw = float(orders_df.loc[0, "배송거리_km_원본"])
    ax2.axvline(d1_z, color="#DC2626", linestyle="-", linewidth=2.0, label=f"DLV007201: {int(d1_raw)}km → Z={d1_z:+.2f}")

    z_ticks = np.array([-1.5, -1.0, 0.0, 1.0, 2.0])
    ax2.set_xticks(z_ticks)
    ax2.set_xticklabels(
        [f"Z={z:+.1f}\n({tr_dist_mean + z*tr_dist_std:.0f}km)\n[MM:{(tr_dist_mean + z*tr_dist_std - tr_dist_min)/(tr_dist_max - tr_dist_min):.2f}]" for z in z_ticks],
        fontsize=8.2,
    )
    ax2.set_ylim(0, max(counts) * 1.28)
    ax2.set_ylabel("주문 건수 (7~8월 2,400건)", fontsize=9.2)
    ax2.set_title("② [PDF p.2~3] 배송거리_km 분포 모양 유지 · 눈금만 Z-score / Min-Max 변환", fontsize=11.2, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", fontsize=8.3, frameon=True)

    # Panel 3 (PDF p.3): Min-Max [0, 1] out-of-range check on real data (총중량_kg)
    ax3 = fig.add_subplot(gs[1, 0])
    w_row = stats_df.loc[stats_df["변수명"] == "총중량_kg"].iloc[0]
    w_min, w_max = float(w_row["학습_최솟값"]), float(w_row["학습_최댓값"])
    tr_w_mm = (train_df["총중량_kg"].to_numpy() - w_min) / (w_max - w_min)
    ev_w_mm = (eval_df["총중량_kg"].to_numpy() - w_min) / (w_max - w_min)

    ax3.hist(tr_w_mm, bins=30, range=(-0.02, 1.02), density=True, alpha=0.45, color="#94A3B8", label="학습(1~6월) Min-Max [0, 1]", edgecolor="#FFFFFF")
    ax3.hist(ev_w_mm, bins=30, range=(-0.02, 1.02), density=True, alpha=0.65, color="#0EA5E9", label="평가(7~8월) Min-Max (같은 기준)", edgecolor="#FFFFFF")
    ax3.axvline(0.0, color="#DC2626", linestyle="--", linewidth=1.8, label="학습 최솟값 (0.22kg → 0.0)")
    ax3.axvline(1.0, color="#475569", linestyle="--", linewidth=1.5, label="학습 최댓값 (27.25kg → 1.0)")

    oor_text = "\n".join([
        f"• {r['기간']} {r['주문ID']}: {r['원본값']:.2f}kg → Min-Max {r['MinMax값']:+.4f} (< 0 이탈)"
        for _, r in oor_df.iterrows()
    ])
    ax3.annotate(
        f"실제 데이터 [0, 1] 범위 이탈 확인 (학습 최소 0.22kg 미만):\n{oor_text}",
        xy=(0.0, 2.2),
        xytext=(0.24, 3.6),
        fontsize=8.5,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF2F2", edgecolor="#FCA5A5"),
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.5),
    )
    ax3.set_xlabel("Min-Max 변환값 (학습 최솟값 0.22kg · 최댓값 27.25kg 기준)", fontsize=9.0)
    ax3.set_ylabel("밀도 (Density)", fontsize=9.2)
    ax3.set_title("③ [PDF p.3] 총중량_kg Min-Max 변환 — 새 주문에서 0~1 범위 이탈 실증", fontsize=11.2, fontweight="bold", pad=10)
    ax3.grid(axis="y", linestyle="--", alpha=0.35)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(loc="upper right", fontsize=8.0, frameon=True)

    # Panel 4 (PDF p.4): Train-only fit & reuse flow diagram on our actual pipeline
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 6)
    ax4.axis("off")
    ax4.set_title("④ [PDF p.4] 표준화 기준은 1~6월 학습에서만 구해 저장·재사용", fontsize=11.4, fontweight="bold", pad=10)

    boxes = [
        (0.1, 3.65, 2.95, 1.80, "#EFF6FF", "#3B82F6", "1~6월 학습 자료 (7,200건)", "원래 숫자형 8개 입력\n(평균·표준편차 산출 근거)"),
        (3.48, 3.65, 3.04, 1.80, "#ECFDF5", "#10B981", "변환 기준 저장 (StandardScaler)", "배송거리: 평균 202.50 / SD 118.97\n출고대기: 평균 1,149.02 / SD 363.16"),
        (6.95, 3.65, 2.95, 1.80, "#EEF2FF", "#6366F1", "로지스틱 기본 학습·저장", "delivery_logistic_11.joblib\n(스케일러 + 계수 36개 일체형)"),
        (0.1, 0.75, 2.95, 1.80, "#FFFBEB", "#F59E0B", "7~8월 평가 / 9~10월 새 입력", "DLV007201: 371km, 1,202건\n(새로 평균·표준편차 계산 안 함)"),
        (3.48, 0.75, 3.04, 1.80, "#ECFDF5", "#10B981", "저장된 1~6월 기준으로 변환", "(371 - 202.50) ÷ 118.97 = +1.4163\n(1202 - 1149.02) ÷ 363.16 = +0.1459"),
        (6.95, 0.75, 2.95, 1.80, "#FEF2F2", "#EF4444", "저장된 계수로 확률 예측", "지연 확률 6.22% 계산\n(동일 기준·일관된 예측 재현)"),
    ]
    for bx, by, bw, bh, bg, edge, title_b, body_b in boxes:
        rect = mpatches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.08,rounding_size=0.22", facecolor=bg, edgecolor=edge, linewidth=1.6)
        ax4.add_patch(rect)
        ax4.text(bx + bw / 2, by + bh * 0.73, title_b, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0F172A")
        ax4.text(bx + bw / 2, by + bh * 0.30, body_b, ha="center", va="center", fontsize=7.6, color="#334155", linespacing=1.40)

    arrows = [
        ((3.12, 4.55), (3.42, 4.55)),
        ((6.58, 4.55), (6.88, 4.55)),
        ((3.12, 1.65), (3.42, 1.65)),
        ((6.58, 1.65), (6.88, 1.65)),
        ((5.00, 3.55), (5.00, 2.65)),
        ((8.42, 3.55), (8.42, 2.65)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax4.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8))

    fig.suptitle(
        "[PDF] 숫자형 변수 스케일링(표준화 · Min-Max) 원리 및 실제 배송 데이터 검증",
        fontsize=15.0,
        fontweight="bold",
        y=0.975,
    )
    plt.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.09)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG_DIST, dpi=180)
    plt.close(fig)


def save_static_png_orders(orders_df, stats_df) -> None:
    """Create 06_PDF_로지스틱_표준화_실제주문5건_계산과정.png (5 actual orders + step-by-step math)."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.2, 6.4), dpi=180, gridspec_kw={"width_ratios": [1.18, 1.0]})

    # Left: 5 actual orders (Z-score comparison for 배송거리_km vs 센터출고대기건수)
    y_idx = np.arange(len(orders_df))[::-1]
    h = 0.34
    d_z = orders_df["배송거리_Zscore"].to_numpy()
    w_z = orders_df["센터출고대기건수_Zscore"].to_numpy()

    b_d = ax1.barh(y_idx + h / 2, d_z, height=h, color="#2563EB", label="배송거리_km (Z-score)", edgecolor="#1E3A8A", linewidth=0.5)
    b_w = ax1.barh(y_idx - h / 2, w_z, height=h, color="#0D9488", label="센터출고대기건수 (Z-score)", edgecolor="#115E59", linewidth=0.5)
    ax1.axvline(0, color="#0F172A", linewidth=1.1)

    for idx, r in orders_df.iterrows():
        yi = y_idx[idx]
        dv, dz = r["배송거리_km_원본"], r["배송거리_Zscore"]
        wv, wz = r["센터출고대기건수_원본"], r["센터출고대기건수_Zscore"]

        d_ha = "left" if dz >= 0 else "right"
        d_off = 0.05 if dz >= 0 else -0.05
        ax1.text(dz + d_off, yi + h / 2, f"{dz:+.4f} ({int(dv)}km)", va="center", ha=d_ha, fontsize=8.4, fontweight="bold", color="#1E3A8A")

        w_ha = "left" if wz >= 0 else "right"
        w_off = 0.05 if wz >= 0 else -0.05
        ax1.text(wz + w_off, yi - h / 2, f"{wz:+.4f} ({int(wv):,}건)", va="center", ha=w_ha, fontsize=8.4, fontweight="bold", color="#115E59")

    ax1.set_yticks(y_idx)
    ax1.set_yticklabels([f"{r['주문ID']}\n({r['물류센터']})" for _, r in orders_df.iterrows()], fontsize=9.2, fontweight="bold")
    ax1.set_xlim(-2.45, 2.50)
    ax1.set_xlabel("표준화 후 Z-score 값 (0 = 1~6월 학습 평균)", fontsize=9.5)
    ax1.set_title("① 실제 평가 주문 5건의 원래 값 → Z-score 변환 결과", fontsize=11.5, fontweight="bold", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=8.8, frameon=True)

    # Right: Step-by-step calculation card for DLV007201
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("② DLV007201 (부산센터) 기존 저장 기준값 계산 과정", fontsize=11.5, fontweight="bold", pad=10)

    r0 = orders_df.iloc[0]
    d_row = stats_df.loc[stats_df["변수명"] == "배송거리_km"].iloc[0]
    w_row = stats_df.loc[stats_df["변수명"] == "센터출고대기건수"].iloc[0]

    cards = [
        (0.2, 6.4, 9.6, 3.1, "#EFF6FF", "#3B82F6",
         "1) 배송거리_km 표준화 (Z-score) — 평균보다 약 +1.42 표준편차 먼 거리",
         f"• 저장된 1~6월 학습 기준 (delivery_logistic_11.joblib): 평균 = {d_row['학습_평균(saved)']:.4f} km, 표준편차 = {d_row['학습_표준편차(saved)']:.4f} km\n"
         f"• Z-score 계산: ({int(r0['배송거리_km_원본'])} - {d_row['학습_평균(saved)']:.4f}) ÷ {d_row['학습_표준편차(saved)']:.4f} = {r0['배송거리_Zscore']:+.4f}\n"
         f"• 참고 Min-Max (1~6월 [{int(d_row['학습_최솟값'])}, {int(d_row['학습_최댓값'])}]km): ({int(r0['배송거리_km_원본'])} - {int(d_row['학습_최솟값'])}) ÷ ({int(d_row['학습_최댓값'])} - {int(d_row['학습_최솟값'])}) = {r0['배송거리_MinMax']:.4f}"),
        (0.2, 2.9, 9.6, 3.1, "#ECFDF5", "#10B981",
         "2) 센터출고대기건수 표준화 (Z-score) — 평균보다 약간(+0.15 표준편차) 높은 대기량",
         f"• 저장된 1~6월 학습 기준 (delivery_logistic_11.joblib): 평균 = {w_row['학습_평균(saved)']:.4f} 건, 표준편차 = {w_row['학습_표준편차(saved)']:.4f} 건\n"
         f"• Z-score 계산: ({int(r0['센터출고대기건수_원본']):,} - {w_row['학습_평균(saved)']:.4f}) ÷ {w_row['학습_표준편차(saved)']:.4f} = {r0['센터출고대기건수_Zscore']:+.4f}\n"
         f"• 참고 Min-Max (1~6월 [{int(w_row['학습_최솟값']):,}, {int(w_row['학습_최댓값']):,}]건): ({int(r0['센터출고대기건수_원본'])} - {int(w_row['학습_최솟값'])}) ÷ ({int(w_row['학습_최댓값'])} - {int(w_row['학습_최솟값'])}) = {r0['센터출고대기건수_MinMax']:.4f}"),
        (0.2, 0.3, 9.6, 2.2, "#F8FAFC", "#64748B",
         "3) 핵심 요약 (PDF 내용 적용)",
         "• 원-핫 인코딩 28개 범주형 열은 이미 0/1이므로 표준화하지 않고 유지(passthrough)함.\n"
         "• 표준화 덕분에 단위가 다른 371km(+1.42)와 1,202건(+0.15)을 같은 척도로 공정하게 비교·학습함."),
    ]
    for bx, by, bw, bh, bg, edge, title_b, body_b in cards:
        rect = mpatches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.1,rounding_size=0.22", facecolor=bg, edgecolor=edge, linewidth=1.6)
        ax2.add_patch(rect)
        ax2.text(bx + 0.28, by + bh - 0.42, title_b, ha="left", va="center", fontsize=9.2, fontweight="bold", color="#0F172A")
        ax2.text(bx + 0.28, by + (bh - 0.65) * 0.48, body_b, ha="left", va="center", fontsize=8.3, color="#334155", linespacing=1.55)

    fig.suptitle(
        "[PDF] 로지스틱 기본 표준화 확인 — 실제 주문 5건 변환값 및 계산 과정",
        fontsize=14.4,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.11, wspace=0.14)
    fig.savefig(OUT_PNG_ORDERS, dpi=180)
    plt.close(fig)


def save_interactive_htmls(eval_df, stats_df, orders_df) -> None:
    """Save interactive Plotly HTMLs for PDF scaling comparison & 5 orders."""
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Coefficients & 배송거리_km Z-score vs Min-Max interactive HTML
    s_sorted = stats_df.sort_values("표준화후_계수절댓값", ascending=True)
    pfig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>① 표준화 후 Z-score 계수 vs 원단위 계수</b>", "<b>② 배송거리_km 분포 (눈금만 Z-score 변환)</b>"),
        horizontal_spacing=0.12,
    )
    pfig.add_trace(
        go.Bar(
            y=s_sorted["변수명"],
            x=s_sorted["표준화후_Zscore_계수"],
            orientation="h",
            name="Z-score 계수",
            marker_color=["#EF4444" if v > 0 else "#2563EB" for v in s_sorted["표준화후_Zscore_계수"]],
            text=[f"{v:+.4f}" for v in s_sorted["표준화후_Zscore_계수"]],
            textposition="outside",
            customdata=s_sorted["표준화전_원단위1당_계수"],
            hovertemplate="<b>%{y}</b><br>표준화 후 계수: %{x:+.4f}<br>표준화 전 원단위 1당 계수: %{customdata:+.6f}<extra></extra>",
        ),
        row=1, col=1,
    )
    d_row = stats_df.loc[stats_df["변수명"] == "배송거리_km"].iloc[0]
    ev_dist = eval_df["배송거리_km"].to_numpy(dtype=float)
    ev_dist_z = (ev_dist - float(d_row["학습_평균(saved)"])) / float(d_row["학습_표준편차(saved)"])
    pfig.add_trace(
        go.Histogram(
            x=ev_dist_z,
            nbinsx=18,
            name="배송거리_km (Z-score)",
            marker_color="#3B82F6",
            hovertemplate="Z-score 구간: %{x:.2f}<br>주문 수: %{y:,}건<extra></extra>",
        ),
        row=1, col=2,
    )
    pfig.add_vline(x=0, line_dash="dash", line_color="#0F172A", row=1, col=2)
    pfig.update_xaxes(title_text="표준화 회귀 계수", range=[-1.3, 0.95], row=1, col=1)
    pfig.update_xaxes(title_text="Z-score (0 = 학습 평균 202.50km)", row=1, col=2)
    pfig.update_yaxes(title_text="주문 건수", row=1, col=2)
    pfig.update_layout(
        title=dict(text="<b>[PDF] 숫자형 변수 스케일링 — 계수 비교 및 분포 형태 유지 확인</b>", x=0.5, xanchor="center"),
        template="plotly_white",
        height=540,
        showlegend=False,
        margin=dict(t=90, b=55, l=140, r=35),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    pfig.write_html(str(OUT_HTML_DIST), include_plotlyjs=True)

    # 2. 5 actual orders interactive HTML
    ofig = go.Figure()
    ofig.add_trace(
        go.Bar(
            name="배송거리_km (Z-score)",
            x=orders_df["주문ID"],
            y=orders_df["배송거리_Zscore"],
            marker_color="#2563EB",
            text=[f"{v:+.4f}" for v in orders_df["배송거리_Zscore"]],
            textposition="outside",
            customdata=np.stack([orders_df["배송거리_km_원본"], orders_df["배송거리_MinMax"], orders_df["물류센터"]], axis=1),
            hovertemplate="<b>%{x} (%{customdata[2]})</b><br>원본 배송거리: %{customdata[0]:.0f} km<br>Z-score: %{y:+.4f}<br>Min-Max: %{customdata[1]:.4f}<extra></extra>",
        )
    )
    ofig.add_trace(
        go.Bar(
            name="센터출고대기건수 (Z-score)",
            x=orders_df["주문ID"],
            y=orders_df["센터출고대기건수_Zscore"],
            marker_color="#0D9488",
            text=[f"{v:+.4f}" for v in orders_df["센터출고대기건수_Zscore"]],
            textposition="outside",
            customdata=np.stack([orders_df["센터출고대기건수_원본"], orders_df["센터출고대기건수_MinMax"], orders_df["물류센터"]], axis=1),
            hovertemplate="<b>%{x} (%{customdata[2]})</b><br>원본 출고대기: %{customdata[0]:,.0f} 건<br>Z-score: %{y:+.4f}<br>Min-Max: %{customdata[1]:.4f}<extra></extra>",
        )
    )
    ofig.add_hline(y=0, line_color="#0F172A", line_width=1.1)
    ofig.update_yaxes(title_text="Z-score 변환값", range=[-1.9, 1.9])
    ofig.update_layout(
        barmode="group",
        title=dict(text="<b>[PDF] 로지스틱 기본 표준화 — 실제 평가 주문 5건 변환 결과</b>", x=0.5, xanchor="center"),
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    ofig.write_html(str(OUT_HTML_ORDERS), include_plotlyjs=True)


def build_pdf_html_report(eval_df, stats_df, orders_df, oor_df) -> None:
    """Create standalone self-contained HTML report with 'PDF' in title and 11th-session styling."""
    plotly_js = plotly.offline.get_plotlyjs()

    d_row = stats_df.loc[stats_df["변수명"] == "배송거리_km"].iloc[0]
    w_row = stats_df.loc[stats_df["변수명"] == "센터출고대기건수"].iloc[0]
    r0 = orders_df.iloc[0]

    payload = {
        "stats": stats_df.to_dict(orient="records"),
        "orders": orders_df.to_dict(orient="records"),
        "oor": oor_df.to_dict(orient="records"),
        "dist_raw": eval_df["배송거리_km"].tolist(),
        "dist_z": ((eval_df["배송거리_km"] - float(d_row["학습_평균(saved)"])) / float(d_row["학습_표준편차(saved)"])).tolist(),
        "dist_mm": ((eval_df["배송거리_km"] - float(d_row["학습_최솟값"])) / (float(d_row["학습_최댓값"]) - float(d_row["학습_최솟값"]))).tolist(),
        "wait_raw": eval_df["센터출고대기건수"].tolist(),
        "wait_z": ((eval_df["센터출고대기건수"] - float(w_row["학습_평균(saved)"])) / float(w_row["학습_표준편차(saved)"])).tolist(),
        "wait_mm": ((eval_df["센터출고대기건수"] - float(w_row["학습_최솟값"])) / (float(w_row["학습_최댓값"]) - float(w_row["학습_최솟값"]))).tolist(),
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = f"""<!DOCTYPE html>
<html lang="ko" data-theme="a">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PDF — 숫자형 변수 스케일링(표준화·Min-Max) 및 로지스틱 적용 분석</title>
  <style>
    :root, [data-theme="a"] {{
      --font-sans: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      --bg-canvas: #F3F5F8;
      --bg-surface: #FFFFFF;
      --bg-subtle: #F8FAFC;
      --text-primary: #191F28;
      --text-secondary: #4E5968;
      --text-muted: #6B7684;
      --border-hairline: #E5E8EB;
      --border-strong: #CBD5E1;
      --accent-primary: #3182F6;
      --accent-weak: #E8F3FF;
      --accent-ink: #1B64DA;
      --accent-on: #FFFFFF;
      --danger: #E11D48;
      --danger-bg: #FFF1F2;
      --ok: #059669;
      --ok-bg: #ECFDF5;
      --radius-card: 20px;
      --radius-inner: 12px;
      --shadow-card: none;
    }}
    [data-theme="d"] {{
      --bg-canvas: #F4F5F6;
      --bg-surface: #FFFFFF;
      --bg-subtle: #EEF2F7;
      --text-primary: #131416;
      --text-secondary: #333B46;
      --text-muted: #4B5563;
      --border-hairline: #D1D6DB;
      --border-strong: #94A3B8;
      --accent-primary: #256EF4;
      --accent-weak: #ECF2FE;
      --accent-ink: #0B3D91;
      --accent-on: #FFFFFF;
      --radius-card: 12px;
      --radius-inner: 8px;
      --shadow-card: 0 1px 2px rgba(19,20,22,0.06);
    }}
    [data-theme="d"] header.card {{ border-top: 5px solid #0B3D91; }}
    [data-theme="e"] {{
      color-scheme: dark;
      --bg-canvas: #0E0F10;
      --bg-surface: #141517;
      --bg-subtle: #1D1F23;
      --text-primary: #FFFFFF;
      --text-secondary: #D4D8E0;
      --text-muted: #9CA3AF;
      --border-hairline: #2C3038;
      --border-strong: #474D59;
      --accent-primary: #00FFA3;
      --accent-weak: #0D3326;
      --accent-ink: #00FFA3;
      --accent-on: #0E0F10;
      --danger: #FF6B81;
      --danger-bg: #3A161F;
      --ok: #00FFA3;
      --ok-bg: #0D3326;
      --radius-card: 12px;
      --radius-inner: 8px;
      --shadow-card: 0 4px 12px rgba(0,0,0,0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg-canvas);
      color: var(--text-primary);
      font-family: var(--font-sans);
      font-variant-numeric: tabular-nums;
      line-height: 1.55;
    }}
    .shell {{ max-width: 1380px; margin: 0 auto; padding: 24px 24px 48px; display: flex; flex-direction: column; gap: 18px; }}
    .card {{ background: var(--bg-surface); border: 1px solid var(--border-hairline); border-radius: var(--radius-card); box-shadow: var(--shadow-card); padding: 20px 22px; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }}
    .eyebrow {{ display: inline-block; font-size: 12px; font-weight: 700; color: var(--accent-ink); background: var(--accent-weak); padding: 4px 10px; border-radius: 999px; margin-bottom: 8px; }}
    h1 {{ margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.02em; }}
    .meta-row {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 8px; }}
    .badge {{ font-size: 12px; font-weight: 500; color: var(--text-secondary); background: var(--bg-subtle); border: 1px solid var(--border-hairline); border-radius: 8px; padding: 3px 10px; }}
    .segmented {{ display: inline-flex; padding: 3px; background: var(--bg-subtle); border: 1px solid var(--border-hairline); border-radius: 999px; gap: 3px; }}
    .seg-btn {{ appearance: none; border: none; background: transparent; color: var(--text-secondary); font-family: inherit; font-size: 12.5px; font-weight: 600; padding: 6px 12px; border-radius: 999px; cursor: pointer; }}
    .seg-btn.active {{ background: var(--accent-primary); color: var(--accent-on); }}
    .grid-4 {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .grid-split {{ display: grid; grid-template-columns: 0.92fr 1.38fr; gap: 16px; }}
    .mini-card {{ background: var(--bg-subtle); border: 1px solid var(--border-hairline); border-radius: var(--radius-inner); padding: 14px 16px; }}
    .mini-card .k {{ font-size: 12px; font-weight: 700; color: var(--accent-ink); margin-bottom: 4px; }}
    .mini-card .t {{ font-size: 15px; font-weight: 700; margin-bottom: 6px; }}
    .mini-card .d {{ font-size: 12.8px; color: var(--text-secondary); }}
    .sec-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 10px; }}
    .sec-title {{ margin: 0; font-size: 16.5px; font-weight: 700; }}
    .sec-sub {{ font-size: 12.5px; color: var(--text-secondary); margin-top: 2px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
    th, td {{ padding: 8px 8px; border-bottom: 1px solid var(--border-hairline); text-align: left; white-space: nowrap; }}
    th {{ font-size: 11.8px; font-weight: 600; color: var(--text-muted); background: var(--bg-subtle); }}
    th.num, td.num {{ text-align: right; }}
    .callout {{ padding: 13px 16px; border-radius: var(--radius-inner); background: var(--accent-weak); color: var(--accent-ink); font-size: 13px; line-height: 1.6; }}
    .callout.warn {{ background: var(--danger-bg); color: var(--danger); }}
    .table-wrap {{ overflow-x: auto; }}
    .ctrl-bar {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    @media (max-width: 1020px) {{ .grid-4, .grid-2, .grid-split {{ grid-template-columns: 1fr; }} }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <div class="shell">
    <header class="card">
      <div class="topbar">
        <div>
          <span class="eyebrow">12회차 · PDF 개념 반영 실제 데이터 분석</span>
          <h1>PDF — 숫자형 변수 스케일링(표준화 · Min-Max) 및 로지스틱 적용 분석</h1>
          <div class="meta-row">
            <span class="badge">기준 모델: 로지스틱 기본 (delivery_logistic_11.joblib 재사용)</span>
            <span class="badge">기준 산출: 1~6월 학습 (7,200건)</span>
            <span class="badge">검증 적용: 7~8월 평가 (2,400건) · 재학습 없음</span>
          </div>
        </div>
        <div class="segmented" role="group" aria-label="테마 선택">
          <button type="button" class="seg-btn active" data-t="a" onclick="setTheme('a')">클린 라이트</button>
          <button type="button" class="seg-btn" data-t="d" onclick="setTheme('d')">공공 고대비</button>
          <button type="button" class="seg-btn" data-t="e" onclick="setTheme('e')">다크 관제</button>
        </div>
      </div>
    </header>

    <section class="grid-4">
      <div class="mini-card">
        <div class="k">PDF 01 · 계수 공정 비교</div>
        <div class="t">단위의 영향을 줄인다</div>
        <div class="d">원단위 계수는 배송거리 <b>+0.0042</b>(/km) vs 부하율 <b>+2.2042</b>로 단위에 끌리지만, Z-score 후에는 <b>+0.4986</b> vs <b>+0.5610</b>으로 공정하게 비교됩니다.</div>
      </div>
      <div class="mini-card">
        <div class="k">PDF 02 · Z-score (표준화)</div>
        <div class="t">(원래 값 − 평균) ÷ 표준편차</div>
        <div class="d">로지스틱 기본 모델이 사용한 방식입니다. 분포 모양은 그대로 유지되며 눈금만 평균 0 · 표준편차 1 단위로 바뀝니다.</div>
      </div>
      <div class="mini-card">
        <div class="k">PDF 03 · Min-Max 변환</div>
        <div class="t">0~1 범위 이탈 가능</div>
        <div class="d">학습 [0.22, 27.25]kg 기준 적용 시 7월 주문 <code>DLV008405</code>(0.21kg)가 <b>-0.0004</b>로 0 미만으로 벗어남을 실제 확인했습니다.</div>
      </div>
      <div class="mini-card">
        <div class="k">PDF 04 · 기준 재사용</div>
        <div class="t">1~6월 기준만 저장·재사용</div>
        <div class="d">저장된 <code>numeric_scaler</code>의 평균·표준편차를 7~8월과 새 주문에도 그대로 적용하고 원-핫 28열은 유지합니다.</div>
      </div>
    </section>

    <section class="card">
      <div class="sec-head">
        <div>
          <h2 class="sec-title">1. 실제 배송 데이터 분포 확인 — 눈금만 바뀌고 모양은 그대로 (Z-score vs Min-Max)</h2>
          <div class="sec-sub">변수와 스케일링 방식을 전환해도 히스토그램의 상대적 모양은 동일하며, x축 눈금만 바뀝니다.</div>
        </div>
        <div class="ctrl-bar">
          <div class="segmented">
            <button type="button" class="seg-btn active" id="btn-var-dist" onclick="setVar('dist')">배송거리_km</button>
            <button type="button" class="seg-btn" id="btn-var-wait" onclick="setVar('wait')">센터출고대기건수</button>
          </div>
          <div class="segmented">
            <button type="button" class="seg-btn active" id="btn-sc-z" onclick="setScale('z')">Z-score (실제 모델)</button>
            <button type="button" class="seg-btn" id="btn-sc-mm" onclick="setScale('mm')">Min-Max (0~1)</button>
          </div>
        </div>
      </div>
      <div class="grid-2">
        <div id="chart-raw" style="height:360px;"></div>
        <div id="chart-scaled" style="height:360px;"></div>
      </div>
    </section>

    <section class="card">
      <div class="sec-head">
        <div>
          <h2 class="sec-title">2. 실제 평가 주문 5건 — 원래 값 vs Z-score vs Min-Max 나란히 보기</h2>
          <div class="sec-sub">7~8월 평가 구간 첫 5개 주문에 대해 저장된 1~6월 기준값으로 변환한 결과입니다.</div>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>주문ID</th>
              <th>물류센터</th>
              <th class="num">배송거리 원본</th>
              <th class="num">배송거리 Z-score</th>
              <th class="num">배송거리 Min-Max</th>
              <th class="num">출고대기 원본</th>
              <th class="num">출고대기 Z-score</th>
              <th class="num">출고대기 Min-Max</th>
            </tr>
          </thead>
          <tbody id="orders-tbody"></tbody>
        </table>
      </div>
      <div class="callout" style="margin-top:12px;">
        <strong>예시 계산 ({r0['주문ID']} · {r0['물류센터']}):</strong><br/>
        • <b>배송거리_km</b> = <code>{int(r0['배송거리_km_원본'])} km</code> → 저장된 1~6월 평균(<code>{d_row['학습_평균(saved)']:.4f}</code>)과 표준편차(<code>{d_row['학습_표준편차(saved)']:.4f}</code>)로 계산:
        <code>({int(r0['배송거리_km_원본'])} − {d_row['학습_평균(saved)']:.4f}) ÷ {d_row['학습_표준편차(saved)']:.4f} = <b>{r0['배송거리_Zscore']:+.4f}</b></code><br/>
        • <b>센터출고대기건수</b> = <code>{int(r0['센터출고대기건수_원본']):,} 건</code> → 저장된 1~6월 평균(<code>{w_row['학습_평균(saved)']:.4f}</code>)과 표준편차(<code>{w_row['학습_표준편차(saved)']:.4f}</code>)로 계산:
        <code>({int(r0['센터출고대기건수_원본'])} − {w_row['학습_평균(saved)']:.4f}) ÷ {w_row['학습_표준편차(saved)']:.4f} = <b>{r0['센터출고대기건수_Zscore']:+.4f}</b></code>
      </div>
    </section>

    <section class="grid-split">
      <div class="card">
        <div class="sec-head">
          <div>
            <h2 class="sec-title">3. 표준화 전·후 로지스틱 계수 비교 (PDF p.1 실증)</h2>
            <div class="sec-sub">단위가 다른 8개 숫자형 변수를 1표준편차 기준으로 정렬</div>
          </div>
        </div>
        <div id="chart-coef" style="height:360px;"></div>
      </div>

      <div class="card">
        <div class="sec-head">
          <div>
            <h2 class="sec-title">4. 숫자형 8개 변수 학습 기준표 & Min-Max 이탈 확인</h2>
            <div class="sec-sub">모든 기준은 1~6월 7,200건에서만 산출해 재사용</div>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>변수명</th>
                <th class="num">학습 평균</th>
                <th class="num">표준편차</th>
                <th class="num">학습 범위</th>
                <th class="num">Z-score 계수</th>
                <th class="num">7~8월 이탈</th>
              </tr>
            </thead>
            <tbody id="stats-tbody"></tbody>
          </table>
        </div>
        <div class="callout warn" style="margin-top:12px;">
          <strong>Min-Max 0~1 범위 이탈 실측:</strong> <code>총중량_kg</code>의 1~6월 범위는 <code>[0.22, 27.25]</code>kg이나, 7월 <code>DLV008405</code>(0.21kg)는 Min-Max <b>-0.0004</b>, 9월 <code>DLV011016</code>(0.20kg)는 <b>-0.0007</b>로 0 아래로 벗어났습니다.
        </div>
      </div>
    </section>
  </div>

  <script>
    const DATA = {data_json};
    let curVar = 'dist';
    let curScale = 'z';
    let curTheme = 'a';

    function setTheme(t) {{
      curTheme = t;
      document.documentElement.setAttribute('data-theme', t);
      document.querySelectorAll('[data-t]').forEach(b => b.classList.toggle('active', b.dataset.t === t));
      renderDistCharts();
      renderCoefChart();
    }}
    function setVar(v) {{
      curVar = v;
      document.getElementById('btn-var-dist').classList.toggle('active', v === 'dist');
      document.getElementById('btn-var-wait').classList.toggle('active', v === 'wait');
      renderDistCharts();
    }}
    function setScale(s) {{
      curScale = s;
      document.getElementById('btn-sc-z').classList.toggle('active', s === 'z');
      document.getElementById('btn-sc-mm').classList.toggle('active', s === 'mm');
      renderDistCharts();
    }}

    function themeLayout() {{
      const dark = curTheme === 'e';
      return {{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {{ family: "Pretendard, Malgun Gothic, sans-serif", color: dark ? '#E5E7EB' : '#1F2937', size: 12 }},
        gridcolor: dark ? '#2C3038' : '#E5E8EB',
        accent: dark ? '#00FFA3' : (curTheme === 'd' ? '#256EF4' : '#3182F6'),
      }};
    }}

    function renderDistCharts() {{
      const tl = themeLayout();
      const raw = curVar === 'dist' ? DATA.dist_raw : DATA.wait_raw;
      const scaled = curVar === 'dist'
        ? (curScale === 'z' ? DATA.dist_z : DATA.dist_mm)
        : (curScale === 'z' ? DATA.wait_z : DATA.wait_mm);
      const varName = curVar === 'dist' ? '배송거리 (km)' : '센터출고대기건수 (건)';
      const scaleName = curScale === 'z' ? 'Z-score 변환값' : 'Min-Max (0~1) 변환값';

      Plotly.react('chart-raw', [{{
        x: raw, type: 'histogram', nbinsx: 18, marker: {{ color: '#64748B', line: {{ color: '#FFFFFF', width: 0.6 }} }},
        hovertemplate: varName + ': %{{x}}<br>주문 수: %{{y:,}}건<extra></extra>'
      }}], {{
        title: {{ text: '<b>변환 전: ' + varName + '</b>', font: {{ size: 13.5 }} }},
        paper_bgcolor: tl.paper_bgcolor, plot_bgcolor: tl.plot_bgcolor, font: tl.font,
        margin: {{ t: 44, b: 44, l: 48, r: 16 }},
        xaxis: {{ title: varName, gridcolor: tl.gridcolor }},
        yaxis: {{ title: '주문 건수', gridcolor: tl.gridcolor }},
      }}, {{ displayModeBar: false, responsive: true }});

      Plotly.react('chart-scaled', [{{
        x: scaled, type: 'histogram', nbinsx: 18, marker: {{ color: tl.accent, line: {{ color: '#FFFFFF', width: 0.6 }} }},
        hovertemplate: scaleName + ': %{{x:.3f}}<br>주문 수: %{{y:,}}건<extra></extra>'
      }}], {{
        title: {{ text: '<b>변환 후: ' + scaleName + ' (모양 동일)</b>', font: {{ size: 13.5 }} }},
        paper_bgcolor: tl.paper_bgcolor, plot_bgcolor: tl.plot_bgcolor, font: tl.font,
        margin: {{ t: 44, b: 44, l: 48, r: 16 }},
        xaxis: {{ title: scaleName, gridcolor: tl.gridcolor }},
        yaxis: {{ title: '주문 건수', gridcolor: tl.gridcolor }},
      }}, {{ displayModeBar: false, responsive: true }});
    }}

    function renderCoefChart() {{
      const tl = themeLayout();
      const s = [...DATA.stats].sort((a, b) => a['표준화후_계수절댓값'] - b['표준화후_계수절댓값']);
      Plotly.react('chart-coef', [{{
        y: s.map(r => r['변수명']),
        x: s.map(r => r['표준화후_Zscore_계수']),
        type: 'bar',
        orientation: 'h',
        marker: {{ color: s.map(r => r['표준화후_Zscore_계수'] > 0 ? '#EF4444' : tl.accent) }},
        text: s.map(r => (r['표준화후_Zscore_계수'] >= 0 ? '+' : '') + r['표준화후_Zscore_계수'].toFixed(4)),
        textposition: 'outside',
        customdata: s.map(r => r['표준화전_원단위1당_계수']),
        hovertemplate: '<b>%{{y}}</b><br>표준화 후 Z-score 계수: %{{x:+.4f}}<br>표준화 전 원단위 1당 계수: %{{customdata:+.6f}}<extra></extra>',
      }}], {{
        paper_bgcolor: tl.paper_bgcolor, plot_bgcolor: tl.plot_bgcolor, font: tl.font,
        margin: {{ t: 16, b: 42, l: 132, r: 58 }},
        xaxis: {{ title: '표준화 후 계수 (1표준편차 기준)', range: [-1.28, 1.02], gridcolor: tl.gridcolor, zeroline: true }},
        yaxis: {{ gridcolor: tl.gridcolor }},
      }}, {{ displayModeBar: false, responsive: true }});
    }}

    function renderTables() {{
      const ob = document.getElementById('orders-tbody');
      ob.innerHTML = DATA.orders.map(r => `
        <tr>
          <td><b>${{r['주문ID']}}</b></td>
          <td>${{r['물류센터']}}</td>
          <td class="num">${{Number(r['배송거리_km_원본']).toLocaleString()}} km</td>
          <td class="num"><b>${{(r['배송거리_Zscore'] >= 0 ? '+' : '') + r['배송거리_Zscore'].toFixed(4)}}</b></td>
          <td class="num">${{r['배송거리_MinMax'].toFixed(4)}}</td>
          <td class="num">${{Number(r['센터출고대기건수_원본']).toLocaleString()}} 건</td>
          <td class="num"><b>${{(r['센터출고대기건수_Zscore'] >= 0 ? '+' : '') + r['센터출고대기건수_Zscore'].toFixed(4)}}</b></td>
          <td class="num">${{r['센터출고대기건수_MinMax'].toFixed(4)}}</td>
        </tr>
      `).join('');

      const sb = document.getElementById('stats-tbody');
      sb.innerHTML = DATA.stats.map(r => `
        <tr>
          <td><b>${{r['변수명']}}</b></td>
          <td class="num">${{r['학습_평균(saved)'].toFixed(4)}}</td>
          <td class="num">${{r['학습_표준편차(saved)'].toFixed(4)}}</td>
          <td class="num">[${{r['학습_최솟값']}}, ${{r['학습_최댓값']}}]</td>
          <td class="num"><b>${{(r['표준화후_Zscore_계수'] >= 0 ? '+' : '') + r['표준화후_Zscore_계수'].toFixed(4)}}</b></td>
          <td class="num">${{r['평가_MinMax_범위이탈건수']}}건</td>
        </tr>
      `).join('');
    }}

    renderTables();
    renderDistCharts();
    renderCoefChart();
  </script>
</body>
</html>
"""
    OUT_REPORT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    train_df, eval_df, stats_df, orders_df, oor_df, scaler = compute_scaling_tables()
    save_static_png_overview(train_df, eval_df, stats_df, orders_df, oor_df, scaler)
    save_static_png_orders(orders_df, stats_df)
    save_interactive_htmls(eval_df, stats_df, orders_df)
    build_pdf_html_report(eval_df, stats_df, orders_df, oor_df)

    print("=== PDF 반영 스케일링 분석 완료 ===")
    print("Saved stats CSV:", OUT_STATS_CSV)
    print("Saved 5 orders CSV:", OUT_ORDERS_CSV)
    print("Saved HTML report:", OUT_REPORT_HTML)
    print("\n--- 실제 주문 5건 (Z-score & Min-Max) ---")
    for _, r in orders_df.iterrows():
        print(
            f"{r['주문ID']} ({r['물류센터']}): "
            f"거리={int(r['배송거리_km_원본'])}km -> Z={r['배송거리_Zscore']:+.4f}, MM={r['배송거리_MinMax']:.4f} | "
            f"대기={int(r['센터출고대기건수_원본'])}건 -> Z={r['센터출고대기건수_Zscore']:+.4f}, MM={r['센터출고대기건수_MinMax']:.4f}"
        )


if __name__ == "__main__":
    main()
