"""Generate October operational predictions using selected_model_12.joblib (leak-free) for Session 12."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
INTERACTIVE_DIR = GRAPHS_DIR / "interactive"

FEATURES_CSV = DATA_DIR / "delivery_features_09.csv"
SELECTED_JSON = RESULTS_DIR / "selected_model_12.json"
SELECTED_JOBLIB = RESULTS_DIR / "selected_model_12.joblib"

OUT_OPS_CSV = RESULTS_DIR / "delivery_operations_12.csv"
OUT_DAILY_CSV = RESULTS_DIR / "oct_operations_daily_summary_12.csv"
OUT_CENTER_CSV = RESULTS_DIR / "oct_operations_center_summary_12.csv"

OUT_PNG = GRAPHS_DIR / "14_10월_운영예측_일별_및_센터별_위험분포.png"
OUT_HTML = INTERACTIVE_DIR / "14_oct_operations_overview_12.html"

FORBIDDEN_COLS = {"실제배송완료일", "배송지연여부"}


def safe_div(num: float, den: float):
    """Divide safely, returning None when denominator is zero."""
    return (num / den) if den > 0 else None


def run_october_predictions():
    """Load selected model & metadata, predict October orders without outcome columns."""
    with open(SELECTED_JSON, encoding="utf-8") as f:
        sel_meta = json.load(f)

    pipe = joblib.load(SELECTED_JOBLIB)
    threshold = float(sel_meta["threshold"])
    input_cols = sel_meta["numeric_inputs"] + sel_meta["categorical_inputs"]

    raw_df = pd.read_csv(FEATURES_CSV)
    dt = pd.to_datetime(raw_df["점검기준일시"])
    oct_df = raw_df[(dt >= "2026-10-01") & (dt < "2026-11-01")].copy()

    # Strictly remove outcome columns before prediction & never evaluate October outcomes
    oct_clean = oct_df.drop(columns=[c for c in FORBIDDEN_COLS if c in oct_df.columns]).copy()
    assert not (FORBIDDEN_COLS & set(oct_clean.columns))

    probs = pipe.predict_proba(oct_clean[input_cols])[:, 1]
    oct_clean["지연예측확률"] = probs.astype(float)
    oct_clean["위험여부"] = (probs >= threshold).astype(int)

    oct_out = oct_clean.sort_values(["점검기준일시", "주문ID"], ascending=[True, True]).reset_index(drop=True)
    assert not (FORBIDDEN_COLS & set(oct_out.columns)), "Outcome columns must be excluded from operations CSV!"
    oct_out.to_csv(OUT_OPS_CSV, index=False, encoding="utf-8-sig")

    # Daily & center operational summaries (pre-shipment predictions only; no actual outcomes)
    oct_out["_날짜"] = pd.to_datetime(oct_out["점검기준일시"]).dt.strftime("%Y-%m-%d")
    daily_rows = []
    for d, grp in oct_out.groupby("_날짜"):
        tot = int(len(grp))
        risk = int(grp["위험여부"].sum())
        due0 = int(((grp["위험여부"] == 1) & (grp["배송여유일수"] <= 0)).sum())
        daily_rows.append({
            "점검일자": d,
            "전체주문": tot,
            "위험주문": risk,
            "예측위험비율": safe_div(risk, tot),
            "위험_당일마감건수": due0,
        })
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(OUT_DAILY_CSV, index=False, encoding="utf-8-sig")

    center_rows = []
    for ctr, grp in oct_out.groupby("물류센터"):
        tot = int(len(grp))
        risk = int(grp["위험여부"].sum())
        due0 = int(((grp["위험여부"] == 1) & (grp["배송여유일수"] <= 0)).sum())
        center_rows.append({
            "물류센터": ctr,
            "전체주문": tot,
            "위험주문": risk,
            "예측위험비율": safe_div(risk, tot),
            "위험_당일마감건수": due0,
        })
    center_df = pd.DataFrame(center_rows).sort_values("물류센터").reset_index(drop=True)
    center_df.to_csv(OUT_CENTER_CSV, index=False, encoding="utf-8-sig")

    oct_out = oct_out.drop(columns=["_날짜"])
    return sel_meta, oct_out, daily_df, center_df


def save_static_png(daily_df: pd.DataFrame, center_df: pd.DataFrame) -> None:
    """Create 14_10월_운영예측_일별_및_센터별_위험분포.png (no outcome metrics)."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.4, 6.6), dpi=180, gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Left: Daily order counts vs predicted risk orders (Oct 1~20)
    x_d = np.arange(len(daily_df))
    tot_d = daily_df["전체주문"].to_numpy()
    risk_d = daily_df["위험주문"].to_numpy()
    norm_d = tot_d - risk_d
    d_lbls = [d[5:] for d in daily_df["점검일자"]]

    ax1.bar(x_d, risk_d, width=0.66, label="위험 예측 주문 (위험여부=1 · 50% 이상)", color="#F59E0B", edgecolor="#92400E", linewidth=0.5)
    ax1.bar(x_d, norm_d, width=0.66, bottom=risk_d, label="정상 예측 주문 (위험여부=0 · 50% 미만)", color="#CBD5E1", edgecolor="#64748B", linewidth=0.5)

    for xi, r_v, t_v in zip(x_d, risk_d, tot_d):
        ax1.text(xi, t_v + 1.2, f"{t_v}", ha="center", va="bottom", fontsize=7.8, color="#334155", fontweight="bold")
        if r_v >= 8:
            ax1.text(xi, r_v / 2, f"{r_v}", ha="center", va="center", fontsize=7.8, color="#1F2937", fontweight="bold")

    # Highlight latest day (2026-10-20)
    ax1.annotate(
        f"최신일 10-20\n전체 {tot_d[-1]}건 · 위험 {risk_d[-1]}건",
        xy=(x_d[-1], tot_d[-1] + 2),
        xytext=(x_d[-1] - 3.6, max(tot_d) + 12),
        fontsize=8.4,
        fontweight="bold",
        color="#B45309",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFBEB", edgecolor="#F59E0B"),
        arrowprops=dict(arrowstyle="->", color="#D97706", lw=1.4),
    )

    ax1.set_xticks(x_d)
    ax1.set_xticklabels(d_lbls, rotation=45, ha="right", fontsize=8.5, fontweight="bold")
    ax1.set_ylim(0, max(tot_d) + 24)
    ax1.set_ylabel("주문 건수 (건)", fontsize=9.8)
    ax1.set_title("① 10월 점검일별 전체 주문 vs 위험 예측 주문 (20일간 945건)", fontsize=11.4, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=8.6, frameon=True)

    # Right: Center breakdown (total, risk, predicted risk rate)
    x_c = np.arange(len(center_df))
    w = 0.34
    tot_c = center_df["전체주문"].to_numpy()
    risk_c = center_df["위험주문"].to_numpy()
    rate_c = center_df["예측위험비율"].to_numpy() * 100

    b1 = ax2.bar(x_c - w / 2, tot_c, w, label="전체 주문 수", color="#94A3B8", edgecolor="#475569", linewidth=0.6)
    b2 = ax2.bar(x_c + w / 2, risk_c, w, label="위험 예측 주문 수", color="#F59E0B", edgecolor="#92400E", linewidth=0.7)

    for bar, v in zip(b1, tot_c):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 6, f"{int(v)}건", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#334155")
    for bar, v, rt in zip(b2, risk_c, rate_c):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 6, f"{int(v)}건\n({rt:.2f}%)", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#B45309")

    ax2.set_xticks(x_c)
    ax2.set_xticklabels(center_df["물류센터"], fontsize=10.5, fontweight="bold")
    ax2.set_ylim(0, max(tot_c) + 75)
    ax2.set_ylabel("주문 건수 (건)", fontsize=9.8)
    ax2.set_title("② 물류센터별 예측 위험 분포 (위험 ÷ 센터 전체 · 실제 지연률 아님)", fontsize=11.2, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", fontsize=8.8, frameon=True)

    fig.suptitle(
        "10월 운영 상황 재현 예측 결과 (가중치 적용 랜덤포레스트 balanced · 50% 기준 · 사후 결과 미사용)",
        fontsize=14.0,
        fontweight="bold",
        y=0.98,
    )
    plt.subplots_adjust(left=0.055, right=0.975, top=0.86, bottom=0.14, wspace=0.20)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)


def save_interactive_html(daily_df: pd.DataFrame, center_df: pd.DataFrame) -> None:
    """Create interactive Plotly HTML for October operational predictions."""
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    pfig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>① 10월 점검일별 예측 위험 물량</b>", "<b>② 물류센터별 예측 위험 분포</b>"),
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.10,
    )
    norm_d = daily_df["전체주문"] - daily_df["위험주문"]
    pfig.add_trace(
        go.Bar(
            name="위험 예측 (≥50%)",
            x=daily_df["점검일자"],
            y=daily_df["위험주문"],
            marker_color="#F59E0B",
            customdata=np.stack([daily_df["전체주문"], daily_df["예측위험비율"] * 100, daily_df["위험_당일마감건수"]], axis=1),
            hovertemplate="<b>%{x}</b><br>위험 예측: %{y:,}건 / 전체 %{customdata[0]:,.0f}건 (%{customdata[1]:.2f}%)<br>위험 중 당일마감: %{customdata[2]:,.0f}건<extra></extra>",
        ),
        row=1, col=1,
    )
    pfig.add_trace(
        go.Bar(
            name="정상 예측 (<50%)",
            x=daily_df["점검일자"],
            y=norm_d,
            marker_color="#CBD5E1",
            hovertemplate="<b>%{x}</b><br>정상 예측: %{y:,}건<extra></extra>",
        ),
        row=1, col=1,
    )
    pfig.add_trace(
        go.Bar(
            name="센터 전체 주문",
            x=center_df["물류센터"],
            y=center_df["전체주문"],
            marker_color="#94A3B8",
            text=[f"{int(v)}건" for v in center_df["전체주문"]],
            textposition="outside",
        ),
        row=1, col=2,
    )
    pfig.add_trace(
        go.Bar(
            name="센터 위험 예측",
            x=center_df["물류센터"],
            y=center_df["위험주문"],
            marker_color="#F59E0B",
            text=[f"{int(v)}건 ({r*100:.2f}%)" for v, r in zip(center_df["위험주문"], center_df["예측위험비율"])],
            textposition="outside",
        ),
        row=1, col=2,
    )
    pfig.update_layout(
        barmode="stack",
        title=dict(text="<b>10월 운영 상황 재현 예측 요약 (945건 · 50% 기준 · 사후 결과 미포함)</b>", x=0.5, xanchor="center"),
        template="plotly_white",
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(t=80, b=105, l=60, r=30),
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
    )
    # Note: make_subplots uses single barmode for layout; convert col 2 neatly by using grouped offset via barmode='group' on whole figure?
    # Better: keep col 1 stacked bars via base or switch layout to group? Let's use clean barmode='group' OR explicit base=0 on col 2!
    pfig.update_layout(barmode="group")
    # Re-stack col 1 traces explicitly using base on trace index 1
    pfig.data[1].base = daily_df["위험주문"]
    pfig.data[0].offsetgroup = "d"
    pfig.data[1].offsetgroup = "d"
    pfig.data[2].offsetgroup = "c1"
    pfig.data[3].offsetgroup = "c2"
    pfig.update_yaxes(title_text="주문 건수 (건)", row=1, col=1)
    pfig.update_yaxes(title_text="주문 건수 (건)", range=[0, max(center_df["전체주문"]) + 70], row=1, col=2)
    pfig.write_html(str(OUT_HTML), include_plotlyjs=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    sel_meta, oct_out, daily_df, center_df = run_october_predictions()
    save_static_png(daily_df, center_df)
    save_interactive_html(daily_df, center_df)

    total = int(len(oct_out))
    risk = int(oct_out["위험여부"].sum())
    due0 = int(((oct_out["위험여부"] == 1) & (oct_out["배송여유일수"] <= 0)).sum())
    latest_dt = str(oct_out["점검기준일시"].max())
    lat_df = oct_out[oct_out["점검기준일시"] == latest_dt]
    lat_risk = lat_df[lat_df["위험여부"] == 1].sort_values(["지연예측확률", "배송여유일수", "주문ID"], ascending=[False, True, True])

    print(f"=== 10월 운영용 예측 완료 ({sel_meta['model_name']} · 50% 기준) ===")
    print(f"Saved CSV: {OUT_OPS_CSV} (shape={oct_out.shape}, forbidden_cols_present={bool(FORBIDDEN_COLS & set(oct_out.columns))})")
    print(f"10월 전체 ({daily_df['점검일자'].min()} ~ {daily_df['점검일자'].max()}, {len(daily_df)}일): {total:,}건 | 위험 예측={risk:,}건 ({risk/total*100:.2f}%) | 위험 중 당일마감={due0:,}건")
    print(f"최신 점검일 ({latest_dt}): 전체={len(lat_df)}건 | 위험 예측={len(lat_risk)}건 ({len(lat_risk)/len(lat_df)*100:.2f}%) | 당일마감={int((lat_risk['배송여유일수']<=0).sum())}건")
    print("\n--- 최신 점검일 (2026-10-20) 위험 주문 목록 ---")
    print(lat_risk[["주문ID", "배송약속일", "물류센터", "운송사", "상품군", "재고확보상태", "배송여유일수", "지연예측확률", "위험여부"]].to_string(index=False))


if __name__ == "__main__":
    main()
