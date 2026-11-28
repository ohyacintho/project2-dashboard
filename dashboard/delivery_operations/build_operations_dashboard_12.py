"""Build self-contained October Operations Dashboard (results/delivery_operations.html) for Session 12."""

import json
from pathlib import Path
import pandas as pd
import plotly


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

OPS_CSV = RESULTS_DIR / "delivery_operations_12.csv"
SELECTED_JSON = RESULTS_DIR / "selected_model_12.json"
MODEL_COMP_CSV = RESULTS_DIR / "model_comparison_12.csv"
SEP_EVAL_CSV = RESULTS_DIR / "sep_evaluation_summary_12.csv"
OUT_HTML = RESULTS_DIR / "delivery_operations.html"

FORBIDDEN_COLS = {"실제배송완료일", "배송지연여부"}


def assign_priority(slack: int, inv: str) -> tuple[int, str, str]:
    """Assign operational priority rank (1~4), reason tag, and action prompt."""
    due0 = slack <= 0
    unsecured = inv in ("미확보", "부분확보")
    if due0 and unsecured:
        return 1, "마감 당일 · 재고 미확보/부분", "즉시 확보·배송 가능 여부 확인"
    if due0 and not unsecured:
        return 2, "마감 당일 · 재고 확보완료", "출고 준비·운송사 인계 확인"
    if (not due0) and unsecured:
        return 3, "여유 있음 · 재고 미확보/부분", "기한 내 확보·출고 가능 여부 확인"
    return 4, "여유 있음 · 재고 확보완료", "배송 일정 확인"


def build_payload():
    """Load leak-free October predictions & selected model metadata into embedded JSON."""
    with open(SELECTED_JSON, encoding="utf-8") as f:
        sel = json.load(f)

    ops_df = pd.read_csv(OPS_CSV)
    assert not (FORBIDDEN_COLS & set(ops_df.columns)), "Forbidden outcome columns found in delivery_operations_12.csv!"

    ops_df["_date"] = pd.to_datetime(ops_df["점검기준일시"]).dt.strftime("%Y-%m-%d")

    records = []
    for _, r in ops_df.iterrows():
        slack = int(r["배송여유일수"])
        inv = str(r["재고확보상태"])
        risk = int(r["위험여부"])
        if risk == 1:
            prio, prio_why, prio_action = assign_priority(slack, inv)
        else:
            prio, prio_why, prio_action = 9, "50% 미만 정상 예측", "-"

        records.append({
            "id": str(r["주문ID"]),
            "dt": str(r["점검기준일시"]),
            "d": str(r["_date"]),
            "due": str(r["배송약속일"]),
            "ctr": str(r["물류센터"]),
            "reg": str(r["배송지역"]),
            "car": str(r["운송사"]),
            "cat": str(r["상품군"]),
            "typ": str(r["배송유형"]),
            "inv": inv,
            "slack": slack,
            "prob": round(float(r["지연예측확률"]), 6),
            "risk": risk,
            "prio": prio,
            "why": prio_why,
            "act": prio_action,
        })

    min_date = str(ops_df["_date"].min())
    max_date = str(ops_df["_date"].max())
    recent3_start = (pd.to_datetime(max_date) - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    recent7_start = (pd.to_datetime(max_date) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")

    comp_df = pd.read_csv(MODEL_COMP_CSV)
    model_comparison = []
    for _, r in comp_df.iterrows():
        model_comparison.append({
            "model": str(r["모델"]),
            "period": str(r["평가기간"]),
            "total": int(r["전체주문"]),
            "actual_pos": int(r["실제지연"]),
            "flagged": int(r["점검대상"]),
            "flag_rate": float(r["점검비율"]),
            "tp": int(r["찾아낸_실제지연(TP)"]),
            "fp": int(r["점검대상_실제정상(FP)"]),
            "fn": int(r["놓친_실제지연(FN)"]),
            "tn": int(r["점검제외_실제정상(TN)"]),
            "acc": float(r["정확도"]),
            "prec": float(r["정밀도"]),
            "rec": float(r["재현율"]),
            "f1": float(r["F1"]),
            "auc": float(r["ROC_AUC"]),
            "is_selected": str(r["모델"]) == sel["model_name"],
        })

    sep_df = pd.read_csv(SEP_EVAL_CSV)
    sep_eval = []
    for _, r in sep_df.iterrows():
        sep_eval.append({
            "split": str(r["기간구분"]),
            "period": str(r["기간"]),
            "total": int(r["전체주문"]),
            "actual_pos": int(r["실제지연"]),
            "delay_rate": float(r["실제지연율"]),
            "flagged": int(r["점검대상"]),
            "flag_rate": float(r["점검비율"]),
            "tp": int(r["찾아낸_실제지연(TP)"]),
            "fp": int(r["점검대상_실제정상(FP)"]),
            "fn": int(r["놓친_실제지연(FN)"]),
            "tn": int(r["점검제외_실제정상(TN)"]),
            "acc": float(r["정확도"]),
            "prec": float(r["정밀도"]),
            "rec": float(r["재현율"]),
            "f1": float(r["F1"]),
            "auc": float(r["ROC_AUC"]),
        })

    ap = sel.get("actual_parameters", {})
    payload = {
        "meta": {
            "model_name": sel["model_name"],
            "estimator": sel.get("estimator", "RandomForestClassifier"),
            "threshold": float(sel["threshold"]),
            "train_period": f"{sel['train_period'][0]} ~ {sel['train_period'][1]}",
            "comparison_period": f"{sel['comparison_period'][0]} ~ {sel['comparison_period'][1]}",
            "final_eval_period": f"{sel['final_evaluation_period'][0]} ~ {sel['final_evaluation_period'][1]}",
            "ops_period": f"{min_date} ~ {max_date}",
            "ops_total": len(records),
            "min_date": min_date,
            "max_date": max_date,
            "recent3_start": recent3_start,
            "recent7_start": recent7_start,
            "centers": sorted(ops_df["물류센터"].unique().tolist()),
            "carriers": sorted(ops_df["운송사"].unique().tolist()),
            "numeric_inputs": sel.get("numeric_inputs", []),
            "categorical_inputs": sel.get("categorical_inputs", []),
            "model_params_summary": (
                f"n_estimators={ap.get('n_estimators', 200)}, "
                f"max_depth={ap.get('max_depth', 6)}, "
                f"min_samples_leaf={ap.get('min_samples_leaf', 50)}, "
                f"class_weight={ap.get('class_weight', 'balanced')}"
            ),
            "selection_reason": sel["selection_reason"],
            "tradeoff": sel["tradeoff"],
        },
        "orders": records,
        "model_comparison": model_comparison,
        "sep_eval": sep_eval,
    }
    return payload


def generate_html(payload: dict) -> str:
    """Render self-contained HTML with Plotly JS, 8-theme switcher, priority matrix, and center click-sync."""
    plotly_js = plotly.offline.get_plotlyjs()
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    m = payload["meta"]

    # Guard against forbidden strings leaking into serialized data payload
    assert "실제배송완료일" not in data_json and "배송지연여부" not in data_json

    return f"""<!DOCTYPE html>
<html lang="ko" data-theme="a">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>배송 운영 현황</title>
  <style>
    :root, [data-theme="a"] {{
      --font-sans: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
      --bg-canvas: oklch(0.966 0.003 248);
      --bg-surface: oklch(1.000 0.000 0);
      --bg-subtle: oklch(0.985 0.002 248);
      --text-primary: oklch(0.237 0.020 258);
      --text-secondary: oklch(0.460 0.028 256);
      --text-muted: oklch(0.562 0.025 254);
      --border-hairline: oklch(0.930 0.005 248);
      --border-strong: oklch(0.874 0.009 248);
      --accent-primary: oklch(0.620 0.191 258);
      --accent-weak: oklch(0.959 0.020 250);
      --accent-ink: oklch(0.480 0.175 258);
      --accent-on: #ffffff;
      --status-success: oklch(0.493 0.143 154);
      --status-success-bg: oklch(0.955 0.028 154);
      --status-warning: oklch(0.640 0.180 56);
      --status-warning-bg: oklch(0.962 0.036 60);
      --status-danger: oklch(0.580 0.210 22);
      --status-danger-bg: oklch(0.965 0.028 22);
      --radius-card: 22px;
      --radius-inner: 14px;
      --radius-pill: 999px;
      --radius-tag: 8px;
      --shadow-card: none;
      --shadow-control: 0 1px 2px oklch(0.155 0.060 261 / 0.04);
      --weight-heading: 700;
      --weight-emphasis: 700;
      --tracking-heading: -0.020em;
      --sel-row-bg: oklch(0.959 0.020 250);
      --chart-total: #CBD5E1;
      --chart-risk: #3182F6;
      --chart-line: #DC2626;
      --transition-ui: 160ms cubic-bezier(0.22, 0.61, 0.36, 1);
    }}
    [data-theme="b"] {{
      --font-sans: Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
      --bg-canvas: oklch(1.000 0.000 0);
      --bg-surface: oklch(1.000 0.000 0);
      --bg-subtle: oklch(0.000 0.000 0 / 0.02);
      --text-primary: oklch(0.000 0.000 0 / 0.85);
      --text-secondary: oklch(0.000 0.000 0 / 0.60);
      --text-muted: oklch(0.000 0.000 0 / 0.45);
      --border-hairline: oklch(0.000 0.000 0 / 0.11);
      --border-strong: oklch(0.000 0.000 0 / 0.18);
      --accent-primary: oklch(0.651 0.191 252.8);
      --accent-weak: oklch(0.919 0.029 244.9);
      --accent-ink: oklch(0.449 0.074 248.6);
      --status-success: oklch(0.510 0.160 145);
      --status-success-bg: oklch(0.000 0.000 0 / 0.04);
      --status-warning: oklch(0.630 0.160 68);
      --status-warning-bg: oklch(0.000 0.000 0 / 0.05);
      --status-danger: oklch(0.580 0.210 24.5);
      --status-danger-bg: oklch(0.960 0.024 24);
      --radius-card: 8px;
      --radius-inner: 6px;
      --radius-pill: 6px;
      --radius-tag: 4px;
      --shadow-card: none;
      --shadow-control: none;
      --weight-heading: 600;
      --weight-emphasis: 600;
      --tracking-heading: 0em;
      --sel-row-bg: oklch(0.956 0.019 248.0);
      --chart-total: #D9D9D9;
      --chart-risk: #1890FF;
      --chart-line: #DC2626;
    }}
    [data-theme="c"] {{
      --font-sans: "Inter", "Noto Sans KR", -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
      --bg-canvas: oklch(0.976 0.001 286);
      --bg-surface: oklch(1.000 0 0);
      --bg-subtle: oklch(0.988 0 286);
      --text-primary: oklch(0 0 0 / 0.85);
      --text-secondary: oklch(0 0 0 / 0.60);
      --text-muted: oklch(0 0 0 / 0.40);
      --border-hairline: oklch(0 0 0 / 0.08);
      --border-strong: oklch(0 0 0 / 0.15);
      --accent-primary: oklch(0.554 0.214 280);
      --accent-weak: oklch(0.554 0.214 280 / 0.10);
      --accent-ink: oklch(0.554 0.214 280);
      --status-success: oklch(0.520 0.150 150);
      --status-success-bg: oklch(0.652 0.168 150 / 0.10);
      --status-warning: oklch(0.640 0.156 55);
      --status-warning-bg: oklch(0.698 0.156 55 / 0.10);
      --status-danger: oklch(0.580 0.177 20);
      --status-danger-bg: oklch(0.634 0.177 20 / 0.10);
      --radius-card: 16px;
      --radius-inner: 12px;
      --radius-pill: 8px;
      --radius-tag: 6px;
      --shadow-card: 0 1px 2px oklch(0 0 0 / 0.05), 0 4px 12px oklch(0 0 0 / 0.08);
      --shadow-control: 0 1px 2px oklch(0 0 0 / 0.08);
      --weight-heading: 700;
      --weight-emphasis: 700;
      --tracking-heading: -0.025em;
      --sel-row-bg: oklch(0.554 0.214 280 / 0.06);
      --chart-total: #D8D6EE;
      --chart-risk: #5E5CE6;
      --chart-line: #DF4857;
    }}
    [data-theme="d"] {{
      --font-sans: "Pretendard GOV", Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif;
      --bg-canvas: oklch(0.965 0.002 247);
      --bg-surface: oklch(1.000 0.000 0);
      --bg-subtle: oklch(0.960 0.008 254);
      --text-primary: oklch(0.190 0.004 264);
      --text-secondary: oklch(0.410 0.014 250);
      --text-muted: oklch(0.488 0.018 247);
      --border-hairline: oklch(0.860 0.007 248);
      --border-strong: oklch(0.662 0.019 246);
      --accent-primary: oklch(0.575 0.214 261);
      --accent-weak: oklch(0.955 0.018 254);
      --accent-ink: oklch(0.345 0.115 257);
      --accent-on: #ffffff;
      --status-success: oklch(0.510 0.120 154);
      --status-success-bg: oklch(0.955 0.025 154);
      --status-warning: oklch(0.520 0.140 58);
      --status-warning-bg: oklch(0.962 0.032 62);
      --status-danger: oklch(0.543 0.192 26);
      --status-danger-bg: oklch(0.945 0.022 17);
      --radius-card: 12px;
      --radius-inner: 8px;
      --radius-pill: 999px;
      --radius-tag: 4px;
      --shadow-card: 0 1px 2px oklch(0.19 0.004 264 / 0.06), 0 1px 1px oklch(0.19 0.004 264 / 0.04);
      --shadow-control: 0 1px 2px oklch(0.19 0.004 264 / 0.06);
      --weight-heading: 700;
      --weight-emphasis: 700;
      --tracking-heading: -0.010em;
      --sel-row-bg: oklch(0.955 0.018 254);
      --chart-total: #CBD5E1;
      --chart-risk: #256EF4;
      --chart-line: #DC2626;
    }}
    [data-theme="d"] header.card {{ border-top: 5px solid oklch(0.292 0.090 255); }}
    [data-theme="e"] {{
      color-scheme: dark;
      --font-sans: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, "Malgun Gothic", sans-serif;
      --bg-canvas: oklch(0.168 0.003 248);
      --bg-surface: oklch(0.210 0.004 264.5);
      --bg-subtle: oklch(0.251 0.005 248);
      --text-primary: oklch(1.000 0.000 0);
      --text-secondary: oklch(0.913 0.011 269.5);
      --text-muted: oklch(0.740 0.024 265.5);
      --border-hairline: oklch(0.320 0.008 258.4);
      --border-strong: oklch(0.450 0.008 258.4);
      --accent-primary: oklch(0.881 0.205 158.3);
      --accent-weak: oklch(0.284 0.061 163.2);
      --accent-ink: oklch(0.881 0.205 158.3);
      --accent-on: oklch(0.168 0.003 248);
      --status-success: oklch(0.815 0.189 158.4);
      --status-success-bg: oklch(0.309 0.049 169.6);
      --status-warning: oklch(0.830 0.165 78);
      --status-warning-bg: oklch(0.320 0.070 68);
      --status-danger: oklch(0.720 0.200 24);
      --status-danger-bg: oklch(0.320 0.105 24);
      --radius-card: 12px;
      --radius-inner: 8px;
      --radius-pill: 999px;
      --radius-tag: 4px;
      --shadow-card: 0 4px 12px 1px oklch(0 0 0 / 0.35), 0 1px 3px 0 oklch(0 0 0 / 0.45);
      --shadow-control: 0 2px 6px oklch(0 0 0 / 0.35);
      --weight-heading: 700;
      --weight-emphasis: 700;
      --tracking-heading: -0.015em;
      --sel-row-bg: oklch(0.284 0.061 163.2 / 0.55);
      --chart-total: #3F4756;
      --chart-risk: #00FFA3;
      --chart-line: #FF6B81;
    }}
    [data-theme="e"] header.card {{ border-top: 3px solid oklch(0.881 0.205 158.3); }}
    [data-theme="f"] {{
      --font-sans: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
      --bg-canvas: oklch(0.967 0.002 248);
      --bg-surface: oklch(1.000 0.000 0);
      --bg-subtle: oklch(0.979 0.002 248);
      --text-primary: oklch(0.226 0.008 264);
      --text-secondary: oklch(0.307 0.017 256);
      --text-muted: oklch(0.477 0.028 264);
      --border-hairline: oklch(0.901 0.007 269);
      --border-strong: oklch(0.867 0.007 269);
      --accent-primary: oklch(0.618 0.195 40);
      --accent-weak: oklch(0.970 0.016 47);
      --accent-ink: oklch(0.529 0.173 38);
      --accent-on: #ffffff;
      --status-success: oklch(0.499 0.095 174);
      --status-success-bg: oklch(0.974 0.015 176);
      --status-warning: oklch(0.592 0.109 85);
      --status-warning-bg: oklch(0.970 0.023 72);
      --status-danger: oklch(0.537 0.206 29);
      --status-danger-bg: oklch(0.965 0.014 17);
      --radius-card: 16px;
      --radius-inner: 12px;
      --radius-pill: 9999px;
      --radius-tag: 6px;
      --shadow-card: 0 1px 4px 0 oklch(0 0 0 / 0.078);
      --shadow-control: 0 1px 3px 0 oklch(0 0 0 / 0.08);
      --weight-heading: 700;
      --weight-emphasis: 700;
      --tracking-heading: -0.015em;
      --sel-row-bg: oklch(0.970 0.016 47);
      --chart-total: #E2E8F0;
      --chart-risk: #E14D00;
      --chart-line: #B91C1C;
    }}
    [data-theme="g"] {{
      --font-sans: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Noto Sans KR", "Malgun Gothic", sans-serif;
      --bg-canvas: oklch(0.964 0.004 271.4);
      --bg-surface: oklch(1.000 0.000 0);
      --bg-subtle: oklch(0.969 0.014 174.2);
      --text-primary: oklch(0.222 0.008 264.4);
      --text-secondary: oklch(0.284 0.012 264.4);
      --text-muted: oklch(0.515 0.021 258.4);
      --border-hairline: oklch(0.930 0.007 260.7);
      --border-strong: oklch(0.875 0.011 261.8);
      --accent-primary: oklch(0.500 0.112 159.8);
      --accent-weak: oklch(0.969 0.014 174.2);
      --accent-ink: oklch(0.342 0.075 160.6);
      --accent-on: #ffffff;
      --status-success: oklch(0.500 0.112 159.8);
      --status-success-bg: oklch(0.969 0.014 174.2);
      --status-warning: oklch(0.521 0.104 74.4);
      --status-warning-bg: oklch(0.967 0.016 79.4);
      --status-danger: oklch(0.550 0.185 20.6);
      --status-danger-bg: oklch(0.972 0.011 10.3);
      --radius-card: 20px;
      --radius-inner: 12px;
      --radius-pill: 16px;
      --radius-tag: 6px;
      --shadow-card: none;
      --shadow-control: none;
      --weight-heading: 700;
      --weight-emphasis: 700;
      --tracking-heading: -0.010em;
      --sel-row-bg: oklch(0.969 0.014 174.2);
      --chart-total: #DCE2EA;
      --chart-risk: #06A96C;
      --chart-line: #DC2626;
    }}
    [data-theme="h"] {{
      --font-sans: "Malgun Gothic", "Noto Sans KR", "Apple SD Gothic Neo", sans-serif;
      --bg-canvas: oklch(0.961 0.000 0);
      --bg-surface: oklch(1.000 0.000 0);
      --bg-subtle: oklch(0.976 0.026 100);
      --text-primary: oklch(0.252 0.000 0);
      --text-secondary: oklch(0.402 0.016 71);
      --text-muted: oklch(0.468 0.000 0);
      --border-hairline: oklch(0.898 0.000 0);
      --border-strong: oklch(0.764 0.000 0);
      --accent-primary: oklch(0.874 0.163 91);
      --accent-weak: oklch(0.976 0.026 100);
      --accent-ink: oklch(0.303 0.028 55);
      --accent-on: oklch(0.303 0.028 55);
      --status-success: oklch(0.440 0.110 155);
      --status-success-bg: oklch(0.955 0.025 155);
      --status-warning: oklch(0.550 0.160 51);
      --status-warning-bg: oklch(0.976 0.032 90);
      --status-danger: oklch(0.540 0.195 25);
      --status-danger-bg: oklch(0.965 0.024 25);
      --radius-card: 0px;
      --radius-inner: 0px;
      --radius-pill: 2px;
      --radius-tag: 2px;
      --shadow-card: none;
      --shadow-control: none;
      --weight-heading: 700;
      --weight-emphasis: 700;
      --tracking-heading: 0em;
      --sel-row-bg: oklch(0.948 0.008 74);
      --chart-total: #D6D2CB;
      --chart-risk: #D9A000;
      --chart-line: #B91C1C;
    }}
    [data-theme="h"] header.card {{ border-top: 6px solid oklch(0.874 0.163 91); }}
    [data-theme="h"] th {{ border-top: 2px solid oklch(0.464 0.021 78); }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg-canvas);
      color: var(--text-primary);
      font-family: var(--font-sans);
      font-variant-numeric: tabular-nums;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}
    .shell {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 22px 24px 52px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-card);
      box-shadow: var(--shadow-card);
      padding: 18px 22px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      font-weight: var(--weight-emphasis);
      color: var(--accent-ink);
      background: var(--accent-weak);
      padding: 4px 11px;
      border-radius: var(--radius-pill);
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: 25px;
      font-weight: var(--weight-heading);
      letter-spacing: var(--tracking-heading);
      line-height: 1.25;
    }}
    .meta-row {{
      margin-top: 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
      font-size: 12.2px;
    }}
    .meta-badge {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 9px;
      border-radius: var(--radius-tag);
      background: var(--bg-subtle);
      border: 1px solid var(--border-hairline);
      color: var(--text-secondary);
      font-weight: 500;
    }}
    .meta-badge.safe {{
      background: var(--status-success-bg);
      color: var(--status-success);
      border-color: transparent;
      font-weight: 600;
    }}
    .topbar-controls {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 6px;
      margin-left: auto;
    }}
    .segmented {{
      display: inline-flex;
      padding: 3px;
      background: var(--bg-subtle);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-pill);
      gap: 3px;
      flex-wrap: wrap;
    }}
    .seg-btn {{
      appearance: none;
      border: none;
      background: transparent;
      color: var(--text-secondary);
      font-family: inherit;
      font-size: 11.8px;
      font-weight: 600;
      padding: 5px 10px;
      border-radius: var(--radius-pill);
      cursor: pointer;
      transition: all var(--transition-ui);
      white-space: nowrap;
    }}
    .seg-btn:hover {{ color: var(--text-primary); }}
    .seg-btn.active {{
      background: var(--accent-primary);
      color: var(--accent-on);
    }}
    .notice-banner {{
      padding: 11px 16px;
      border-radius: var(--radius-inner);
      background: var(--accent-weak);
      color: var(--accent-ink);
      font-size: 13px;
      line-height: 1.55;
    }}
    .filter-grid {{
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      gap: 12px;
    }}
    .field-group {{
      display: flex;
      flex-direction: column;
      gap: 5px;
    }}
    .field-label {{
      font-size: 11.5px;
      font-weight: 600;
      color: var(--text-muted);
    }}
    input[type="date"], input[type="text"], select {{
      font-family: inherit;
      font-size: 13px;
      font-weight: 500;
      color: var(--text-primary);
      background: var(--bg-surface);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-inner);
      padding: 6px 10px;
      height: 36px;
      outline: none;
    }}
    input:focus, select:focus {{ border-color: var(--accent-primary); }}
    .btn {{
      appearance: none;
      font-family: inherit;
      font-size: 12.8px;
      font-weight: 600;
      height: 36px;
      padding: 0 13px;
      border-radius: var(--radius-inner);
      border: 1px solid var(--border-strong);
      background: var(--bg-surface);
      color: var(--text-primary);
      cursor: pointer;
      transition: all var(--transition-ui);
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }}
    .btn:hover {{ background: var(--bg-subtle); }}
    .btn.active-quick {{
      background: var(--accent-weak);
      color: var(--accent-ink);
      border-color: var(--accent-primary);
    }}
    .btn.primary {{
      background: var(--accent-primary);
      border-color: var(--accent-primary);
      color: var(--accent-on);
    }}
    .btn:disabled {{ opacity: 0.45; cursor: not-allowed; }}

    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .kpi-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-card);
      box-shadow: var(--shadow-card);
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .kpi-card.emphasis {{
      background: var(--sel-row-bg);
      border-color: var(--accent-primary);
    }}
    .kpi-title {{
      font-size: 13px;
      font-weight: 600;
      color: var(--text-secondary);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .kpi-value {{
      font-size: 29px;
      font-weight: var(--weight-heading);
      letter-spacing: var(--tracking-heading);
      line-height: 1.2;
      margin-top: 2px;
    }}
    .kpi-sub {{
      font-size: 12.3px;
      color: var(--text-secondary);
    }}

    .sec-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .sec-title {{
      margin: 0;
      font-size: 16px;
      font-weight: var(--weight-heading);
      letter-spacing: var(--tracking-heading);
    }}
    .sec-sub {{
      font-size: 12.4px;
      color: var(--text-secondary);
      margin-top: 2px;
    }}

    .prio-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .prio-card {{
      text-align: left;
      appearance: none;
      width: 100%;
      font-family: inherit;
      background: var(--bg-subtle);
      border: 1.5px solid var(--border-hairline);
      border-radius: var(--radius-inner);
      padding: 13px 15px;
      cursor: pointer;
      transition: all var(--transition-ui);
      display: flex;
      flex-direction: column;
      gap: 5px;
      color: var(--text-primary);
    }}
    .prio-card:hover {{ border-color: var(--border-strong); transform: translateY(-1px); }}
    .prio-card.active {{
      background: var(--sel-row-bg);
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 1px var(--accent-primary);
    }}
    .prio-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      width: 100%;
    }}
    .prio-badge {{
      font-size: 11.5px;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: var(--radius-tag);
    }}
    .prio-1 {{ background: var(--status-danger-bg); color: var(--status-danger); }}
    .prio-2 {{ background: var(--status-warning-bg); color: var(--status-warning); }}
    .prio-3 {{ background: var(--accent-weak); color: var(--accent-ink); }}
    .prio-4 {{ background: var(--status-success-bg); color: var(--status-success); }}
    .prio-cnt {{
      font-size: 22px;
      font-weight: var(--weight-heading);
    }}
    .prio-cond {{ font-size: 12.2px; font-weight: 600; color: var(--text-secondary); }}
    .prio-act {{ font-size: 12.2px; color: var(--text-primary); font-weight: 500; }}

    .split-grid {{
      display: grid;
      grid-template-columns: 1.32fr 1fr;
      gap: 16px;
      align-items: stretch;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12.8px;
    }}
    th, td {{
      padding: 9px 9px;
      border-bottom: 1px solid var(--border-hairline);
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }}
    th {{
      font-size: 11.8px;
      font-weight: 600;
      color: var(--text-muted);
      background: var(--bg-subtle);
    }}
    th.num, td.num {{ text-align: right; }}
    th.center, td.center {{ text-align: center; }}
    tbody tr.clickable-row {{ cursor: pointer; }}
    tbody tr:hover {{ background: var(--bg-subtle); }}
    tbody tr.selected-center {{ background: var(--sel-row-bg); font-weight: 600; }}

    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: var(--radius-tag);
      font-size: 11.5px;
      font-weight: 600;
      white-space: nowrap;
    }}
    .pill-high {{ background: var(--status-danger-bg); color: var(--status-danger); }}
    .pill-mid {{ background: var(--status-warning-bg); color: var(--status-warning); }}
    .pill-ok {{ background: var(--status-success-bg); color: var(--status-success); }}
    .pill-muted {{ background: var(--bg-subtle); color: var(--text-secondary); }}

    .prob-cell {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      justify-content: flex-end;
    }}
    .prob-track {{
      width: 56px;
      height: 6px;
      border-radius: 999px;
      background: var(--border-hairline);
      overflow: hidden;
    }}
    .prob-fill {{
      height: 100%;
      border-radius: 999px;
      background: var(--accent-primary);
    }}
    .prob-fill.high {{ background: var(--status-danger); }}
    .empty-state {{
      padding: 36px 20px;
      text-align: center;
      border: 1px dashed var(--border-strong);
      border-radius: var(--radius-inner);
      background: var(--bg-subtle);
      color: var(--text-secondary);
      font-size: 13.5px;
    }}
    .pager {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      font-size: 12.5px;
      color: var(--text-secondary);
      flex-wrap: wrap;
    }}
    .guide-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .guide-box {{
      padding: 12px 14px;
      border-radius: var(--radius-inner);
      background: var(--bg-subtle);
      border: 1px solid var(--border-hairline);
      font-size: 12.5px;
      color: var(--text-secondary);
      line-height: 1.5;
    }}
    .guide-box strong {{ display: block; color: var(--text-primary); margin-bottom: 4px; font-size: 12.8px; }}
    .main-tabs {{
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--border-hairline);
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }}
    .tab-group {{
      display: inline-flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .main-tab {{
      appearance: none;
      font-family: inherit;
      font-size: 13.5px;
      font-weight: 700;
      padding: 8px 16px;
      border-radius: var(--radius-inner);
      border: 1px solid var(--border-strong);
      background: var(--bg-subtle);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all var(--transition-ui);
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }}
    .main-tab:hover {{ color: var(--text-primary); border-color: var(--accent-primary); }}
    .main-tab.active {{
      background: var(--accent-primary);
      border-color: var(--accent-primary);
      color: var(--accent-on);
    }}
    .main-tab.active .pill {{
      background: rgba(255,255,255,0.22);
      color: var(--accent-on);
    }}
    .tab-panel {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .ops-ref-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      padding: 14px 20px;
      font-size: 12.8px;
      color: var(--text-secondary);
    }}
    .caveat-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .caveat-item {{
      padding: 13px 15px;
      border-radius: var(--radius-inner);
      background: var(--bg-subtle);
      border: 1px solid var(--border-hairline);
      font-size: 12.7px;
      color: var(--text-secondary);
      line-height: 1.55;
    }}
    .caveat-item strong {{
      display: block;
      color: var(--text-primary);
      margin-bottom: 4px;
      font-size: 13px;
    }}
    tbody tr.model-row-selected {{
      background: var(--sel-row-bg);
      font-weight: 600;
    }}
    @media (max-width: 1080px) {{
      .kpi-grid, .prio-grid, .split-grid, .guide-grid, .caveat-grid {{ grid-template-columns: 1fr; }}
      .topbar-controls {{ align-items: flex-start; }}
    }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <div class="shell">
    <header class="card">
      <div class="topbar">
        <div>
          <span class="eyebrow">12회차 · 오전 9시 출고 전 사전 점검 운영</span>
          <h1>배송 운영 현황</h1>
          <div class="meta-row">
            <span class="meta-badge">최신 점검일: {m['max_date']} (09:00 기준)</span>
            <span class="meta-badge">확정 모델: {m['model_name']}</span>
            <span class="meta-badge">위험 기준: 예측확률 50% 이상 (≥50.00%)</span>
            <span class="meta-badge">재현 범위: {m['ops_period']} ({m['ops_total']:,}건)</span>
            <span class="meta-badge safe">과거 운영 재현 · 사후 배송 결과 미포함</span>
          </div>
        </div>
        <div class="topbar-controls">
          <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap; justify-content:flex-end;">
            <span class="field-label">화면 스타일</span>
            <div class="segmented" role="group" aria-label="화면 테마 선택">
              <button type="button" class="seg-btn active" data-theme-btn="a" onclick="setTheme('a')">클린모듈</button>
              <button type="button" class="seg-btn" data-theme-btn="b" onclick="setTheme('b')">헤어라인</button>
              <button type="button" class="seg-btn" data-theme-btn="c" onclick="setTheme('c')">라운드</button>
              <button type="button" class="seg-btn" data-theme-btn="d" onclick="setTheme('d')">공공고대비</button>
              <button type="button" class="seg-btn" data-theme-btn="e" onclick="setTheme('e')">다크관제</button>
              <button type="button" class="seg-btn" data-theme-btn="f" onclick="setTheme('f')">웜오렌지</button>
              <button type="button" class="seg-btn" data-theme-btn="g" onclick="setTheme('g')">그린금융</button>
              <button type="button" class="seg-btn" data-theme-btn="h" onclick="setTheme('h')">골드직각</button>
            </div>
          </div>
          <div id="active-scope-badge" class="meta-badge">현재 조회: 최신 점검일 ({m['max_date']})</div>
        </div>
      </div>
      <nav class="main-tabs" role="tablist" aria-label="화면 탭 전환">
        <div class="tab-group">
          <button type="button" role="tab" id="tab-btn-ops" class="main-tab active" aria-selected="true" onclick="switchTab('ops')">
            운영 현황 · 우선 점검 <span class="pill pill-muted">메인 화면</span>
          </button>
          <button type="button" role="tab" id="tab-btn-model" class="main-tab" aria-selected="false" onclick="switchTab('model')">
            모델 기준 · 비교 참고 <span class="pill pill-muted">백업 탭</span>
          </button>
        </div>
        <span class="field-label">추후 배포 대비 · 데이터·Plotly 내장 단일 HTML (로컬 저장 전용)</span>
      </nav>
    </header>

    <div id="tab-ops" class="tab-panel">
      <div class="notice-banner">
        <strong>과거 운영 상황 재현 안내:</strong> 이 화면은 제공된 교육용 자료로 <strong>2026년 10월 오전 9시</strong> 출고 전 점검 상황을 재현한 것이며, 실제 배송 완료일과 실제 지연 여부는 포함되어 있지 않습니다. 표시된 건수는 제공 자료 범위이며, '위험 주문'은 확정 지연이 아닌 사전 점검 후보입니다.
      </div>

    <!-- 조회 필터 -->
    <section class="card">
      <div class="filter-grid">
        <div class="field-group">
          <span class="field-label">점검 시작일</span>
          <input type="date" id="filter-start" min="{m['min_date']}" max="{m['max_date']}" value="{m['max_date']}" onchange="onFilterChange()" />
        </div>
        <div class="field-group">
          <span class="field-label">점검 종료일</span>
          <input type="date" id="filter-end" min="{m['min_date']}" max="{m['max_date']}" value="{m['max_date']}" onchange="onFilterChange()" />
        </div>
        <div class="field-group">
          <span class="field-label">기간 빠른 선택</span>
          <div style="display:flex; gap:6px; flex-wrap:wrap;">
            <button type="button" id="btn-latest" class="btn active-quick" onclick="setQuickRange('latest')">최신일 ({m['max_date']})</button>
            <button type="button" id="btn-recent3" class="btn" onclick="setQuickRange('recent3')">최근 3일</button>
            <button type="button" id="btn-recent7" class="btn" onclick="setQuickRange('recent7')">최근 7일</button>
            <button type="button" id="btn-all-oct" class="btn" onclick="setQuickRange('all')">10월 전체</button>
          </div>
        </div>
        <div class="field-group">
          <span class="field-label">물류센터</span>
          <select id="filter-center" onchange="onFilterChange()">
            <option value="전체">전체 센터</option>
          </select>
        </div>
        <div class="field-group">
          <span class="field-label">운송사</span>
          <select id="filter-carrier" onchange="onFilterChange()">
            <option value="전체">전체 운송사</option>
          </select>
        </div>
        <div class="field-group">
          <span class="field-label">&nbsp;</span>
          <button type="button" id="btn-reset" class="btn" onclick="resetAllFilters()">초기화</button>
        </div>
      </div>
    </section>

    <!-- 핵심 KPI 3개 -->
    <section class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-title"><span>조회 주문 수</span><span class="pill pill-muted" id="kpi-period-pill">{m['max_date']}</span></div>
        <div class="kpi-value" id="kpi-total">0건</div>
        <div class="kpi-sub" id="kpi-total-sub">선택 조건 내 오전 9시 점검 대상 전체</div>
      </div>
      <div class="kpi-card emphasis">
        <div class="kpi-title"><span>위험 주문 수 (위험여부=1)</span><span class="pill pill-high">50% 이상</span></div>
        <div class="kpi-value" id="kpi-risk">0건</div>
        <div class="kpi-sub" id="kpi-risk-rate">예측 위험 비율: -</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title"><span>위험 중 오늘까지 배송 (여유일수 ≤ 0)</span><span class="pill pill-mid">긴급 확인</span></div>
        <div class="kpi-value" id="kpi-due0">0건</div>
        <div class="kpi-sub" id="kpi-due0-rate">위험 주문 중 비중: -</div>
      </div>
    </section>

    <!-- 대응 우선순위 1~4순위 카드 -->
    <section class="card">
      <div class="sec-head">
        <div>
          <h2 class="sec-title">대응 우선순위 (위험 주문 기준 · 클릭해 해당 순위만 보기)</h2>
          <div class="sec-sub">우선순위는 이번 실습의 업무 기준이며, 확인 제안은 실제 지연 원인이나 조치 완료 상태가 아닙니다.</div>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <span class="meta-badge" id="prio-filter-badge">전체 위험 주문 표시 중</span>
          <button type="button" id="btn-prio-all" class="btn" onclick="selectPriority(0)">전체 위험 보기</button>
        </div>
      </div>
      <div class="prio-grid">
        <button type="button" class="prio-card" id="prio-card-1" onclick="selectPriority(1)">
          <div class="prio-top"><span class="prio-badge prio-1">1순위</span><span class="prio-cnt" id="prio-cnt-1">0건</span></div>
          <div class="prio-cond">배송여유일수 ≤ 0 + 미확보·부분확보</div>
          <div class="prio-act">확인: 즉시 확보·배송 가능 여부 확인</div>
        </button>
        <button type="button" class="prio-card" id="prio-card-2" onclick="selectPriority(2)">
          <div class="prio-top"><span class="prio-badge prio-2">2순위</span><span class="prio-cnt" id="prio-cnt-2">0건</span></div>
          <div class="prio-cond">배송여유일수 ≤ 0 + 확보완료</div>
          <div class="prio-act">확인: 출고 준비·운송사 인계 확인</div>
        </button>
        <button type="button" class="prio-card" id="prio-card-3" onclick="selectPriority(3)">
          <div class="prio-top"><span class="prio-badge prio-3">3순위</span><span class="prio-cnt" id="prio-cnt-3">0건</span></div>
          <div class="prio-cond">배송여유일수 &gt; 0 + 미확보·부분확보</div>
          <div class="prio-act">확인: 기한 내 확보·출고 가능 여부 확인</div>
        </button>
        <button type="button" class="prio-card" id="prio-card-4" onclick="selectPriority(4)">
          <div class="prio-top"><span class="prio-badge prio-4">4순위</span><span class="prio-cnt" id="prio-cnt-4">0건</span></div>
          <div class="prio-cond">배송여유일수 &gt; 0 + 확보완료</div>
          <div class="prio-act">확인: 배송 일정 확인</div>
        </button>
      </div>
    </section>

    <!-- 센터별 분포 그래프 + 요약 표 (클릭 시 조회 조건 연동) -->
    <section class="split-grid">
      <div class="card">
        <div class="sec-head">
          <div>
            <h2 class="sec-title">물류센터별 위험 주문 분포 (막대 클릭 시 센터 필터 연동)</h2>
            <div class="sec-sub">예측 위험 비율 = 위험 주문 수 ÷ 센터별 조회 주문 수 (실제 지연률이 아님 · 하루 표본으로 장기 품질 단정 금지)</div>
          </div>
          <button type="button" id="btn-chart-center-all" class="btn" style="display:none;" onclick="toggleCenterFromChart(state.center)">전체 센터 보기</button>
        </div>
        <div id="center-chart" style="height:310px;"></div>
        <div id="center-chart-empty" class="empty-state" style="display:none;">선택한 조건에 해당하는 주문이 없어 센터 그래프를 표시할 수 없습니다.</div>
      </div>

      <div class="card">
        <div class="sec-head">
          <div>
            <h2 class="sec-title">센터별 요약 (행 클릭으로 필터 전환)</h2>
            <div class="sec-sub">선택된 센터를 다시 누르면 전체 센터로 돌아갑니다.</div>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>물류센터</th>
                <th class="num">전체 주문</th>
                <th class="num">위험 주문</th>
                <th class="num">예측 위험 비율</th>
                <th class="num">위험·당일마감</th>
              </tr>
            </thead>
            <tbody id="center-tbody"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- 우선 점검 목록 -->
    <section class="card">
      <div class="sec-head">
        <div>
          <h2 class="sec-title">우선 점검 목록 <span class="pill pill-muted" id="table-count-pill">0건</span></h2>
          <div class="sec-sub">정렬: 대응 순위(1→4) → 배송여유일수 오름차순 → 지연예측확률 내림차순 → 주문ID 순</div>
        </div>
        <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
          <input type="text" id="filter-search" placeholder="주문ID·센터·운송사·상품군 검색" oninput="onSearchInput()" style="width:230px;" />
          <button type="button" id="btn-csv" class="btn primary" onclick="downloadCsv()">CSV 다운로드 (전체 0건)</button>
        </div>
      </div>

      <div id="list-empty" class="empty-state" style="display:none;">
        선택한 조회 조건·우선순위·검색어에 해당하는 위험 주문이 없습니다.
      </div>
      <div class="table-wrap" id="list-table-wrap">
        <table>
          <thead>
            <tr>
              <th class="center">순번</th>
              <th class="center">순위</th>
              <th>주문ID</th>
              <th>점검일</th>
              <th>배송약속일</th>
              <th class="num">여유일수</th>
              <th>재고 상태</th>
              <th>물류센터</th>
              <th>운송사</th>
              <th>상품군</th>
              <th class="num">지연예측확률</th>
              <th>확인할 사항 (업무 제안)</th>
            </tr>
          </thead>
          <tbody id="priority-tbody"></tbody>
        </table>
      </div>
      <div class="pager" id="pager-bar">
        <span id="pager-info">0건 표시</span>
        <div style="display:flex; gap:6px; align-items:center;">
          <button type="button" class="btn" id="btn-prev" onclick="changePage(-1)">이전</button>
          <span id="pager-page">1 / 1</span>
          <button type="button" class="btn" id="btn-next" onclick="changePage(1)">다음</button>
        </div>
      </div>
    </section>

      <section class="card ops-ref-bar">
        <div>
          <b>적용 기준 요약:</b> {m['model_name']} · 위험 임계값 <b>{int(m['threshold']*100)}% 이상(≥50.00%)</b> · 학습 <b>1~6월</b> / 선정 비교 <b>7~8월</b> / 최종 평가 <b>9월</b> · 상세 선정 이유·6개 모델 비교·주의점은 별도 백업 탭에 분리되어 있습니다.
        </div>
        <button type="button" id="btn-open-model-tab" class="btn" onclick="switchTab('model')">모델 기준 · 비교 참고 열기 →</button>
      </section>
    </div>

    <div id="tab-model" class="tab-panel" style="display:none;">
      <div class="notice-banner" style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
        <div>
          <strong>백업 참고 탭 안내:</strong> 메인 화면이 오전 9시 주문 점검에 집중할 수 있도록 최종 모델 기준, 6개 후보 모델 비교, 9월 최종 평가 및 주의사항을 별도 탭으로 분리해 보관합니다.
        </div>
        <button type="button" id="btn-back-ops" class="btn primary" onclick="switchTab('ops')">← 운영 현황으로 돌아가기</button>
      </div>

      <section class="card">
        <div class="sec-head">
          <div>
            <h2 class="sec-title">최종 운영 모델 및 사용 기준 요약</h2>
            <div class="sec-sub">학습·선정 비교·최종 평가 단계에서 설정을 고정해 10월 예측에 그대로 적용했습니다.</div>
          </div>
          <span class="meta-badge safe">설정 고정 · 9~10월 재학습/재튜닝 미실시</span>
        </div>
        <div class="guide-grid">
          <div class="guide-box">
            <strong>1. 최종 선정 모델 &amp; 입력 변수</strong>
            <b>{m['model_name']}</b><br/>
            분류기: {m['estimator']}<br/>
            주요 설정: {m['model_params_summary']}<br/>
            입력 변수: 숫자형 {len(m['numeric_inputs'])}개 · 범주형 {len(m['categorical_inputs'])}개 (총 {len(m['numeric_inputs']) + len(m['categorical_inputs'])}개)
          </div>
          <div class="guide-box">
            <strong>2. 위험 임계값 &amp; 단계별 기간</strong>
            위험 임계값: <b>예측확률 {int(m['threshold']*100)}% 이상 (≥50.00%)</b><br/>
            • 학습 기간: {m['train_period']} (7,200건)<br/>
            • 비교 선정: {m['comparison_period']} (2,400건)<br/>
            • 최종 평가: {m['final_eval_period']} (1,455건)
          </div>
          <div class="guide-box">
            <strong>3. 선택한 이유 (선정 근거)</strong>
            {m['selection_reason']}<br/>
            로지스틱 가중치(<b>1:3</b>) 대비 7~8월 지연 탐지 재현율이 높고, 9월 평가에서도 약 70% 수준을 유지해 사전 점검 놓침을 줄이는 목적에 부합합니다.
          </div>
          <div class="guide-box">
            <strong>4. 감수할 점 · 운영 부담</strong>
            {m['tradeoff']}<br/>
            실제 회사의 가용 인력·하루 처리 한도는 자료에 없으므로, 우선순위(1~4순위) 기준으로 확인 순서를 정합니다.
          </div>
        </div>
      </section>

      <section class="card">
        <div class="sec-head">
          <div>
            <h2 class="sec-title">후보 6개 모델 성능 비교 (선정 비교 기간: 7~8월 2,400건 · 임계값 50% 기준)</h2>
            <div class="sec-sub">오프라인 백업 비교표입니다. 실제 지연 416건(지연율 17.33%) 기준으로 각 모델의 지연 탐지력과 점검 부담을 비교했습니다.</div>
          </div>
        </div>
        <div id="model-comp-chart" style="height:330px;"></div>
        <div class="table-wrap" style="margin-top:12px;">
          <table>
            <thead>
              <tr>
                <th>모델명</th>
                <th class="center">선정 여부</th>
                <th class="num">점검대상</th>
                <th class="num">점검비율</th>
                <th class="num">탐지 지연(TP)</th>
                <th class="num">과잉 점검(FP)</th>
                <th class="num">놓친 지연(FN)</th>
                <th class="num">정확도</th>
                <th class="num">정밀도</th>
                <th class="num">재현율</th>
                <th class="num">F1 Score</th>
                <th class="num">ROC-AUC</th>
              </tr>
            </thead>
            <tbody id="model-comp-tbody"></tbody>
          </table>
        </div>
      </section>

      <section class="card">
        <div class="sec-head">
          <div>
            <h2 class="sec-title">선정 모델 최종 평가 (7~8월 선정 비교 vs 9월 최종 평가)</h2>
            <div class="sec-sub">9월은 7~8월에 고른 모델·설정을 변경 없이 검증한 기간이며, 기간에 따라 성능 수치가 달라질 수 있음을 참고합니다.</div>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>기간 구분</th>
                <th>기간</th>
                <th class="num">전체 주문</th>
                <th class="num">실제 지연(비율)</th>
                <th class="num">점검대상(비율)</th>
                <th class="num">TP</th>
                <th class="num">FP</th>
                <th class="num">FN</th>
                <th class="num">정확도</th>
                <th class="num">정밀도</th>
                <th class="num">재현율</th>
                <th class="num">F1 Score</th>
                <th class="num">ROC-AUC</th>
              </tr>
            </thead>
            <tbody id="sep-eval-tbody"></tbody>
          </table>
        </div>
      </section>

      <section class="card">
        <div class="sec-head">
          <div>
            <h2 class="sec-title">현업 적용 시 주의점 및 고려해야 할 사항</h2>
            <div class="sec-sub">점검 담당자가 대시보드 수치를 해석할 때 함께 확인해야 할 한계와 점검 기준입니다.</div>
          </div>
        </div>
        <div class="caveat-grid">
          <div class="caveat-item">
            <strong>① 위험 주문 = 실제 지연 확정이 아님 (오탐 감수 필요)</strong>
            9월 최종 평가 기준 정밀도는 <b>31.14%</b>(501건 점검 중 실제 지연 156건 · 실제 정상 345건)입니다. 즉, 위험 목록에는 정상 배송될 주문도 다수 포함되므로 현장에서는 불필요한 취소·과잉 조치가 나지 않도록 재고 상태와 약속일을 확인하는 절차로 사용합니다.
          </div>
          <div class="caveat-item">
            <strong>② 50% 미만 ‘정상 예측’에서도 지연이 발생할 수 있음 (놓친 지연 FN)</strong>
            7~8월 101건, 9월 67건의 실제 지연은 예측확률 50% 미만이어서 점검 목록에서 빠졌습니다. 따라서 긴급 고객 문의나 특수 품목·특정 센터 물량 급증 시에는 정상 예측 구간 주문도 보조 모니터링이 필요합니다.
          </div>
          <div class="caveat-item">
            <strong>③ 시기별 데이터 변화와 주기적 사후 검증 필요</strong>
            7~8월(<b>재현율 75.72% · AUC 0.8020</b>)에서 9월(<b>재현율 69.96% · AUC 0.7905</b>)로 이동하며 실제 지연율과 탐지 수치가 다소 내려왔습니다. 운영 데이터가 축적된 뒤에는 실제 배송 완료 결과를 대조해 임계값·가중치 재점검 여부를 검토해야 합니다.
          </div>
          <div class="caveat-item">
            <strong>④ 10월 데이터 범위·인력 한도·사후 정보 비포함 주의</strong>
            10월 운영 화면(<b>945건</b>)은 과거 오전 9시 시점을 재현한 교육용 표본이며 <b>실제 배송 완료일과 실제 지연 여부</b>가 전혀 포함되지 않았습니다. 표시 건수를 실제 회사 전체 물량이나 가용 인력 처리 한도로 해석하지 않도록 유의해야 합니다.
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const DATA = {data_json};
    const state = {{
      tab: "ops",
      start: DATA.meta.max_date,
      end: DATA.meta.max_date,
      center: "전체",
      carrier: "전체",
      priority: 0,
      search: "",
      page: 1,
      pageSize: 25,
      theme: "a",
    }};

    function cssVar(name) {{
      return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    }}

    function fmtPct(num, den) {{
      if (!den || den <= 0) return "계산 불가";
      return ((num / den) * 100).toFixed(2) + "%";
    }}

    function setTheme(t) {{
      state.theme = t;
      document.documentElement.setAttribute("data-theme", t);
      document.querySelectorAll("[data-theme-btn]").forEach(btn => {{
        btn.classList.toggle("active", btn.dataset.themeBtn === t);
      }});
      renderAll();
    }}

    function initSelects() {{
      const cSel = document.getElementById("filter-center");
      DATA.meta.centers.forEach(c => {{
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        cSel.appendChild(opt);
      }});
      const kSel = document.getElementById("filter-carrier");
      DATA.meta.carriers.forEach(c => {{
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        kSel.appendChild(opt);
      }});
    }}

    function setQuickRange(kind) {{
      if (kind === "latest") {{
        state.start = DATA.meta.max_date;
        state.end = DATA.meta.max_date;
      }} else if (kind === "recent3") {{
        state.start = DATA.meta.recent3_start;
        state.end = DATA.meta.max_date;
      }} else if (kind === "recent7") {{
        state.start = DATA.meta.recent7_start;
        state.end = DATA.meta.max_date;
      }} else {{
        state.start = DATA.meta.min_date;
        state.end = DATA.meta.max_date;
      }}
      document.getElementById("filter-start").value = state.start;
      document.getElementById("filter-end").value = state.end;
      state.page = 1;
      renderAll();
    }}

    function onFilterChange() {{
      state.start = document.getElementById("filter-start").value || DATA.meta.min_date;
      state.end = document.getElementById("filter-end").value || DATA.meta.max_date;
      state.center = document.getElementById("filter-center").value;
      state.carrier = document.getElementById("filter-carrier").value;
      state.page = 1;
      renderAll();
    }}

    function selectPriority(p) {{
      state.priority = (state.priority === p) ? 0 : p;
      state.page = 1;
      renderAll();
    }}

    function onSearchInput() {{
      state.search = document.getElementById("filter-search").value.trim().toLowerCase();
      state.page = 1;
      renderAll();
    }}

    function resetAllFilters() {{
      state.start = DATA.meta.max_date;
      state.end = DATA.meta.max_date;
      state.center = "전체";
      state.carrier = "전체";
      state.priority = 0;
      state.search = "";
      state.page = 1;
      document.getElementById("filter-start").value = state.start;
      document.getElementById("filter-end").value = state.end;
      document.getElementById("filter-center").value = "전체";
      document.getElementById("filter-carrier").value = "전체";
      document.getElementById("filter-search").value = "";
      renderAll();
    }}

    function toggleCenterFromChart(centerName) {{
      state.center = (state.center === centerName) ? "전체" : centerName;
      document.getElementById("filter-center").value = state.center;
      state.page = 1;
      renderAll();
    }}

    function getBaseFiltered() {{
      return DATA.orders.filter(o => {{
        if (state.start && o.d < state.start) return false;
        if (state.end && o.d > state.end) return false;
        if (state.center !== "전체" && o.ctr !== state.center) return false;
        if (state.carrier !== "전체" && o.car !== state.carrier) return false;
        return true;
      }});
    }}

    function getDateCarrierFilteredForCenterChart() {{
      return DATA.orders.filter(o => {{
        if (state.start && o.d < state.start) return false;
        if (state.end && o.d > state.end) return false;
        if (state.carrier !== "전체" && o.car !== state.carrier) return false;
        return true;
      }});
    }}

    function getSortedRiskOrders(baseOrders) {{
      const q = state.search;
      return baseOrders
        .filter(o => {{
          if (o.risk !== 1) return false;
          if (state.priority !== 0 && o.prio !== state.priority) return false;
          if (q) {{
            const hay = `${{o.id}} ${{o.ctr}} ${{o.car}} ${{o.cat}} ${{o.inv}} ${{o.d}}`.toLowerCase();
            if (!hay.includes(q)) return false;
          }}
          return true;
        }})
        .sort((a, b) => {{
          if (a.prio !== b.prio) return a.prio - b.prio;
          if (a.slack !== b.slack) return a.slack - b.slack;
          if (b.prob !== a.prob) return b.prob - a.prob;
          return a.id.localeCompare(b.id);
        }});
    }}

    function renderQuickButtons() {{
      const isLatest = state.start === DATA.meta.max_date && state.end === DATA.meta.max_date;
      const isR3 = state.start === DATA.meta.recent3_start && state.end === DATA.meta.max_date;
      const isR7 = state.start === DATA.meta.recent7_start && state.end === DATA.meta.max_date;
      const isAll = state.start === DATA.meta.min_date && state.end === DATA.meta.max_date;
      document.getElementById("btn-latest").classList.toggle("active-quick", isLatest);
      document.getElementById("btn-recent3").classList.toggle("active-quick", isR3);
      document.getElementById("btn-recent7").classList.toggle("active-quick", isR7);
      document.getElementById("btn-all-oct").classList.toggle("active-quick", isAll);

      const pTxt = isLatest
        ? `최신 점검일 (${{DATA.meta.max_date}})`
        : isR3
          ? `최근 3일 (${{DATA.meta.recent3_start}} ~ ${{DATA.meta.max_date}})`
          : isR7
            ? `최근 7일 (${{DATA.meta.recent7_start}} ~ ${{DATA.meta.max_date}})`
            : (state.start === state.end ? state.start : `${{state.start}} ~ ${{state.end}}`);
      document.getElementById("kpi-period-pill").textContent = pTxt;
      document.getElementById("active-scope-badge").textContent =
        `현재 조회: ${{pTxt}} · 센터: ${{state.center}} · 운송사: ${{state.carrier}}`;
    }}

    function renderKpisAndPriorities(baseOrders) {{
      const total = baseOrders.length;
      const riskOrders = baseOrders.filter(o => o.risk === 1);
      const riskCnt = riskOrders.length;
      const due0Cnt = riskOrders.filter(o => o.slack <= 0).length;

      document.getElementById("kpi-total").textContent = `${{total.toLocaleString()}}건`;
      document.getElementById("kpi-risk").textContent = `${{riskCnt.toLocaleString()}}건`;
      document.getElementById("kpi-due0").textContent = `${{due0Cnt.toLocaleString()}}건`;
      document.getElementById("kpi-risk-rate").textContent = `예측 위험 비율: ${{fmtPct(riskCnt, total)}}`;
      document.getElementById("kpi-due0-rate").textContent = `위험 주문 중 비중: ${{fmtPct(due0Cnt, riskCnt)}}`;

      for (let r = 1; r <= 4; r++) {{
        const c = riskOrders.filter(o => o.prio === r).length;
        document.getElementById(`prio-cnt-${{r}}`).textContent = `${{c.toLocaleString()}}건`;
        document.getElementById(`prio-card-${{r}}`).classList.toggle("active", state.priority === r);
      }}
      document.getElementById("btn-prio-all").classList.toggle("active-quick", state.priority === 0);
      document.getElementById("prio-filter-badge").textContent =
        state.priority === 0 ? `전체 위험 주문 (${{riskCnt.toLocaleString()}}건)` : `${{state.priority}}순위만 필터링 중`;
    }}

    function renderCenterChartAndTable(baseOrders) {{
      // Center breakdown respects date + carrier; if a center is selected in filter, show all centers for comparison
      // OR show matching scope. Wait: prompt 8-1 says 현황·목록·그래프에 같은 조회 조건 적용, 8-3 says 센터 선택 시 조회 조건과 화면이 함께 바뀜.
      // Showing baseOrders in summary & highlighting selected center (or filtering to baseOrders) keeps 100% filter consistency!
      const clearBtn = document.getElementById("btn-chart-center-all");
      if (state.center !== "전체") {{
        clearBtn.style.display = "inline-flex";
        clearBtn.textContent = `${{state.center}} 해제 (전체 보기)`;
      }} else {{
        clearBtn.style.display = "none";
      }}

      const scopeOrders = baseOrders;
      const hasAny = scopeOrders.length > 0;
      const chartEl = document.getElementById("center-chart");
      const emptyEl = document.getElementById("center-chart-empty");

      const tbody = document.getElementById("center-tbody");
      const activeCenters = state.center === "전체" ? DATA.meta.centers : [state.center];

      const rows = activeCenters.map(ctr => {{
        const sub = scopeOrders.filter(o => o.ctr === ctr);
        const tot = sub.length;
        const rsk = sub.filter(o => o.risk === 1).length;
        const due0 = sub.filter(o => o.risk === 1 && o.slack <= 0).length;
        const rate = tot > 0 ? (rsk / tot) * 100 : null;
        return {{ ctr, tot, rsk, due0, rate, rateTxt: fmtPct(rsk, tot) }};
      }});

      tbody.innerHTML = rows.map(r => `
        <tr class="clickable-row ${{state.center === r.ctr ? 'selected-center' : ''}}" onclick="toggleCenterFromChart('${{r.ctr}}')">
          <td><b>${{r.ctr}}</b></td>
          <td class="num">${{r.tot.toLocaleString()}}건</td>
          <td class="num"><b>${{r.rsk.toLocaleString()}}건</b></td>
          <td class="num">${{r.rateTxt}}</td>
          <td class="num">${{r.due0.toLocaleString()}}건</td>
        </tr>
      `).join("");

      if (!hasAny) {{
        chartEl.style.display = "none";
        emptyEl.style.display = "block";
        return;
      }}
      chartEl.style.display = "block";
      emptyEl.style.display = "none";

      const textCol = cssVar("--text-primary") || "#1F2937";
      const gridCol = cssVar("--border-hairline") || "#E5E7EB";
      const totalCol = cssVar("--chart-total") || "#CBD5E1";
      const riskCol = cssVar("--chart-risk") || "#3182F6";
      const lineCol = cssVar("--chart-line") || "#DC2626";

      const traces = [
        {{
          name: "전체 주문 수",
          x: rows.map(r => r.ctr),
          y: rows.map(r => r.tot),
          type: "bar",
          xaxis: "x",
          yaxis: "y",
          marker: {{ color: totalCol }},
          text: rows.map(r => `${{r.tot}}건`),
          textposition: "outside",
          hovertemplate: "<b>%{{x}}</b><br>전체 주문: %{{y:,}}건<extra></extra>",
        }},
        {{
          name: "위험 주문 수",
          x: rows.map(r => r.ctr),
          y: rows.map(r => r.rsk),
          type: "bar",
          xaxis: "x",
          yaxis: "y",
          marker: {{ color: riskCol }},
          text: rows.map(r => `${{r.rsk}}건`),
          textposition: "outside",
          customdata: rows.map(r => r.rateTxt),
          hovertemplate: "<b>%{{x}}</b><br>위험 주문: %{{y:,}}건 (예측 위험 비율: %{{customdata}})<extra></extra>",
        }},
        {{
          name: "예측 위험 비율 (%)",
          x: rows.map(r => r.ctr),
          y: rows.map(r => r.rate !== null ? Number(r.rate.toFixed(2)) : 0),
          type: "bar",
          xaxis: "x2",
          yaxis: "y2",
          marker: {{ color: lineCol }},
          text: rows.map(r => r.rateTxt),
          textposition: "outside",
          hovertemplate: "<b>%{{x}}</b><br>예측 위험 비율: %{{y:.2f}}% (실제 지연률 아님)<extra></extra>",
        }},
      ];

      const maxBar = Math.max(...rows.map(r => r.tot), 10);
      const maxRate = Math.max(...rows.map(r => r.rate || 0), 50);
      const layout = {{
        barmode: "group",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {{ family: cssVar("--font-sans"), color: textCol, size: 12 }},
        margin: {{ t: 36, b: 36, l: 46, r: 18 }},
        xaxis: {{ domain: [0.0, 0.56], gridcolor: gridCol }},
        yaxis: {{ title: "주문 수 (건)", range: [0, maxBar * 1.24], gridcolor: gridCol }},
        xaxis2: {{ domain: [0.66, 1.0], gridcolor: gridCol }},
        yaxis2: {{ title: "예측 위험 비율 (%)", anchor: "x2", range: [0, Math.min(115, Math.max(65, maxRate * 1.32))], gridcolor: gridCol }},
        annotations: [
          {{ text: "<b>센터별 전체 vs 위험 주문 수</b>", x: 0.28, y: 1.07, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 12.5, color: textCol }} }},
          {{ text: "<b>예측 위험 비율 (위험÷전체)</b>", x: 0.83, y: 1.07, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 12.5, color: textCol }} }},
        ],
        legend: {{ orientation: "h", y: -0.18, x: 0.5, xanchor: "center" }},
      }};

      Plotly.react(chartEl, traces, layout, {{ displayModeBar: false, responsive: true }}).then(() => {{
        chartEl.removeAllListeners && chartEl.removeAllListeners("plotly_click");
        chartEl.on("plotly_click", ev => {{
          if (ev && ev.points && ev.points.length > 0) {{
            toggleCenterFromChart(ev.points[0].x);
          }}
        }});
      }});
    }}

    function renderPriorityTable(sortedRisk) {{
      const total = sortedRisk.length;
      document.getElementById("table-count-pill").textContent = `${{total.toLocaleString()}}건`;
      const csvBtn = document.getElementById("btn-csv");
      csvBtn.textContent = `CSV 다운로드 (전체 ${{total.toLocaleString()}}건)`;
      csvBtn.disabled = total === 0;

      const wrap = document.getElementById("list-table-wrap");
      const emptyEl = document.getElementById("list-empty");
      const pager = document.getElementById("pager-bar");

      if (total === 0) {{
        wrap.style.display = "none";
        pager.style.display = "none";
        emptyEl.style.display = "block";
        document.getElementById("priority-tbody").innerHTML = "";
        return;
      }}

      wrap.style.display = "block";
      pager.style.display = "flex";
      emptyEl.style.display = "none";

      const maxPage = Math.max(1, Math.ceil(total / state.pageSize));
      if (state.page > maxPage) state.page = maxPage;
      const startIdx = (state.page - 1) * state.pageSize;
      const slice = sortedRisk.slice(startIdx, startIdx + state.pageSize);

      document.getElementById("pager-info").textContent = `${{startIdx + 1}}~${{Math.min(startIdx + slice.length, total)}}건 / 전체 ${{total.toLocaleString()}}건`;
      document.getElementById("pager-page").textContent = `${{state.page}} / ${{maxPage}}`;
      document.getElementById("btn-prev").disabled = state.page <= 1;
      document.getElementById("btn-next").disabled = state.page >= maxPage;

      const tbody = document.getElementById("priority-tbody");
      tbody.innerHTML = slice.map((o, idx) => {{
        const invCls = o.inv === "미확보" ? "pill-high" : (o.inv === "부분확보" ? "pill-mid" : "pill-ok");
        const pCls = `prio-${{o.prio}}`;
        const pct = (o.prob * 100).toFixed(2) + "%";
        return `
          <tr>
            <td class="center">${{startIdx + idx + 1}}</td>
            <td class="center"><span class="prio-badge ${{pCls}}">${{o.prio}}순위</span></td>
            <td><b>${{o.id}}</b></td>
            <td>${{o.d}}</td>
            <td>${{o.due}}</td>
            <td class="num">${{o.slack}}일</td>
            <td><span class="pill ${{invCls}}">${{o.inv}}</span></td>
            <td>${{o.ctr}}</td>
            <td>${{o.car}}</td>
            <td>${{o.cat}}</td>
            <td class="num">
              <div class="prob-cell">
                <div class="prob-track"><div class="prob-fill ${{o.prob >= 0.70 ? 'high' : ''}}" style="width:${{Math.min(100, o.prob * 100).toFixed(1)}}%;"></div></div>
                <span><b>${{pct}}</b></span>
              </div>
            </td>
            <td>${{o.act}}</td>
          </tr>
        `;
      }}).join("");
    }}

    function changePage(delta) {{
      state.page += delta;
      renderAll();
    }}

    function buildCsvPayload() {{
      const baseOrders = getBaseFiltered();
      const rows = getSortedRiskOrders(baseOrders);
      const headers = ["대응순위", "주문ID", "점검기준일시", "배송약속일", "배송여유일수", "재고확보상태", "물류센터", "운송사", "상품군", "지연예측확률", "위험여부", "확인할사항"];
      const esc = v => {{
        const s = String(v ?? "");
        return /[",\\n]/.test(s) ? `"${{s.replace(/"/g, '""')}}"` : s;
      }};
      const lines = [headers.join(",")];
      rows.forEach(o => {{
        lines.push([
          `${{o.prio}}순위`, o.id, o.dt, o.due, o.slack, o.inv, o.ctr, o.car, o.cat, o.prob, o.risk, o.act
        ].map(esc).join(","));
      }});
      return {{ rowCount: rows.length, content: "\\uFEFF" + lines.join("\\r\\n") }};
    }}

    function downloadCsv() {{
      const payload = buildCsvPayload();
      if (payload.rowCount === 0) return;
      const blob = new Blob([payload.content], {{ type: "text/csv;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `delivery_priority_${{state.start}}_${{state.end}}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    function switchTab(tabName) {{
      state.tab = tabName === "model" ? "model" : "ops";
      const isOps = state.tab === "ops";
      document.getElementById("tab-ops").style.display = isOps ? "flex" : "none";
      document.getElementById("tab-model").style.display = isOps ? "none" : "flex";
      const btnOps = document.getElementById("tab-btn-ops");
      const btnModel = document.getElementById("tab-btn-model");
      btnOps.classList.toggle("active", isOps);
      btnOps.setAttribute("aria-selected", isOps ? "true" : "false");
      btnModel.classList.toggle("active", !isOps);
      btnModel.setAttribute("aria-selected", !isOps ? "true" : "false");
      if (isOps) {{
        Plotly.Plots.resize(document.getElementById("center-chart"));
      }} else {{
        renderModelReference();
        Plotly.Plots.resize(document.getElementById("model-comp-chart"));
      }}
      window.scrollTo({{ top: 0, behavior: "instant" }});
    }}

    function renderModelReference() {{
      const rows = DATA.model_comparison || [];
      const tbody = document.getElementById("model-comp-tbody");
      tbody.innerHTML = rows.map(r => `
        <tr class="${{r.is_selected ? 'model-row-selected' : ''}}">
          <td><b>${{r.model}}</b></td>
          <td class="center">${{r.is_selected ? '<span class="pill pill-ok">운영 선정</span>' : '<span class="pill pill-muted">후보</span>'}}</td>
          <td class="num">${{r.flagged.toLocaleString()}}건</td>
          <td class="num">${{(r.flag_rate * 100).toFixed(2)}}%</td>
          <td class="num"><b>${{r.tp.toLocaleString()}}건</b></td>
          <td class="num">${{r.fp.toLocaleString()}}건</td>
          <td class="num">${{r.fn.toLocaleString()}}건</td>
          <td class="num">${{(r.acc * 100).toFixed(2)}}%</td>
          <td class="num">${{(r.prec * 100).toFixed(2)}}%</td>
          <td class="num"><b>${{(r.rec * 100).toFixed(2)}}%</b></td>
          <td class="num">${{(r.f1 * 100).toFixed(2)}}%</td>
          <td class="num">${{r.auc.toFixed(4)}}</td>
        </tr>
      `).join("");

      const sepRows = DATA.sep_eval || [];
      document.getElementById("sep-eval-tbody").innerHTML = sepRows.map(s => `
        <tr class="${{s.split.includes('9월') ? 'model-row-selected' : ''}}">
          <td><b>${{s.split}}</b></td>
          <td>${{s.period}}</td>
          <td class="num">${{s.total.toLocaleString()}}건</td>
          <td class="num">${{s.actual_pos.toLocaleString()}}건 (${{(s.delay_rate * 100).toFixed(2)}}%)</td>
          <td class="num">${{s.flagged.toLocaleString()}}건 (${{(s.flag_rate * 100).toFixed(2)}}%)</td>
          <td class="num"><b>${{s.tp.toLocaleString()}}건</b></td>
          <td class="num">${{s.fp.toLocaleString()}}건</td>
          <td class="num">${{s.fn.toLocaleString()}}건</td>
          <td class="num">${{(s.acc * 100).toFixed(2)}}%</td>
          <td class="num">${{(s.prec * 100).toFixed(2)}}%</td>
          <td class="num"><b>${{(s.rec * 100).toFixed(2)}}%</b></td>
          <td class="num">${{(s.f1 * 100).toFixed(2)}}%</td>
          <td class="num">${{s.auc.toFixed(4)}}</td>
        </tr>
      `).join("");

      const textCol = cssVar("--text-primary") || "#1F2937";
      const gridCol = cssVar("--border-hairline") || "#E5E7EB";
      const totalCol = cssVar("--chart-total") || "#CBD5E1";
      const riskCol = cssVar("--chart-risk") || "#3182F6";
      const lineCol = cssVar("--chart-line") || "#DC2626";
      const warnCol = cssVar("--status-warning") || "#D97706";

      const shortNames = rows.map(r => r.is_selected ? `★ ${{r.model}}` : r.model);
      const traces = [
        {{
          name: "재현율 (지연 포착률 %)",
          x: shortNames,
          y: rows.map(r => Number((r.rec * 100).toFixed(2))),
          type: "bar",
          marker: {{ color: riskCol }},
          text: rows.map(r => `${{(r.rec * 100).toFixed(2)}}%`),
          textposition: "outside",
        }},
        {{
          name: "정밀도 (점검 적중률 %)",
          x: shortNames,
          y: rows.map(r => Number((r.prec * 100).toFixed(2))),
          type: "bar",
          marker: {{ color: lineCol }},
          text: rows.map(r => `${{(r.prec * 100).toFixed(2)}}%`),
          textposition: "outside",
        }},
        {{
          name: "F1 Score (%)",
          x: shortNames,
          y: rows.map(r => Number((r.f1 * 100).toFixed(2))),
          type: "bar",
          marker: {{ color: warnCol }},
          text: rows.map(r => `${{(r.f1 * 100).toFixed(2)}}%`),
          textposition: "outside",
        }},
        {{
          name: "정확도 (%)",
          x: shortNames,
          y: rows.map(r => Number((r.acc * 100).toFixed(2))),
          type: "bar",
          marker: {{ color: totalCol }},
          text: rows.map(r => `${{(r.acc * 100).toFixed(2)}}%`),
          textposition: "outside",
        }},
      ];

      Plotly.react(document.getElementById("model-comp-chart"), traces, {{
        barmode: "group",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {{ family: cssVar("--font-sans"), color: textCol, size: 11.5 }},
        margin: {{ t: 28, b: 44, l: 48, r: 16 }},
        xaxis: {{ gridcolor: gridCol }},
        yaxis: {{ title: "성능 지표 (%)", range: [0, 108], gridcolor: gridCol }},
        legend: {{ orientation: "h", y: -0.22, x: 0.5, xanchor: "center" }},
      }}, {{ displayModeBar: false, responsive: true }});
    }}

    function renderAll() {{
      const baseOrders = getBaseFiltered();
      const sortedRisk = getSortedRiskOrders(baseOrders);
      renderQuickButtons();
      renderKpisAndPriorities(baseOrders);
      renderCenterChartAndTable(baseOrders);
      renderPriorityTable(sortedRisk);
      if (state.tab === "model") renderModelReference();
    }}

    initSelects();
    renderAll();
  </script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    html = generate_html(payload)
    OUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"Saved self-contained dashboard: {OUT_HTML} ({size_kb:,.1f} KB, orders={len(payload['orders'])})")


if __name__ == "__main__":
    main()
