"""Build Option 3 Enhanced Operations Workbench Dashboard (results/delivery_operations_extra.html)."""

import json
import sys
from pathlib import Path
import pandas as pd
import plotly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_operations_dashboard_12 as base_dash


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUT_HTML = RESULTS_DIR / "delivery_operations_extra.html"


def build_extra_payload() -> dict:
    """Reuse base Session 12 payload and enrich orders with leak-free operational detail fields."""
    payload = base_dash.build_payload()
    ops_df = pd.read_csv(base_dash.OPS_CSV)
    assert not (base_dash.FORBIDDEN_COLS & set(ops_df.columns)), "Forbidden outcome columns found!"

    detail_map = {}
    for _, r in ops_df.iterrows():
        oid = str(r["주문ID"])
        detail_map[oid] = {
            "ord_dt": str(r["주문일시"]),
            "qty": int(r["상품수량"]),
            "kg": round(float(r["총중량_kg"]), 2),
            "km": int(round(float(r["배송거리_km"]))),
            "pend": int(r["센터출고대기건수"]),
            "plan": int(r["센터당일처리계획건수"]),
            "elapsed": round(float(r["주문후_경과시간"]), 1),
            "load": round(float(r["센터물량부하율"]), 4),
            "dow": str(r["점검요일"]),
        }

    for o in payload["orders"]:
        o.update(detail_map[o["id"]])

    return payload


def generate_extra_html(payload: dict) -> str:
    """Render self-contained Option 3 HTML with Order Detail Viewer, Facts vs Questions, and Status/Memo Persistence."""
    plotly_js = plotly.offline.get_plotlyjs()
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    m = payload["meta"]

    # Strict leak guard on serialized payload
    assert "실제배송완료일" not in data_json and "배송지연여부" not in data_json

    return f"""<!DOCTYPE html>
<html lang="ko" data-theme="a">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>배송 운영 현황 · 주문 점검 워크벤치 (추가실습 선택 3)</title>
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
      --accent-on: #ffffff;
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
      --accent-on: #ffffff;
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
      padding: 20px 24px 48px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-card);
      box-shadow: var(--shadow-card);
      padding: 16px 20px;
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
      margin-bottom: 6px;
    }}
    h1 {{
      margin: 0;
      font-size: 24px;
      font-weight: var(--weight-heading);
      letter-spacing: var(--tracking-heading);
      line-height: 1.25;
    }}
    .meta-row {{
      margin-top: 7px;
      display: flex;
      flex-wrap: wrap;
      gap: 7px 8px;
      font-size: 12px;
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
      font-size: 11.6px;
      font-weight: 600;
      padding: 4px 9px;
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
    .main-tabs {{
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid var(--border-hairline);
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }}
    .tab-group {{ display: inline-flex; gap: 8px; flex-wrap: wrap; }}
    .main-tab {{
      appearance: none;
      font-family: inherit;
      font-size: 13px;
      font-weight: 700;
      padding: 7px 14px;
      border-radius: var(--radius-inner);
      border: 1px solid var(--border-strong);
      background: var(--bg-subtle);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all var(--transition-ui);
      display: inline-flex;
      align-items: center;
      gap: 6px;
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
    .tab-panel {{ display: flex; flex-direction: column; gap: 14px; }}
    .notice-banner {{
      padding: 10px 15px;
      border-radius: var(--radius-inner);
      background: var(--accent-weak);
      color: var(--accent-ink);
      font-size: 12.6px;
      line-height: 1.5;
    }}
    .filter-grid {{
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      gap: 10px;
    }}
    .field-group {{ display: flex; flex-direction: column; gap: 4px; }}
    .field-label {{ font-size: 11.4px; font-weight: 600; color: var(--text-muted); }}
    input[type="date"], input[type="text"], select, textarea {{
      font-family: inherit;
      font-size: 12.8px;
      font-weight: 500;
      color: var(--text-primary);
      background: var(--bg-surface);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-inner);
      padding: 5px 10px;
      outline: none;
    }}
    input[type="date"], input[type="text"], select {{ height: 34px; }}
    textarea {{ width: 100%; resize: vertical; line-height: 1.45; min-height: 68px; }}
    input:focus, select:focus, textarea:focus {{ border-color: var(--accent-primary); }}
    .btn {{
      appearance: none;
      font-family: inherit;
      font-size: 12.4px;
      font-weight: 600;
      height: 34px;
      padding: 0 11px;
      border-radius: var(--radius-inner);
      border: 1px solid var(--border-strong);
      background: var(--bg-surface);
      color: var(--text-primary);
      cursor: pointer;
      transition: all var(--transition-ui);
      display: inline-flex;
      align-items: center;
      gap: 5px;
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
    .btn.sm {{ height: 29px; font-size: 11.8px; padding: 0 9px; }}
    .btn:disabled {{ opacity: 0.45; cursor: not-allowed; }}

    .kpi-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .kpi-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-card);
      box-shadow: var(--shadow-card);
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }}
    .kpi-card.emphasis {{ background: var(--sel-row-bg); border-color: var(--accent-primary); }}
    .kpi-title {{
      font-size: 12.6px;
      font-weight: 600;
      color: var(--text-secondary);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .kpi-value {{
      font-size: 27px;
      font-weight: var(--weight-heading);
      letter-spacing: var(--tracking-heading);
      line-height: 1.2;
    }}
    .kpi-sub {{ font-size: 12px; color: var(--text-secondary); }}

    .sec-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .sec-title {{
      margin: 0;
      font-size: 15.5px;
      font-weight: var(--weight-heading);
      letter-spacing: var(--tracking-heading);
      display: flex;
      align-items: center;
      gap: 7px;
      flex-wrap: wrap;
    }}
    .sec-sub {{ font-size: 12px; color: var(--text-secondary); margin-top: 2px; }}

    .prio-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .prio-card {{
      text-align: left;
      appearance: none;
      width: 100%;
      font-family: inherit;
      background: var(--bg-subtle);
      border: 1.5px solid var(--border-hairline);
      border-radius: var(--radius-inner);
      padding: 11px 13px;
      cursor: pointer;
      transition: all var(--transition-ui);
      display: flex;
      flex-direction: column;
      gap: 4px;
      color: var(--text-primary);
    }}
    .prio-card:hover {{ border-color: var(--border-strong); }}
    .prio-card.active {{
      background: var(--sel-row-bg);
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 1px var(--accent-primary);
    }}
    .prio-top {{ display: flex; justify-content: space-between; align-items: center; width: 100%; }}
    .prio-badge {{ font-size: 11.3px; font-weight: 700; padding: 2px 7px; border-radius: var(--radius-tag); }}
    .prio-1 {{ background: var(--status-danger-bg); color: var(--status-danger); }}
    .prio-2 {{ background: var(--status-warning-bg); color: var(--status-warning); }}
    .prio-3 {{ background: var(--accent-weak); color: var(--accent-ink); }}
    .prio-4 {{ background: var(--status-success-bg); color: var(--status-success); }}
    .prio-cnt {{ font-size: 20px; font-weight: var(--weight-heading); }}
    .prio-cond {{ font-size: 11.8px; font-weight: 600; color: var(--text-secondary); }}
    .prio-act {{ font-size: 11.8px; color: var(--text-primary); font-weight: 500; }}

    /* 2-Column Operations Workbench */
    .workbench-grid {{
      display: grid;
      grid-template-columns: 1.28fr 1fr;
      gap: 14px;
      align-items: start;
    }}
    .detail-panel {{
      background: var(--bg-subtle);
      border: 1.5px solid var(--border-strong);
      border-radius: var(--radius-inner);
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 11px;
    }}
    .detail-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 8px;
      padding-bottom: 9px;
      border-bottom: 1px solid var(--border-hairline);
    }}
    .detail-id {{ font-size: 18px; font-weight: var(--weight-heading); letter-spacing: -0.015em; }}
    .detail-subhead {{
      font-size: 11.5px;
      font-weight: 700;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      margin-bottom: 5px;
    }}
    .fact-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 10px;
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-tag);
      padding: 9px 11px;
      font-size: 12px;
    }}
    .fact-cell {{ display: flex; flex-direction: column; gap: 1px; }}
    .fact-k {{ font-size: 11px; color: var(--text-muted); font-weight: 600; }}
    .fact-v {{ font-size: 12.2px; color: var(--text-primary); font-weight: 600; }}
    .q-box {{
      background: var(--bg-surface);
      border: 1px dashed var(--accent-primary);
      border-radius: var(--radius-tag);
      padding: 9px 11px;
      font-size: 12px;
      color: var(--text-primary);
    }}
    .q-box ul {{ margin: 4px 0 5px; padding-left: 17px; }}
    .q-box li {{ margin-bottom: 3px; line-height: 1.45; }}
    .q-note {{ font-size: 11px; color: var(--text-muted); line-height: 1.4; }}
    .work-box {{
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-tag);
      padding: 10px 11px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .status-group {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .status-opt {{
      flex: 1;
      min-width: 78px;
      appearance: none;
      font-family: inherit;
      font-size: 12px;
      font-weight: 700;
      padding: 6px 8px;
      border-radius: var(--radius-tag);
      border: 1px solid var(--border-strong);
      background: var(--bg-subtle);
      color: var(--text-secondary);
      cursor: pointer;
      transition: all var(--transition-ui);
    }}
    .status-opt.active[data-st="미점검"] {{
      background: var(--bg-surface);
      border-color: var(--text-secondary);
      color: var(--text-primary);
      box-shadow: 0 0 0 1px var(--text-secondary);
    }}
    .status-opt.active[data-st="확인중"] {{
      background: var(--status-warning-bg);
      border-color: var(--status-warning);
      color: var(--status-warning);
      box-shadow: 0 0 0 1px var(--status-warning);
    }}
    .status-opt.active[data-st="조치완료"] {{
      background: var(--status-success-bg);
      border-color: var(--status-success);
      color: var(--status-success);
      box-shadow: 0 0 0 1px var(--status-success);
    }}

    .split-grid {{ display: grid; grid-template-columns: 1.32fr 1fr; gap: 14px; align-items: stretch; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12.2px; }}
    th, td {{
      padding: 7px 8px;
      border-bottom: 1px solid var(--border-hairline);
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }}
    th {{
      font-size: 11.3px;
      font-weight: 600;
      color: var(--text-muted);
      background: var(--bg-subtle);
    }}
    th.num, td.num {{ text-align: right; }}
    th.center, td.center {{ text-align: center; }}
    tbody tr.clickable-row {{ cursor: pointer; transition: background var(--transition-ui); }}
    tbody tr.clickable-row:hover {{ background: var(--bg-subtle); }}
    tbody tr.selected-order-row {{
      background: var(--sel-row-bg);
      box-shadow: inset 3px 0 0 var(--accent-primary);
    }}
    tbody tr.selected-center {{ background: var(--sel-row-bg); font-weight: 600; }}

    .pill {{
      display: inline-flex;
      align-items: center;
      padding: 2px 7px;
      border-radius: var(--radius-tag);
      font-size: 11.2px;
      font-weight: 600;
      white-space: nowrap;
    }}
    .pill-high {{ background: var(--status-danger-bg); color: var(--status-danger); }}
    .pill-mid {{ background: var(--status-warning-bg); color: var(--status-warning); }}
    .pill-ok {{ background: var(--status-success-bg); color: var(--status-success); }}
    .pill-muted {{ background: var(--bg-subtle); color: var(--text-secondary); border: 1px solid var(--border-hairline); }}

    .empty-state {{
      padding: 28px 18px;
      text-align: center;
      border: 1px dashed var(--border-strong);
      border-radius: var(--radius-inner);
      background: var(--bg-subtle);
      color: var(--text-secondary);
      font-size: 13px;
    }}
    .pager {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
      font-size: 12px;
      color: var(--text-secondary);
      flex-wrap: wrap;
    }}
    .ops-ref-bar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      padding: 12px 18px;
      font-size: 12.4px;
      color: var(--text-secondary);
    }}
    .guide-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .guide-box {{
      padding: 12px 14px;
      border-radius: var(--radius-inner);
      background: var(--bg-subtle);
      border: 1px solid var(--border-hairline);
      font-size: 12.4px;
      color: var(--text-secondary);
      line-height: 1.5;
    }}
    .guide-box strong {{ display: block; color: var(--text-primary); margin-bottom: 4px; font-size: 12.8px; }}
    .caveat-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .caveat-item {{
      padding: 13px 15px;
      border-radius: var(--radius-inner);
      background: var(--bg-subtle);
      border: 1px solid var(--border-hairline);
      font-size: 12.6px;
      color: var(--text-secondary);
      line-height: 1.55;
    }}
    .caveat-item strong {{ display: block; color: var(--text-primary); margin-bottom: 4px; font-size: 13px; }}
    tbody tr.model-row-selected {{ background: var(--sel-row-bg); font-weight: 600; }}

    @media (max-width: 1120px) {{
      .kpi-grid, .prio-grid, .workbench-grid, .split-grid, .guide-grid, .caveat-grid {{ grid-template-columns: 1fr; }}
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
          <span class="eyebrow">12회차 · 추가실습 선택 3 · 담당자 점검 편의 개선판</span>
          <h1>배송 운영 현황 · 주문 점검 워크벤치</h1>
          <div class="meta-row">
            <span class="meta-badge">최신 점검일: {m['max_date']} (09:00 기준)</span>
            <span class="meta-badge">확정 모델: {m['model_name']}</span>
            <span class="meta-badge">위험 기준: 예측확률 50% 이상 (≥50.00%)</span>
            <span class="meta-badge">재현 범위: {m['ops_period']} ({m['ops_total']:,}건)</span>
            <span class="meta-badge safe">사후 배송 결과 미포함 · 점검기록 주문ID별 저장</span>
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
            운영 현황 · 주문 점검 워크벤치 <span class="pill pill-muted">메인 화면</span>
          </button>
          <button type="button" role="tab" id="tab-btn-model" class="main-tab" aria-selected="false" onclick="switchTab('model')">
            모델 기준 · 비교 참고 <span class="pill pill-muted">백업 탭</span>
          </button>
        </div>
        <span class="field-label">개선 기능: ① 주문 상세(사실·확인질문 분리) ② 주문ID별 점검상태·메모 저장/불러오기</span>
      </nav>
    </header>

    <div id="tab-ops" class="tab-panel">
      <div class="notice-banner">
        <strong>과거 운영 상황 재현 안내:</strong> 이 화면은 제공된 교육용 자료로 <strong>2026년 10월 오전 9시</strong> 출고 전 점검 상황을 재현한 것이며, 실제 배송 완료일과 실제 지연 여부는 포함되어 있지 않습니다. 담당자 점검 상태(미점검·확인중·조치완료)와 메모는 작업 기록이며 모델 예측이나 실제 배송 결과를 바꾸지 않습니다.
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
            <div style="display:flex; gap:5px; flex-wrap:wrap;">
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
            <span class="field-label">재고 상태</span>
            <select id="filter-inv" onchange="onFilterChange()">
              <option value="전체">전체 재고</option>
              <option value="미확보계열">미확보·부분확보</option>
              <option value="확보완료">확보완료</option>
            </select>
          </div>
          <div class="field-group">
            <span class="field-label">점검 상태 (기록)</span>
            <select id="filter-work" onchange="onFilterChange()">
              <option value="전체">전체 기록</option>
              <option value="미점검">미점검만</option>
              <option value="확인중">확인중만</option>
              <option value="조치완료">조치완료만</option>
            </select>
          </div>
          <div class="field-group">
            <span class="field-label">&nbsp;</span>
            <button type="button" id="btn-reset" class="btn" onclick="resetAllFilters()">필터 초기화</button>
          </div>
        </div>
      </section>

      <!-- 핵심 KPI 3종 -->
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
          <div style="display:flex; gap:7px; align-items:center; flex-wrap:wrap;">
            <span class="meta-badge" id="prio-filter-badge">전체 위험 주문 표시 중</span>
            <button type="button" id="btn-prio-all" class="btn sm" onclick="selectPriority(0)">전체 위험 보기</button>
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

      <!-- 개선된 2열 주문 점검 워크벤치 (우선 점검 목록 + 상세/사실·확인질문/상태·메모) -->
      <section class="card" id="workbench-card">
        <div class="sec-head">
          <div>
            <h2 class="sec-title">
              우선 점검 목록 &amp; 주문 상세·메모 워크벤치
              <span class="pill pill-muted" id="table-count-pill">0건</span>
              <span class="pill pill-muted" id="work-prog-pill">미점검 0 · 확인중 0 · 조치완료 0</span>
            </h2>
            <div class="sec-sub">목록 행을 클릭하면 오른쪽에 [확인된 사실], [추가로 확인할 질문], [주문ID별 점검 상태·메모] 패널이 바로 열립니다.</div>
          </div>
          <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap;">
            <input type="text" id="filter-search" placeholder="주문ID·센터·운송사·지역·메모 검색" oninput="onSearchInput()" style="width:215px;" />
            <input type="file" id="import-notes-input" accept=".json,application/json" style="display:none;" onchange="importNotesFile(event)" />
            <button type="button" id="btn-export-notes" class="btn" onclick="exportNotesJson()">기록 저장(.json)</button>
            <button type="button" id="btn-import-notes" class="btn" onclick="document.getElementById('import-notes-input').click()">기록 불러오기</button>
            <button type="button" id="btn-clear-notes" class="btn" onclick="clearAllWorkNotes()">기록 초기화</button>
            <button type="button" id="btn-csv" class="btn primary" onclick="downloadCsv()">CSV 다운로드 (전체 0건)</button>
          </div>
        </div>

        <div class="workbench-grid">
          <!-- 좌측: 목록 테이블 -->
          <div>
            <div id="list-empty" class="empty-state" style="display:none;">
              선택한 조회 조건·우선순위·점검 상태·검색어에 해당하는 위험 주문이 없습니다.
            </div>
            <div class="table-wrap" id="list-table-wrap">
              <table>
                <thead>
                  <tr>
                    <th class="center">순번</th>
                    <th class="center">순위</th>
                    <th>주문ID</th>
                    <th>점검일</th>
                    <th class="num">여유</th>
                    <th>재고</th>
                    <th>센터·운송사</th>
                    <th class="num">지연확률</th>
                    <th class="center">점검상태</th>
                  </tr>
                </thead>
                <tbody id="priority-tbody"></tbody>
              </table>
            </div>
            <div class="pager" id="pager-bar">
              <span id="pager-info">0건 표시</span>
              <div style="display:flex; gap:6px; align-items:center;">
                <button type="button" class="btn sm" id="btn-prev" onclick="changePage(-1)">이전</button>
                <span id="pager-page">1 / 1</span>
                <button type="button" class="btn sm" id="btn-next" onclick="changePage(1)">다음</button>
              </div>
            </div>
          </div>

          <!-- 우측: 주문 상세 · 확인된 사실/질문 분리 · 상태/메모 -->
          <div class="detail-panel" id="order-detail-panel">
            <div id="detail-empty" class="empty-state" style="display:none;">
              조회 조건에 맞는 주문이 없어 상세정보를 표시할 수 없습니다.
            </div>
            <div id="detail-body">
              <div class="detail-head">
                <div>
                  <div style="display:flex; align-items:center; gap:7px; flex-wrap:wrap;">
                    <span class="detail-id" id="d-id">-</span>
                    <span class="prio-badge prio-1" id="d-prio">1순위</span>
                    <span class="pill pill-high" id="d-prob">-</span>
                  </div>
                  <div class="sec-sub" id="d-act-hint" style="margin-top:3px;">-</div>
                </div>
                <span class="pill pill-muted" id="d-work-badge">미점검</span>
              </div>

              <div style="margin-top:9px;">
                <div class="detail-subhead">
                  <span>1) 확인된 사실 (10월 점검 시점 입력·예측 정보)</span>
                  <span class="pill pill-ok">사실 정보</span>
                </div>
                <div class="fact-grid" id="d-facts-grid"></div>
              </div>

              <div style="margin-top:9px;">
                <div class="detail-subhead">
                  <span>2) 추가로 확인할 질문 (담당자 조치 검토용)</span>
                  <span class="pill pill-mid">예측 원인 아님</span>
                </div>
                <div class="q-box">
                  <ul id="d-questions-list"></ul>
                  <div class="q-note">※ 위 질문은 담당자가 현장 확인 시 참고할 업무 질문이며, 모델이 위험으로 분류한 직접 원인으로 해석하지 않습니다.</div>
                </div>
              </div>

              <div style="margin-top:9px;">
                <div class="detail-subhead">
                  <span>3) 담당자 점검 상태 · 메모 (주문ID별 저장)</span>
                  <span class="field-label" id="d-save-msg">미점검 상태 · 저장된 메모 없음</span>
                </div>
                <div class="work-box">
                  <div class="status-group" role="group" aria-label="점검 상태 선택">
                    <button type="button" class="status-opt active" data-st="미점검" onclick="setDetailStatus('미점검')">미점검</button>
                    <button type="button" class="status-opt" data-st="확인중" onclick="setDetailStatus('확인중')">확인중</button>
                    <button type="button" class="status-opt" data-st="조치완료" onclick="setDetailStatus('조치완료')">조치완료</button>
                  </div>
                  <textarea id="detail-memo" placeholder="확인한 재고 확보 시각, 운송사 인계 메모 등을 기록하세요..."></textarea>
                  <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; flex-wrap:wrap;">
                    <span class="q-note">작업 기록은 예측 위험여부·실제 배송 결과를 바꾸지 않습니다.</span>
                    <div style="display:flex; gap:6px;">
                      <button type="button" class="btn sm" id="btn-clear-order-note" onclick="clearSingleNote()">메모 비우기</button>
                      <button type="button" class="btn primary sm" id="btn-save-note" onclick="saveCurrentOrderNote()">이 주문 기록 저장</button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- 센터별 분포 그래프 + 요약 표 -->
      <section class="split-grid">
        <div class="card">
          <div class="sec-head">
            <div>
              <h2 class="sec-title">물류센터별 위험 주문 분포 (막대 클릭 시 센터 필터 연동)</h2>
              <div class="sec-sub">예측 위험 비율 = 위험 주문 수 ÷ 센터별 조회 주문 수 (실제 지연률이 아님 · 하루 표본으로 장기 품질 단정 금지)</div>
            </div>
            <button type="button" id="btn-chart-center-all" class="btn sm" style="display:none;" onclick="toggleCenterFromChart(state.center)">전체 센터 보기</button>
          </div>
          <div id="center-chart" style="height:290px;"></div>
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

      <section class="card ops-ref-bar">
        <div>
          <b>적용 기준 요약:</b> {m['model_name']} · 위험 임계값 <b>{int(m['threshold']*100)}% 이상(≥50.00%)</b> · 학습 <b>1~6월</b> / 선정 비교 <b>7~8월</b> / 최종 평가 <b>9월</b> · 상세 선정 근거·6개 모델 비교는 백업 탭에 분리되어 있습니다.
        </div>
        <button type="button" id="btn-open-model-tab" class="btn" onclick="switchTab('model')">모델 기준 · 비교 참고 열기 →</button>
      </section>
    </div>

    <!-- 탭 2: 모델 기준 · 비교 참고 (백업 탭) -->
    <div id="tab-model" class="tab-panel" style="display:none;">
      <div class="notice-banner" style="display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;">
        <div>
          <strong>백업 참고 탭 안내:</strong> 메인 화면이 오전 9시 주문 점검 워크벤치에 집중할 수 있도록 최종 모델 기준, 6개 후보 모델 비교, 9월 최종 평가 및 주의사항을 별도 탭으로 분리해 보관합니다.
        </div>
        <button type="button" id="btn-back-ops" class="btn primary" onclick="switchTab('ops')">← 운영 워크벤치로 돌아가기</button>
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
            9월 최종 평가 기준 정밀도는 <b>31.14%</b>(501건 점검 중 실제 지연 156건 · 실제 정상 345건)입니다. 위험 목록은 사전 확인 대상일 뿐 확정 지연이 아닙니다.
          </div>
          <div class="caveat-item">
            <strong>② 50% 미만 ‘정상 예측’에서도 지연이 발생할 수 있음 (놓친 지연 FN)</strong>
            7~8월 101건, 9월 67건의 실제 지연은 50% 미만으로 분류되었습니다. 긴급 문의·센터 물량 급증 구간은 정상 예측 주문도 보조 점검이 필요합니다.
          </div>
          <div class="caveat-item">
            <strong>③ 시기별 데이터 변화와 주기적 사후 검증 필요</strong>
            7~8월(<b>재현율 75.72% · AUC 0.8020</b>) 대비 9월(<b>재현율 69.96% · AUC 0.7905</b>)에 수치가 다소 내려왔으므로 운영 결과 축적 후 주기적 재점검이 필요합니다.
          </div>
          <div class="caveat-item">
            <strong>④ 10월 데이터 범위·인력 한도·사후 정보 비포함 주의</strong>
            10월 운영 화면(<b>945건</b>)은 과거 오전 9시 재현 자료이며 <b>실제 배송 완료일과 실제 지연 여부</b>가 전혀 포함되지 않았습니다.
          </div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const DATA = {data_json};
    const STORAGE_KEY = "delivery_ops_extra_worklogs_v1";

    const state = {{
      tab: "ops",
      start: DATA.meta.max_date,
      end: DATA.meta.max_date,
      center: "전체",
      carrier: "전체",
      invFilter: "전체",
      workFilter: "전체",
      priority: 0,
      search: "",
      page: 1,
      pageSize: 12,
      selectedId: null,
      draftStatus: "미점검",
      theme: "a",
      notes: {{}},
    }};

    function loadStoredNotes() {{
      try {{
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {{
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object") state.notes = parsed;
        }}
      }} catch (e) {{
        state.notes = {{}};
      }}
    }}

    function persistNotes() {{
      try {{
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state.notes));
      }} catch (e) {{
        // ignore storage quota errors gracefully
      }}
    }}

    function getOrderLog(orderId) {{
      const entry = state.notes[orderId];
      if (!entry || typeof entry !== "object") {{
        return {{ status: "미점검", memo: "", updated_at: "" }};
      }}
      const st = ["미점검", "확인중", "조치완료"].includes(entry.status) ? entry.status : "미점검";
      return {{
        status: st,
        memo: typeof entry.memo === "string" ? entry.memo : "",
        updated_at: typeof entry.updated_at === "string" ? entry.updated_at : "",
      }};
    }}

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
      state.invFilter = document.getElementById("filter-inv").value;
      state.workFilter = document.getElementById("filter-work").value;
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
      state.invFilter = "전체";
      state.workFilter = "전체";
      state.priority = 0;
      state.search = "";
      state.page = 1;
      state.selectedId = null;
      document.getElementById("filter-start").value = state.start;
      document.getElementById("filter-end").value = state.end;
      document.getElementById("filter-center").value = "전체";
      document.getElementById("filter-carrier").value = "전체";
      document.getElementById("filter-inv").value = "전체";
      document.getElementById("filter-work").value = "전체";
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
        if (state.invFilter === "미확보계열" && o.inv === "확보완료") return false;
        if (state.invFilter === "확보완료" && o.inv !== "확보완료") return false;
        return true;
      }});
    }}

    function getSortedRiskOrders(baseOrders) {{
      const q = state.search;
      return baseOrders
        .filter(o => {{
          if (o.risk !== 1) return false;
          if (state.priority !== 0 && o.prio !== state.priority) return false;
          const log = getOrderLog(o.id);
          if (state.workFilter !== "전체" && log.status !== state.workFilter) return false;
          if (q) {{
            const hay = `${{o.id}} ${{o.ctr}} ${{o.car}} ${{o.cat}} ${{o.reg}} ${{o.inv}} ${{o.d}} ${{log.status}} ${{log.memo}}`.toLowerCase();
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

      let cntUn = 0, cntIng = 0, cntDone = 0;
      riskOrders.forEach(o => {{
        const st = getOrderLog(o.id).status;
        if (st === "확인중") cntIng++;
        else if (st === "조치완료") cntDone++;
        else cntUn++;
      }});
      document.getElementById("work-prog-pill").textContent = `미점검 ${{cntUn}} · 확인중 ${{cntIng}} · 조치완료 ${{cntDone}}`;
    }}

    function statusBadgeHtml(st) {{
      if (st === "조치완료") return `<span class="pill pill-ok">조치완료</span>`;
      if (st === "확인중") return `<span class="pill pill-mid">확인중</span>`;
      return `<span class="pill pill-muted">미점검</span>`;
    }}

    function buildQuestionsForOrder(o) {{
      const qs = [];
      if (o.inv !== "확보완료") {{
        qs.push(`재고 확보 상태가 '${{o.inv}}'입니다. 금일 입고·할당 예정 시각과 대체 출고 가능 여부를 확인했나요?`);
      }}
      if (o.slack <= 0) {{
        qs.push(`배송여유일수가 ${{o.slack}}일(당일 마감 이하)입니다. 금일 출고 컷오프 전 상차 및 '${{o.car}}' 인계가 가능한가요?`);
      }} else {{
        qs.push(`약속일(${{o.due}})까지 여유 ${{o.slack}}일 구간입니다. 예정된 출고·집하 일정에 지연 소지가 없나요?`);
      }}
      if (o.load >= 1.1) {{
        qs.push(`${{o.ctr}} 물량 부하율이 ${{(o.load * 100).toFixed(2)}}%(대기 ${{o.pend.toLocaleString()}}/계획 ${{o.plan.toLocaleString()}})입니다. 우선 피킹 배정이 필요한가요?`);
      }} else {{
        qs.push(`'${{o.reg}}' 방면(${{o.km}}km · '${{o.typ}}') 운송사의 집하 차량·일정에 특이사항이 없나요?`);
      }}
      return qs;
    }}

    function selectOrder(orderId) {{
      state.selectedId = orderId;
      renderAll();
    }}

    function setDetailStatus(st) {{
      state.draftStatus = st;
      document.querySelectorAll(".status-opt").forEach(b => {{
        b.classList.toggle("active", b.dataset.st === st);
      }});
    }}

    function saveCurrentOrderNote() {{
      if (!state.selectedId) return;
      const memo = document.getElementById("detail-memo").value.trim();
      const st = state.draftStatus || "미점검";
      if (st === "미점검" && memo === "") {{
        delete state.notes[state.selectedId];
      }} else {{
        state.notes[state.selectedId] = {{
          status: st,
          memo: memo,
          updated_at: new Date().toISOString().slice(0, 19).replace("T", " "),
        }};
      }}
      persistNotes();
      renderAll();
      const msgEl = document.getElementById("d-save-msg");
      if (msgEl) msgEl.textContent = `${{st}} · 저장 완료 (${{state.selectedId}})`;
    }}

    function clearSingleNote() {{
      if (!state.selectedId) return;
      delete state.notes[state.selectedId];
      persistNotes();
      state.draftStatus = "미점검";
      document.getElementById("detail-memo").value = "";
      renderAll();
    }}

    function clearAllWorkNotes() {{
      state.notes = {{}};
      persistNotes();
      renderAll();
    }}

    function buildNotesBackupPayload() {{
      return {{
        version: 1,
        exported_at: new Date().toISOString(),
        order_count: Object.keys(state.notes).length,
        notes: state.notes,
      }};
    }}

    function exportNotesJson() {{
      const payload = buildNotesBackupPayload();
      const blob = new Blob([JSON.stringify(payload, null, 2)], {{ type: "application/json;charset=utf-8;" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `delivery_work_notes_${{state.start}}_${{state.end}}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    function applyImportedNotesObject(obj) {{
      const src = (obj && obj.notes && typeof obj.notes === "object") ? obj.notes : obj;
      if (!src || typeof src !== "object") return 0;
      let count = 0;
      Object.entries(src).forEach(([oid, val]) => {{
        if (val && typeof val === "object") {{
          const st = ["미점검", "확인중", "조치완료"].includes(val.status) ? val.status : "미점검";
          const memo = typeof val.memo === "string" ? val.memo : "";
          const updated_at = typeof val.updated_at === "string" ? val.updated_at : "";
          if (st !== "미점검" || memo !== "") {{
            state.notes[oid] = {{ status: st, memo, updated_at }};
            count++;
          }}
        }}
      }});
      persistNotes();
      renderAll();
      return count;
    }}

    function importNotesFile(ev) {{
      const file = ev.target.files && ev.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = e => {{
        try {{
          const parsed = JSON.parse(String(e.target.result || "{{}}"));
          const n = applyImportedNotesObject(parsed);
          const msgEl = document.getElementById("d-save-msg");
          if (msgEl) msgEl.textContent = `JSON 불러오기 완료 (${{n}}건 반영)`;
        }} catch (err) {{
          alert("유효한 JSON 작업 기록 파일이 아닙니다.");
        }}
        ev.target.value = "";
      }};
      reader.readAsText(file, "utf-8");
    }}

    function renderDetailPanel(sortedRisk) {{
      const emptyEl = document.getElementById("detail-empty");
      const bodyEl = document.getElementById("detail-body");
      if (sortedRisk.length === 0) {{
        state.selectedId = null;
        emptyEl.style.display = "block";
        bodyEl.style.display = "none";
        return;
      }}
      emptyEl.style.display = "none";
      bodyEl.style.display = "block";

      let order = sortedRisk.find(o => o.id === state.selectedId);
      if (!order) {{
        order = sortedRisk[0];
        state.selectedId = order.id;
      }}

      const log = getOrderLog(order.id);
      state.draftStatus = log.status;
      document.getElementById("d-id").textContent = order.id;
      const pEl = document.getElementById("d-prio");
      pEl.className = `prio-badge prio-${{order.prio}}`;
      pEl.textContent = `${{order.prio}}순위`;
      document.getElementById("d-prob").textContent = `예측확률 ${{(order.prob * 100).toFixed(2)}}%`;
      document.getElementById("d-act-hint").textContent = `우선순위 기준: ${{order.why}} → ${{order.act}}`;
      document.getElementById("d-work-badge").outerHTML = statusBadgeHtml(log.status).replace('<span ', '<span id="d-work-badge" ');

      document.getElementById("d-facts-grid").innerHTML = `
        <div class="fact-cell"><span class="fact-k">점검기준일시 (요일)</span><span class="fact-v">${{order.dt}} (${{order.dow}})</span></div>
        <div class="fact-cell"><span class="fact-k">주문일시 (경과시간)</span><span class="fact-v">${{order.ord_dt}} (${{order.elapsed}}h)</span></div>
        <div class="fact-cell"><span class="fact-k">배송약속일 (여유일수)</span><span class="fact-v">${{order.due}} (${{order.slack}}일)</span></div>
        <div class="fact-cell"><span class="fact-k">재고확보상태</span><span class="fact-v">${{order.inv}}</span></div>
        <div class="fact-cell"><span class="fact-k">물류센터 · 물량부하율</span><span class="fact-v">${{order.ctr}} (${{(order.load * 100).toFixed(2)}}% · ${{order.pend}}/${{order.plan}})</span></div>
        <div class="fact-cell"><span class="fact-k">배송지역 · 배송거리</span><span class="fact-v">${{order.reg}} · ${{order.km}} km</span></div>
        <div class="fact-cell"><span class="fact-k">운송사 · 배송유형</span><span class="fact-v">${{order.car}} · ${{order.typ}}</span></div>
        <div class="fact-cell"><span class="fact-k">상품군 · 수량 · 중량</span><span class="fact-v">${{order.cat}} · ${{order.qty}}개 · ${{order.kg}} kg</span></div>
      `;

      const qs = buildQuestionsForOrder(order);
      document.getElementById("d-questions-list").innerHTML = qs.map(q => `<li>${{q}}</li>`).join("");

      document.querySelectorAll(".status-opt").forEach(b => {{
        b.classList.toggle("active", b.dataset.st === log.status);
      }});
      document.getElementById("detail-memo").value = log.memo;
      document.getElementById("d-save-msg").textContent =
        (log.status === "미점검" && !log.memo)
          ? "미점검 상태 · 저장된 메모 없음"
          : `저장됨 (${{log.status}}${{log.updated_at ? ' · ' + log.updated_at : ''}})`;
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

      document.getElementById("priority-tbody").innerHTML = slice.map((o, idx) => {{
        const invCls = o.inv === "미확보" ? "pill-high" : (o.inv === "부분확보" ? "pill-mid" : "pill-ok");
        const pCls = `prio-${{o.prio}}`;
        const pct = (o.prob * 100).toFixed(2) + "%";
        const log = getOrderLog(o.id);
        const selCls = o.id === state.selectedId ? "selected-order-row" : "";
        return `
          <tr class="clickable-row ${{selCls}}" onclick="selectOrder('${{o.id}}')">
            <td class="center">${{startIdx + idx + 1}}</td>
            <td class="center"><span class="prio-badge ${{pCls}}">${{o.prio}}순위</span></td>
            <td><b>${{o.id}}</b></td>
            <td>${{o.d}}</td>
            <td class="num">${{o.slack}}일</td>
            <td><span class="pill ${{invCls}}">${{o.inv}}</span></td>
            <td>${{o.ctr}} · ${{o.car}}</td>
            <td class="num"><b>${{pct}}</b></td>
            <td class="center">${{statusBadgeHtml(log.status)}}</td>
          </tr>
        `;
      }}).join("");
    }}

    function renderCenterChartAndTable(baseOrders) {{
      const clearBtn = document.getElementById("btn-chart-center-all");
      if (state.center !== "전체") {{
        clearBtn.style.display = "inline-flex";
        clearBtn.textContent = `${{state.center}} 해제 (전체 보기)`;
      }} else {{
        clearBtn.style.display = "none";
      }}

      const hasAny = baseOrders.length > 0;
      const chartEl = document.getElementById("center-chart");
      const emptyEl = document.getElementById("center-chart-empty");
      const activeCenters = state.center === "전체" ? DATA.meta.centers : [state.center];

      const rows = activeCenters.map(ctr => {{
        const sub = baseOrders.filter(o => o.ctr === ctr);
        const tot = sub.length;
        const rsk = sub.filter(o => o.risk === 1).length;
        const due0 = sub.filter(o => o.risk === 1 && o.slack <= 0).length;
        const rate = tot > 0 ? (rsk / tot) * 100 : null;
        return {{ ctr, tot, rsk, due0, rate, rateTxt: fmtPct(rsk, tot) }};
      }});

      document.getElementById("center-tbody").innerHTML = rows.map(r => `
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
        }},
      ];

      const maxBar = Math.max(...rows.map(r => r.tot), 10);
      const maxRate = Math.max(...rows.map(r => r.rate || 0), 50);
      Plotly.react(chartEl, traces, {{
        barmode: "group",
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: {{ family: cssVar("--font-sans"), color: textCol, size: 12 }},
        margin: {{ t: 34, b: 36, l: 46, r: 18 }},
        xaxis: {{ domain: [0.0, 0.56], gridcolor: gridCol }},
        yaxis: {{ title: "주문 수 (건)", range: [0, maxBar * 1.24], gridcolor: gridCol }},
        xaxis2: {{ domain: [0.66, 1.0], gridcolor: gridCol }},
        yaxis2: {{ title: "예측 위험 비율 (%)", anchor: "x2", range: [0, Math.min(115, Math.max(65, maxRate * 1.32))], gridcolor: gridCol }},
        annotations: [
          {{ text: "<b>센터별 전체 vs 위험 주문 수</b>", x: 0.28, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 12.2, color: textCol }} }},
          {{ text: "<b>예측 위험 비율 (위험÷전체)</b>", x: 0.83, y: 1.08, xref: "paper", yref: "paper", showarrow: false, font: {{ size: 12.2, color: textCol }} }},
        ],
        legend: {{ orientation: "h", y: -0.18, x: 0.5, xanchor: "center" }},
      }}, {{ displayModeBar: false, responsive: true }}).then(() => {{
        chartEl.removeAllListeners && chartEl.removeAllListeners("plotly_click");
        chartEl.on("plotly_click", ev => {{
          if (ev && ev.points && ev.points.length > 0) toggleCenterFromChart(ev.points[0].x);
        }});
      }});
    }}

    function changePage(delta) {{
      state.page += delta;
      renderAll();
    }}

    function buildCsvPayload() {{
      const baseOrders = getBaseFiltered();
      const rows = getSortedRiskOrders(baseOrders);
      const headers = [
        "대응순위", "주문ID", "점검기준일시", "배송약속일", "배송여유일수",
        "재고확보상태", "물류센터", "운송사", "배송유형", "상품군", "배송지역",
        "지연예측확률", "위험여부", "담당자점검상태", "담당자메모"
      ];
      const esc = v => {{
        const s = String(v ?? "");
        return /[",\\n]/.test(s) ? `"${{s.replace(/"/g, '""')}}"` : s;
      }};
      const lines = [headers.join(",")];
      rows.forEach(o => {{
        const log = getOrderLog(o.id);
        lines.push([
          `${{o.prio}}순위`, o.id, o.dt, o.due, o.slack, o.inv, o.ctr, o.car, o.typ, o.cat, o.reg,
          o.prob, o.risk, log.status, log.memo
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
      a.download = `delivery_priority_extra_${{state.start}}_${{state.end}}.csv`;
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
      document.getElementById("model-comp-tbody").innerHTML = rows.map(r => `
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

      Plotly.react(document.getElementById("model-comp-chart"), [
        {{ name: "재현율 (지연 포착률 %)", x: shortNames, y: rows.map(r => Number((r.rec * 100).toFixed(2))), type: "bar", marker: {{ color: riskCol }}, text: rows.map(r => `${{(r.rec * 100).toFixed(2)}}%`), textposition: "outside" }},
        {{ name: "정밀도 (점검 적중률 %)", x: shortNames, y: rows.map(r => Number((r.prec * 100).toFixed(2))), type: "bar", marker: {{ color: lineCol }}, text: rows.map(r => `${{(r.prec * 100).toFixed(2)}}%`), textposition: "outside" }},
        {{ name: "F1 Score (%)", x: shortNames, y: rows.map(r => Number((r.f1 * 100).toFixed(2))), type: "bar", marker: {{ color: warnCol }}, text: rows.map(r => `${{(r.f1 * 100).toFixed(2)}}%`), textposition: "outside" }},
        {{ name: "정확도 (%)", x: shortNames, y: rows.map(r => Number((r.acc * 100).toFixed(2))), type: "bar", marker: {{ color: totalCol }}, text: rows.map(r => `${{(r.acc * 100).toFixed(2)}}%`), textposition: "outside" }},
      ], {{
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
      renderDetailPanel(sortedRisk);
      renderPriorityTable(sortedRisk);
      renderCenterChartAndTable(baseOrders);
      if (state.tab === "model") renderModelReference();
    }}

    loadStoredNotes();
    initSelects();
    renderAll();
  </script>
</body>
</html>
"""


def main() -> None:
    payload = build_extra_payload()
    html = generate_extra_html(payload)
    OUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"Saved Option 3 dashboard: {OUT_HTML} ({size_kb:,.1f} KB, orders={len(payload['orders'])})")


if __name__ == "__main__":
    main()
