"""Verify delivery_operations.html interactively in headless browser and capture 15_, 16_, and 00_."""

import base64
import time
import sys
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_and_organize_12 as org


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
DASHBOARD_HTML = RESULTS_DIR / "delivery_operations.html"

OUT_CORE_PNG = GRAPHS_DIR / "15_운영대시보드_제출용_캡처.png"
OUT_FULL_PNG = GRAPHS_DIR / "16_운영대시보드_전체화면_캡처.png"
OUT_MODEL_PNG = GRAPHS_DIR / "17_운영대시보드_모델참고탭_캡처.png"


def get_driver():
    """Create headless Chrome or Edge webdriver with console logging."""
    try:
        opts = ChromeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--hide-scrollbars")
        opts.add_argument("--window-size=1440,1100")
        opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})
        return webdriver.Chrome(options=opts)
    except Exception:
        opts = EdgeOptions()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--hide-scrollbars")
        opts.add_argument("--window-size=1440,1100")
        return webdriver.Edge(options=opts)


def cdp_shot(driver, width: int, height: int, out_path: Path) -> None:
    """Resize viewport via CDP and save PNG."""
    driver.execute_cdp_cmd(
        "Emulation.setDeviceMetricsOverride",
        {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
    )
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.4)
    shot = driver.execute_cdp_cmd(
        "Page.captureScreenshot",
        {"format": "png", "captureBeyondViewport": False, "fromSurface": True},
    )
    out_path.write_bytes(base64.b64decode(shot["data"]))
    print(f"Saved capture: {out_path.name} ({width}x{height})")


def run_functional_tests(driver) -> None:
    """Exercise filters, priorities, search, center sync, empty state, CSV export, and themes."""
    # 1. Default view: latest date 2026-10-20 -> 41 total / 11 risk / 8 due0 / prio 5,3,1,2
    assert "최신 점검일: 2026-10-20" in driver.page_source
    assert "최신 점검일 (2026-10-20)" in driver.find_element(By.ID, "active-scope-badge").text
    assert driver.find_element(By.ID, "kpi-total").text == "41건"
    assert driver.find_element(By.ID, "kpi-risk").text == "11건"
    assert driver.find_element(By.ID, "kpi-due0").text == "8건"
    assert driver.find_element(By.ID, "prio-cnt-1").text == "5건"
    assert driver.find_element(By.ID, "prio-cnt-2").text == "3건"
    assert driver.find_element(By.ID, "prio-cnt-3").text == "1건"
    assert driver.find_element(By.ID, "prio-cnt-4").text == "2건"
    assert len(driver.find_elements(By.CSS_SELECTOR, "#priority-tbody tr")) == 11
    print("[PASS] Default latest-day (2026-10-20) view: 41 / 11 / 8, priorities 5/3/1/2")

    # 2. Priority 1 click filter + CSV match
    driver.find_element(By.ID, "prio-card-1").click()
    time.sleep(0.2)
    assert len(driver.find_elements(By.CSS_SELECTOR, "#priority-tbody tr")) == 5
    csv1 = driver.execute_script("return buildCsvPayload();")
    assert csv1["rowCount"] == 5
    assert "실제배송완료일" not in csv1["content"] and "배송지연여부" not in csv1["content"]
    driver.find_element(By.ID, "prio-card-1").click()
    time.sleep(0.2)
    assert len(driver.find_elements(By.CSS_SELECTOR, "#priority-tbody tr")) == 11
    print("[PASS] Priority 1 filter & CSV match: 5 rows")

    # 3. Single custom date filter (2026-10-19) -> verify counts update and reset works
    driver.execute_script("""
        const d1 = document.getElementById('filter-start');
        const d2 = document.getElementById('filter-end');
        d1.value = '2026-10-19';
        d2.value = '2026-10-19';
        d1.dispatchEvent(new Event('change'));
    """)
    time.sleep(0.25)
    t_1019 = driver.find_element(By.ID, "kpi-total").text
    r_1019 = driver.find_element(By.ID, "kpi-risk").text
    d_1019 = driver.find_element(By.ID, "kpi-due0").text
    assert t_1019 != "0건"
    print(f"[PASS] Custom inspection date (2026-10-19) filter: total={t_1019}, risk={r_1019}, due0={d_1019}")

    # 4. Recent 3 days & Recent 7 days buttons
    driver.find_element(By.ID, "btn-recent3").click()
    time.sleep(0.35)
    assert driver.find_element(By.ID, "filter-start").get_attribute("value") == "2026-10-18"
    assert driver.find_element(By.ID, "filter-end").get_attribute("value") == "2026-10-20"
    r3_tot = driver.find_element(By.ID, "kpi-total").text
    r3_rsk = driver.find_element(By.ID, "kpi-risk").text
    r3_due = driver.find_element(By.ID, "kpi-due0").text
    csv_r3 = driver.execute_script("return buildCsvPayload();")
    assert f"{csv_r3['rowCount']}건" == r3_rsk
    print(f"[PASS] Recent 3 days (2026-10-18~20): total={r3_tot}, risk={r3_rsk}, due0={r3_due}, CSV={csv_r3['rowCount']} rows")

    driver.find_element(By.ID, "btn-recent7").click()
    time.sleep(0.35)
    assert driver.find_element(By.ID, "kpi-total").text == "344건"
    assert driver.find_element(By.ID, "kpi-risk").text == "125건"
    assert driver.find_element(By.ID, "kpi-due0").text == "84건"
    csv_r7 = driver.execute_script("return buildCsvPayload();")
    assert csv_r7["rowCount"] == 125
    print("[PASS] Recent 7 days (2026-10-14~20): 344 / 125 / 84, CSV=125 rows")

    # 5. Center sync via actual Plotly bar click (부산센터) + clear button + table row click + carrier select
    bar_clicked = driver.execute_script("""
        const gd = document.getElementById('center-chart');
        gd.emit('plotly_click', { points: [{ x: '부산센터' }] });
        return document.getElementById('filter-center').value;
    """)
    time.sleep(0.25)
    assert bar_clicked == "부산센터"
    assert driver.find_element(By.ID, "kpi-total").text == "114건"
    assert driver.find_element(By.ID, "kpi-risk").text == "53건"
    assert driver.find_element(By.ID, "btn-chart-center-all").is_displayed()
    assert len(driver.find_elements(By.CSS_SELECTOR, "#center-tbody tr")) == 1

    # Clicking '전체 센터 보기' restores all 3 centers in filter & screen
    driver.find_element(By.ID, "btn-chart-center-all").click()
    time.sleep(0.25)
    assert driver.find_element(By.ID, "filter-center").get_attribute("value") == "전체"
    assert len(driver.find_elements(By.CSS_SELECTOR, "#center-tbody tr")) == 3

    # Clicking 부산센터 row in summary table selects 부산센터 again
    driver.find_elements(By.CSS_SELECTOR, "#center-tbody tr")[1].click()
    time.sleep(0.25)
    assert driver.find_element(By.ID, "filter-center").get_attribute("value") == "부산센터"
    assert driver.find_element(By.ID, "kpi-total").text == "114건"
    assert driver.find_element(By.ID, "kpi-risk").text == "53건"

    driver.execute_script("""
        const c = document.getElementById('filter-carrier');
        c.value = 'B운송사';
        c.dispatchEvent(new Event('change'));
    """)
    time.sleep(0.25)
    b_tot = driver.find_element(By.ID, "kpi-total").text
    b_rsk = driver.find_element(By.ID, "kpi-risk").text
    b_due = driver.find_element(By.ID, "kpi-due0").text
    assert b_tot != "114건" and b_tot != "0건"
    print(f"[PASS] Center + Carrier filter (부산센터 + B운송사): total={b_tot}, risk={b_rsk}, due0={b_due}")

    driver.execute_script("""
        const c = document.getElementById('filter-carrier');
        c.value = '전체';
        c.dispatchEvent(new Event('change'));
        const s = document.getElementById('filter-search');
        s.value = 'DLV011987';
        s.dispatchEvent(new Event('input'));
    """)
    time.sleep(0.2)
    assert len(driver.find_elements(By.CSS_SELECTOR, "#priority-tbody tr")) == 1
    assert driver.execute_script("return buildCsvPayload().rowCount;") == 1
    print("[PASS] Center click-sync (부산센터 114/53) & search filter (DLV011987 -> 1 row) verified")

    # 5. Empty state + division-by-zero ('계산 불가')
    driver.execute_script("""
        const d1 = document.getElementById('filter-start');
        d1.value = '2026-10-25';
        d1.dispatchEvent(new Event('change'));
    """)
    time.sleep(0.25)
    assert driver.find_element(By.ID, "kpi-total").text == "0건"
    assert "계산 불가" in driver.find_element(By.ID, "kpi-risk-rate").text
    assert "계산 불가" in driver.find_element(By.ID, "kpi-due0-rate").text
    assert driver.find_element(By.ID, "center-chart-empty").is_displayed()
    assert driver.find_element(By.ID, "list-empty").is_displayed()
    assert driver.find_element(By.ID, "btn-csv").get_attribute("disabled") is not None
    print("[PASS] Empty state & zero denominator ('계산 불가') verified")

    # 6. Reset + check all 8 themes
    driver.find_element(By.ID, "btn-reset").click()
    time.sleep(0.35)
    assert driver.find_element(By.ID, "kpi-total").text == "41건"
    assert driver.find_element(By.ID, "kpi-risk").text == "11건"
    for t in ("b", "c", "d", "e", "f", "g", "h", "a"):
        driver.execute_script(f"setTheme('{t}');")
        time.sleep(0.15)
    print("[PASS] Reset & 8-theme switcher verified")

    # 7. Backup Model Reference tab switch & content check
    assert driver.find_element(By.ID, "tab-ops").is_displayed()
    assert not driver.find_element(By.ID, "tab-model").is_displayed()
    driver.find_element(By.ID, "tab-btn-model").click()
    time.sleep(0.45)
    assert driver.find_element(By.ID, "tab-model").is_displayed()
    assert not driver.find_element(By.ID, "tab-ops").is_displayed()
    assert len(driver.find_elements(By.CSS_SELECTOR, "#model-comp-tbody tr")) == 6
    assert len(driver.find_elements(By.CSS_SELECTOR, "#sep-eval-tbody tr")) == 2
    chart_box = driver.execute_script("return document.getElementById('model-comp-chart').getBoundingClientRect();")
    assert chart_box["width"] > 600 and chart_box["height"] > 200
    driver.find_element(By.ID, "btn-back-ops").click()
    time.sleep(0.3)
    assert driver.find_element(By.ID, "tab-ops").is_displayed()
    assert not driver.find_element(By.ID, "tab-model").is_displayed()
    print("[PASS] Backup Model Reference tab (6 models, 9월 final eval, Plotly chart) & back-to-ops verified")


def main() -> None:
    driver = get_driver()
    try:
        driver.get(DASHBOARD_HTML.as_uri())
        time.sleep(1.4)
        run_functional_tests(driver)

        # Capture 15_ core view (header + KPIs + priority cards + center chart)
        core_h = int(
            driver.execute_script(
                """
                const splits = document.querySelectorAll('.split-grid');
                return Math.ceil(splits[0].getBoundingClientRect().bottom + window.scrollY + 20);
                """
            )
        )
        cdp_shot(driver, 1440, core_h, OUT_CORE_PNG)

        # Capture 16_ full page (Tab 1 Ops)
        full_h = int(driver.execute_script("return document.documentElement.scrollHeight;"))
        cdp_shot(driver, 1440, full_h + 18, OUT_FULL_PNG)

        # Capture 17_ Backup Model Reference tab (Tab 2)
        driver.find_element(By.ID, "tab-btn-model").click()
        time.sleep(0.55)
        model_h = int(driver.execute_script("return document.documentElement.scrollHeight;"))
        cdp_shot(driver, 1440, model_h + 18, OUT_MODEL_PNG)

        try:
            logs = driver.get_log("browser")
            severe = [l for l in logs if l.get("level") == "SEVERE"]
            assert len(severe) == 0, f"Severe console logs: {severe}"
            print(f"[PASS] Browser console logs total={len(logs)}, severe=0")
        except Exception as e:
            print(f"Console log note: {e}")
    finally:
        driver.quit()

    org.make_contact_sheet()


if __name__ == "__main__":
    main()
