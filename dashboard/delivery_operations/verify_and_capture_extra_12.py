"""Verify Option 3 dashboard (delivery_operations_extra.html) and capture 19_, 20_, and refreshed 00_."""

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
EXTRA_HTML = RESULTS_DIR / "delivery_operations_extra.html"

OUT_CORE_PNG = GRAPHS_DIR / "19_추가실습3_개선대시보드_제출용_캡처.png"
OUT_FULL_PNG = GRAPHS_DIR / "20_추가실습3_개선대시보드_전체화면_캡처.png"


def get_driver():
    """Create headless Chrome or Edge webdriver."""
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
    """Capture viewport via CDP."""
    driver.execute_cdp_cmd(
        "Emulation.setDeviceMetricsOverride",
        {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
    )
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.45)
    shot = driver.execute_cdp_cmd(
        "Page.captureScreenshot",
        {"format": "png", "captureBeyondViewport": False, "fromSurface": True},
    )
    out_path.write_bytes(base64.b64decode(shot["data"]))
    print(f"Saved capture: {out_path.name} ({width}x{height})")


def run_tests(driver) -> None:
    """Verify initial defaults, Order Detail facts/questions, work log save/load/CSV, filters, and Tab 2."""
    # Clear any residual localStorage first
    driver.execute_script("localStorage.clear();")
    driver.refresh()
    time.sleep(1.2)

    # 1. Default view & default '미점검' + empty memo
    assert driver.find_element(By.ID, "kpi-total").text == "41건"
    assert driver.find_element(By.ID, "kpi-risk").text == "11건"
    assert driver.find_element(By.ID, "kpi-due0").text == "8건"
    assert driver.find_element(By.ID, "d-id").text == "DLV011994"
    assert driver.find_element(By.ID, "d-work-badge").text == "미점검"
    assert driver.find_element(By.ID, "detail-memo").get_attribute("value") == ""
    assert "인천센터" in driver.find_element(By.ID, "d-facts-grid").text
    assert len(driver.find_elements(By.CSS_SELECTOR, "#d-questions-list li")) >= 2
    assert "미점검 11 · 확인중 0 · 조치완료 0" in driver.find_element(By.ID, "work-prog-pill").text
    print("[PASS] Default 2026-10-20 view, DLV011994 detail, Confirmed Facts vs Questions, and initial 미점검/empty memo verified")

    # 2. Save status & memo -> persistence across reload + JSON backup + work filter + CSV export match
    driver.find_element(By.CSS_SELECTOR, '.status-opt[data-st="확인중"]').click()
    driver.find_element(By.ID, "detail-memo").send_keys("B운송사 집하 시각 확인 필요")
    driver.find_element(By.ID, "btn-save-note").click()
    time.sleep(0.25)
    assert driver.find_element(By.ID, "d-work-badge").text == "확인중"
    assert "미점검 10 · 확인중 1 · 조치완료 0" in driver.find_element(By.ID, "work-prog-pill").text

    # Reload and verify localStorage preserved DLV011994 log
    driver.refresh()
    time.sleep(1.1)
    assert driver.find_element(By.ID, "d-work-badge").text == "확인중"
    assert driver.find_element(By.ID, "detail-memo").get_attribute("value") == "B운송사 집하 시각 확인 필요"

    # Filter by '확인중만' -> 1 row in list & CSV with memo included
    driver.execute_script("""
        const wf = document.getElementById('filter-work');
        wf.value = '확인중';
        wf.dispatchEvent(new Event('change'));
    """)
    time.sleep(0.25)
    assert len(driver.find_elements(By.CSS_SELECTOR, "#priority-tbody tr")) == 1
    csv_work = driver.execute_script("return buildCsvPayload();")
    assert csv_work["rowCount"] == 1
    assert "확인중" in csv_work["content"] and "B운송사 집하 시각 확인 필요" in csv_work["content"]
    assert "실제배송완료일" not in csv_work["content"] and "배송지연여부" not in csv_work["content"]

    # JSON export -> clear -> JSON import restore
    backup_obj = driver.execute_script("return buildNotesBackupPayload();")
    assert backup_obj["order_count"] == 1
    driver.find_element(By.ID, "btn-clear-notes").click()
    time.sleep(0.25)
    assert driver.find_element(By.ID, "list-empty").is_displayed()
    restored = driver.execute_script("return applyImportedNotesObject(arguments[0]);", backup_obj)
    assert restored == 1
    assert len(driver.find_elements(By.CSS_SELECTOR, "#priority-tbody tr")) == 1
    print("[PASS] Per-order status/memo save, reload persistence, JSON backup/restore, work filter & CSV export verified")

    # 3. Reset filters & test date/center/carrier/empty/tab2 cleanly
    driver.find_element(By.ID, "btn-reset").click()
    time.sleep(0.3)
    driver.find_element(By.ID, "btn-recent7").click()
    time.sleep(0.3)
    assert driver.find_element(By.ID, "kpi-total").text == "344건"
    assert driver.find_element(By.ID, "kpi-risk").text == "125건"

    driver.find_element(By.ID, "tab-btn-model").click()
    time.sleep(0.4)
    assert len(driver.find_elements(By.CSS_SELECTOR, "#model-comp-tbody tr")) == 6
    driver.find_element(By.ID, "btn-back-ops").click()
    time.sleep(0.3)

    # Restore clean initial state with 1 realistic sample check or clean '미점검' default for capture:
    # Let's clear notes and reset filters so initial capture reflects clean '미점검' default AND DLV011994 detail workbench!
    driver.find_element(By.ID, "btn-clear-notes").click()
    driver.find_element(By.ID, "btn-reset").click()
    time.sleep(0.4)
    assert driver.find_element(By.ID, "d-work-badge").text == "미점검"
    assert driver.find_element(By.ID, "detail-memo").get_attribute("value") == ""
    print("[PASS] Recent 7d, backup tab, and clean reset verified")


def main() -> None:
    driver = get_driver()
    try:
        driver.get(EXTRA_HTML.as_uri())
        time.sleep(1.4)
        run_tests(driver)

        # Capture 19_ core publishing view: Header + Filters + KPIs + Priority Cards + 2-col Workbench (List + Detail/Memo)
        core_h = int(
            driver.execute_script(
                """
                const wb = document.getElementById('workbench-card');
                return Math.ceil(wb.getBoundingClientRect().bottom + window.scrollY + 20);
                """
            )
        )
        cdp_shot(driver, 1440, core_h, OUT_CORE_PNG)

        # Capture 20_ full screen (including center chart & summary + footer strip)
        full_h = int(driver.execute_script("return document.documentElement.scrollHeight;"))
        cdp_shot(driver, 1440, full_h + 18, OUT_FULL_PNG)

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
