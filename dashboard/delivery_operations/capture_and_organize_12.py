"""Capture HTML report screenshots, verify browser behavior, and refresh 00_전체_결과물_한눈에보기.png."""

import base64
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"

REPORT_HTML = RESULTS_DIR / "PDF_숫자형_변수_스케일링_분석.html"
OUT_CORE_PNG = GRAPHS_DIR / "07_PDF_스케일링_웹보고서_제출용_캡처.png"
OUT_FULL_PNG = GRAPHS_DIR / "08_PDF_스케일링_웹보고서_전체화면_캡처.png"
OUT_SHEET = GRAPHS_DIR / "00_전체_결과물_한눈에보기.png"

SHEET_ITEMS = [
    ("01_로지스틱_C별_평가성능_지표_비교.png", "01. 로지스틱 C별 학습·평가 지표"),
    ("02_로지스틱_C별_혼동행렬_비교.png", "02. 로지스틱 C별 혼동행렬·추이"),
    ("03_로지스틱_가중치별_평가성능_지표_비교.png", "03. 가중치별(1:1~1:5) 성능 지표"),
    ("04_로지스틱_가중치별_혼동행렬_및_물량_비교.png", "04. 가중치별 혼동행렬·물량"),
    ("05_PDF_숫자형_변수_스케일링_원리_및_분포비교.png", "05. [PDF] 스케일링 원리·분포 비교"),
    ("06_PDF_로지스틱_표준화_실제주문5건_계산과정.png", "06. [PDF] 실제 주문 5건 계산 과정"),
    ("07_PDF_스케일링_웹보고서_제출용_캡처.png", "07. [PDF] 웹보고서 핵심 영역 캡처"),
    ("08_PDF_스케일링_웹보고서_전체화면_캡처.png", "08. [PDF] 웹보고서 전체 화면 캡처"),
    ("09_PDF반영_이전vs이후_달라진점_비교.png", "09. [PDF] 이전 vs 이후 달라진 점 비교"),
    ("10_6개모델_평가성능_지표_비교.png", "10. 6개 모델 성능 지표 비교"),
    ("11_6개모델_혼동행렬_비교.png", "11. 6개 모델 혼동행렬 비교"),
    ("12_6개모델_ROC곡선_및_AUC_비교.png", "12. 6개 모델 ROC 곡선·AUC 비교"),
    ("13_선정모델_가중치RF_7~8월vs9월_최종평가_비교.png", "13. 선정모델 7~8월 vs 9월 최종평가"),
    ("14_10월_운영예측_일별_및_센터별_위험분포.png", "14. 10월 운영예측 일별·센터 분포"),
    ("15_운영대시보드_제출용_캡처.png", "15. 배송 운영 대시보드 핵심 캡처"),
    ("16_운영대시보드_전체화면_캡처.png", "16. 배송 운영 대시보드 전체화면 캡처"),
    ("17_운영대시보드_모델참고탭_캡처.png", "17. 배송 운영 대시보드 모델기준 백업탭"),
    ("18_운영대시보드_Slack공유용_1순위점검_캡처.png", "18. Slack 공유용(운영+1순위 목록)"),
    ("19_추가실습3_개선대시보드_제출용_캡처.png", "19. [선택3] 개선 워크벤치 핵심 캡처"),
    ("20_추가실습3_개선대시보드_전체화면_캡처.png", "20. [선택3] 개선 워크벤치 전체화면"),
]


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


def cdp_capture(driver, width: int, height: int, out_path: Path) -> None:
    """Resize viewport via CDP and save screenshot cleanly."""
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
    with Image.open(out_path) as im:
        print(f"Saved: {out_path.name} ({im.size[0]}x{im.size[1]} px)")


def capture_and_verify_html() -> None:
    """Test interactive controls and save core + full-page captures."""
    driver = get_driver()
    try:
        driver.get(REPORT_HTML.as_uri())
        time.sleep(1.2)

        n_plots = driver.execute_script("return document.querySelectorAll('.js-plotly-plot').length;")
        n_orders = driver.execute_script("return document.querySelectorAll('#orders-tbody tr').length;")
        n_stats = driver.execute_script("return document.querySelectorAll('#stats-tbody tr').length;")
        assert n_plots == 3, f"Expected 3 Plotly charts, got {n_plots}"
        assert n_orders == 5, f"Expected 5 sample orders, got {n_orders}"
        assert n_stats == 8, f"Expected 8 numeric stats, got {n_stats}"

        # Interactive toggle checks
        driver.execute_script("setVar('wait'); setScale('mm');")
        time.sleep(0.25)
        driver.execute_script("setTheme('d');")
        time.sleep(0.2)
        driver.execute_script("setTheme('e');")
        time.sleep(0.2)
        driver.execute_script("setTheme('a'); setVar('dist'); setScale('z');")
        time.sleep(0.35)

        try:
            logs = driver.get_log("browser")
            severe = [l for l in logs if l.get("level") == "SEVERE"]
            assert len(severe) == 0, f"Browser SEVERE logs: {severe}"
            print(f"[PASS] Browser console logs total={len(logs)}, severe=0")
        except Exception as e:
            print(f"Console log note: {e}")

        # Core capture: header + 4 PDF cards + distribution comparison section
        core_h = int(
            driver.execute_script(
                """
                const cards = document.querySelectorAll('.shell > section.card');
                return Math.ceil(cards[0].getBoundingClientRect().bottom + window.scrollY + 20);
                """
            )
        )
        cdp_capture(driver, 1440, core_h, OUT_CORE_PNG)

        # Full page capture
        full_h = int(driver.execute_script("return document.documentElement.scrollHeight;"))
        cdp_capture(driver, 1440, full_h + 16, OUT_FULL_PNG)
    finally:
        driver.quit()


def make_contact_sheet() -> None:
    """Create 00_전체_결과물_한눈에보기.png covering 01~20 in a 5x4 grid."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(5, 4, figsize=(25.5, 27.0))
    axes_flat = axes.flatten()
    for ax, (fname, caption) in zip(axes_flat, SHEET_ITEMS):
        fpath = GRAPHS_DIR / fname
        if fpath.exists():
            img = mpimg.imread(fpath)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "미생성", ha="center", va="center", fontsize=11, color="#94A3B8")
        ax.set_title(caption, fontsize=10.5, fontweight="bold", pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#CBD5E1")
            sp.set_linewidth(1.2)
    for j in range(len(SHEET_ITEMS), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("[2026-11-28] 12회차 배송 지연 예측 전체 결과물 한눈에 보기 (01 ~ 20)", fontsize=15.5, fontweight="bold", y=0.989)
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.968], h_pad=2.2, w_pad=1.5)
    fig.savefig(OUT_SHEET, dpi=170)
    plt.close(fig)
    print(f"Saved contact sheet: {OUT_SHEET}")


def main() -> None:
    capture_and_verify_html()
    make_contact_sheet()


if __name__ == "__main__":
    main()
