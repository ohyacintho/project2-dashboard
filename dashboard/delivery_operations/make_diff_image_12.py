"""Create annotated visual comparison image (Before vs After PDF reflection) for Session 12."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"

OUT_DIFF_PNG = GRAPHS_DIR / "09_PDF반영_이전vs이후_달라진점_비교.png"


def main() -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(18.0, 10.8), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")

    ax = fig.add_axes([0.015, 0.02, 0.97, 0.90])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fig.suptitle(
        "[PDF 반영] 이전(1~3단계) vs 이후(PDF 반영) 달라진 점 한눈에 비교",
        fontsize=17.0,
        fontweight="bold",
        color="#0F172A",
        y=0.968,
    )
    fig.text(
        0.5,
        0.928,
        "기존 5개 모델·C=1·예측값은 100% 유지하면서, PDF(숫자형 변수 스케일링 1~4p) 실증·실제 주문 5건·웹 보고서(HTML)를 추가",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#475569",
    )

    # Outer cards: Left (Before), Center (Key Changes), Right (After)
    left_card = mpatches.FancyBboxPatch(
        (1.0, 13.0), 30.5, 84.0,
        boxstyle="round,pad=0.4,rounding_size=1.4",
        facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.8,
    )
    mid_card = mpatches.FancyBboxPatch(
        (34.5, 13.0), 31.0, 84.0,
        boxstyle="round,pad=0.4,rounding_size=1.4",
        facecolor="#EFF6FF", edgecolor="#60A5FA", linewidth=2.0,
    )
    right_card = mpatches.FancyBboxPatch(
        (68.5, 13.0), 30.5, 84.0,
        boxstyle="round,pad=0.4,rounding_size=1.4",
        facecolor="#FFFFFF", edgecolor="#2563EB", linewidth=2.2,
    )
    bottom_bar = mpatches.FancyBboxPatch(
        (1.0, 2.0), 98.0, 8.2,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor="#ECFDF5", edgecolor="#34D399", linewidth=1.6,
    )
    for p in (left_card, mid_card, right_card, bottom_bar):
        ax.add_patch(p)

    # Left header
    ax.text(16.25, 93.2, "이전 (PDF 반영 전 · 버전1)", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#334155")
    ax.text(16.25, 89.2, "1~3단계: C 비교 · 가중치 비교 중심 (01 ~ 04)", ha="center", va="center", fontsize=9.6, color="#64748B")

    # Right header
    ax.text(83.75, 93.2, "이후 (PDF 반영 추가 · 버전2)", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#1D4ED8")
    ax.text(83.75, 89.2, "4단계 추가: PDF 원리 실증 + 웹 보고서 (05 ~ 09)", ha="center", va="center", fontsize=9.6, color="#2563EB")

    # Center header
    ax.text(50.0, 93.2, "핵심 달라진 점 5가지", ha="center", va="center", fontsize=13.0, fontweight="bold", color="#1E3A8A")
    ax.text(50.0, 89.2, "기존 모델 재학습 없이 실제 데이터로 검증 확장", ha="center", va="center", fontsize=9.6, color="#3B82F6")

    # Left text summary box
    left_txt = (
        "[1] 다룬 내용\n"
        "  • C(0.01~100) · 가중치(1:1~1:5) 성능 비교\n"
        "  • 표준화는 '적용했다' 사실만 확인\n\n"
        "[2] 제목 · 파일 구성\n"
        "  • 정적 PNG 4종 (01_ ~ 04_) + 시트(00_)\n"
        "  • Plotly 차트 4종 / 실습기록 버전1.md\n\n"
        "[3] 웹 보고서 (HTML)\n"
        "  • 단일 차트 HTML만 존재 (통합 보고서 없음)"
    )
    left_box = mpatches.FancyBboxPatch(
        (2.4, 15.2), 27.7, 29.0,
        boxstyle="round,pad=0.35,rounding_size=0.9",
        facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1.2,
    )
    ax.add_patch(left_box)
    ax.text(3.4, 29.7, left_txt, ha="left", va="center", fontsize=8.8, color="#334155", linespacing=1.42)

    # Right text summary box
    right_txt = (
        "[1] 다룬 내용 (PDF 1~4p 실증)\n"
        "  • 계수 단위 비교 + Z-score vs Min-Max 검증\n"
        "  • 총중량_kg Min-Max 이탈 + 실제 5주문 계산\n\n"
        "[2] 제목 · 파일 구성 (PDF 접두 통일)\n"
        "  • 05_PDF_~09_PDF_ 추가 + 3x3 시트(00_) 갱신\n"
        "  • 기준통계·5주문 CSV 2종 + 버전2.md 기록\n\n"
        "[3] 웹 보고서 (HTML) 신설\n"
        "  • PDF_숫자형_변수_스케일링_분석.html (3테마)"
    )
    right_box = mpatches.FancyBboxPatch(
        (69.9, 15.2), 27.7, 29.0,
        boxstyle="round,pad=0.35,rounding_size=0.9",
        facecolor="#F8FAFC", edgecolor="#BFDBFE", linewidth=1.2,
    )
    ax.add_patch(right_box)
    ax.text(70.9, 29.7, right_txt, ha="left", va="center", fontsize=8.8, color="#1E293B", linespacing=1.42)

    # Center 5 change cards
    changes = [
        ("1. 계수 단위 차이 해소 실증 (PDF p.1)",
         "원단위: 배송거리 +0.0042(/km) vs 부하율 +2.2042\n-> Z-score 후: 여유일수 -0.8670 · 부하율 +0.5610 · 거리 +0.4986"),
        ("2. 분포 모양 유지 & Min-Max 비교 (PDF p.2~3)",
         "히스토그램 모양은 그대로이고 눈금만 변경됨을 확인\n-> 총중량_kg에서 7월(-0.0004)·9월(-0.0007) 0~1 이탈 실측"),
        ("3. 학습 기준 재사용 & 실제 주문 5건 (PDF p.4)",
         "1~6월 저장 StandardScaler로 DLV007201~007205 변환\n-> 예: DLV007201 (371km -> +1.4163 / 1,202건 -> +0.1459)"),
        ("4. 제목에 'PDF' + 산출물명 표기 적용",
         "그래프(05_~09_PDF_...), HTML(PDF_숫자형_...), MD 제목 통일\n-> 기준통계 CSV · 실제주문 5건 CSV · 버전2.md 누적 저장"),
        ("5. 단독 실행 웹 보고서(HTML) & 브라우저 검수",
         "변수·스케일·테마 3종 실시간 전환 가능 HTML 최초 추가\n-> 헤드리스 브라우저 콘솔 오류 0건 및 캡처(07_, 08_) 완료"),
    ]
    y_top = 84.5
    card_h = 12.6
    gap = 1.6
    for i, (ctitle, cbody) in enumerate(changes):
        cy = y_top - i * (card_h + gap) - card_h
        cbox = mpatches.FancyBboxPatch(
            (36.1, cy), 27.8, card_h,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            facecolor="#FFFFFF", edgecolor="#93C5FD", linewidth=1.3,
        )
        ax.add_patch(cbox)
        ax.text(37.1, cy + card_h - 3.0, ctitle, ha="left", va="center", fontsize=9.2, fontweight="bold", color="#1E40AF")
        ax.text(37.1, cy + 4.3, cbody, ha="left", va="center", fontsize=8.2, color="#334155", linespacing=1.38)

    # Arrows connecting Left -> Center -> Right
    ax.annotate("", xy=(34.2, 66.0), xytext=(31.8, 66.0), arrowprops=dict(arrowstyle="->", color="#64748B", lw=2.4))
    ax.annotate("", xy=(68.2, 66.0), xytext=(65.8, 66.0), arrowprops=dict(arrowstyle="->", color="#2563EB", lw=2.6))

    # Bottom banner text
    ax.text(
        50.0,
        6.1,
        "그대로 유지된 항목: 기존 5개 모델('로지스틱 기본' 포함) · C=1 고정 기준 · delivery_predictions_12.csv (2,400행 x 51열) 예측 결과는 재학습/수정 없이 100% 동일 유지",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color="#065F46",
    )

    # Embed 4 actual thumbnail previews inside Left & Right cards
    # Left thumbnails: 01_ and 04_
    ax_l1 = fig.add_axes([0.036, 0.455, 0.268, 0.315])
    ax_l1.imshow(mpimg.imread(GRAPHS_DIR / "01_로지스틱_C별_평가성능_지표_비교.png"))
    ax_l1.set_title("기존 예시: 01. C별 성능 · 04. 가중치 물량", fontsize=8.6, fontweight="bold", color="#475569", pad=4)
    ax_l1.set_xticks([]); ax_l1.set_yticks([])
    for sp in ax_l1.spines.values():
        sp.set_edgecolor("#CBD5E1"); sp.set_linewidth(1.0)

    # Right thumbnails: 05_ (left half) and 07_ (right half) stacked neatly or 05_ + 07_ side-by-side
    ax_r1 = fig.add_axes([0.692, 0.595, 0.132, 0.165])
    ax_r1.imshow(mpimg.imread(GRAPHS_DIR / "05_PDF_숫자형_변수_스케일링_원리_및_분포비교.png"))
    ax_r1.set_title("05. [PDF] 원리·분포", fontsize=8.1, fontweight="bold", color="#1D4ED8", pad=3)
    ax_r1.set_xticks([]); ax_r1.set_yticks([])
    for sp in ax_r1.spines.values():
        sp.set_edgecolor("#93C5FD"); sp.set_linewidth(1.1)

    ax_r2 = fig.add_axes([0.832, 0.595, 0.132, 0.165])
    ax_r2.imshow(mpimg.imread(GRAPHS_DIR / "06_PDF_로지스틱_표준화_실제주문5건_계산과정.png"))
    ax_r2.set_title("06. [PDF] 5주문 계산", fontsize=8.1, fontweight="bold", color="#1D4ED8", pad=3)
    ax_r2.set_xticks([]); ax_r2.set_yticks([])
    for sp in ax_r2.spines.values():
        sp.set_edgecolor("#93C5FD"); sp.set_linewidth(1.1)

    ax_r3 = fig.add_axes([0.692, 0.435, 0.272, 0.135])
    ax_r3.imshow(mpimg.imread(GRAPHS_DIR / "07_PDF_스케일링_웹보고서_제출용_캡처.png"))
    ax_r3.set_title("07. [PDF] 단독 실행 웹 보고서 (HTML)", fontsize=8.2, fontweight="bold", color="#1D4ED8", pad=3)
    ax_r3.set_xticks([]); ax_r3.set_yticks([])
    for sp in ax_r3.spines.values():
        sp.set_edgecolor("#2563EB"); sp.set_linewidth(1.2)

    fig.savefig(OUT_DIFF_PNG, dpi=180, facecolor="#F8FAFC")
    plt.close(fig)
    print(f"Saved: {OUT_DIFF_PNG}")


if __name__ == "__main__":
    main()
