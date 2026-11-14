# Project 2 (04회차) 입고검사 품질 대시보드 GitHub 연동 및 Cloudflare 배포 실습 정리 보고서

* **작성 일자:** 2026년 11월 14일 (2026-11-14)
* **실습 담당자:** 품질관리/구매 담당자 (GitHub: `ohyacintho`)
* **프로젝트 저장소:** [https://github.com/ohyacintho/project2-dashboard](https://github.com/ohyacintho/project2-dashboard)
* **로컬 Git 루트 경로:** `c:\Users\최지영\AI2\project2-dashboard\`
* **로컬 웹 서버 주소:** `http://localhost:8080/`

---

## 1. 실습 배경 및 핵심 목표

본 실습은 **02회차(탐색적 데이터 분석, EDA)**와 **03회차(협력사 재계약 종합 검토)**에서 개별적으로 제작한 HTML 품질 대시보드를 단일 포털 사이트로 통합하고, 로컬 PC 환경을 벗어나 **전 세계 누구나 웹 브라우저에서 대시보드를 직접 조작할 수 있도록 클라우드 웹 호스팅(Cloudflare Pages) 환경을 구축**하는 것을 목표로 합니다.

### 💡 서비스 역할 및 개념 비교 (초보자 가이드)

| 구분 | 역할 | 비유 | 본 실습에서의 구체적 동작 |
| :--- | :--- | :--- | :--- |
| **내 컴퓨터 (`localhost:8080`)** | 작업실 | 내 방 책상 위의 인쇄물 | 로컬에서 대시보드 기능 및 필터 동작을 사전 검증 |
| **깃허브 (GitHub)** | 안전한 보관함 | 클라우드 금고 / 소스코드 창고 | `main` 브랜치에 배포 파일(HTML, CSS, 아이콘) 백업 및 버전 관리 |
| **클라우드플레어 (Cloudflare)** | 인터넷 전광판 | 전 세계에 공개된 웹사이트 | 깃허브의 파일을 감지하여 무료 정적 호스팅 주소(`*.pages.dev`) 자동 발급 |

> **최종 목적:** 사내 동료, 의사결정권자, 외부 평가자에게 카카오톡이나 이메일로 **URL 하나만 전달하면, 별도 설치 없이 스마트폰이나 PC 브라우저에서 대시보드를 열어보고 필터와 차트를 직접 체험**할 수 있도록 구현함.

---

## 2. 02회차 & 03회차 HTML 대시보드 데이터 내장 여부 정밀 점검

클라우드 배포 시 파일 경로 깨짐이나 데이터 비동기 호출(CORS/404) 문제를 방지하기 위해 기존 원본 HTML의 구조를 정밀 점검했습니다.

| 점검 항목 | 02회차 (`inspection_dashboard.html`) | 03회차 (`supplier_review.html`) |
| :--- | :--- | :--- |
| **데이터 내장 여부** | **100% 완전 내장 (Self-contained)** | **100% 완전 내장 (Self-contained)** |
| **데이터 내장 방식** | `<script>` 내 `const RAW_DATA = [...]`<br>(JSON 데이터 약 260 KB) 직접 포함 | `<script id="inspection-data" type="application/json">`<br>(6,000건 전수 검사 레코드 약 1.47 MB) 완전 내장 |
| **시각화 차트 방식** | 순수 HTML / CSS / Vanilla JS 렌더링 | 동적 SVG 벡터 차트 + 추가실습 차트 4종 **Base64 Data URI(`data:image/png;base64,...`)** 100% 내장 |
| **외부 참조 파일** | **없음 (0개)** | **없음 (0개)** (툴바 다운로드 버튼용 PNG 2장 동봉) |
| **배포 호환성** | 외부 서버 의존성이 전혀 없어 **즉시 배포 가능** | 폴더 경로에 구애받지 않고 **오프라인/클라우드 완벽 구동** |

---

## 3. 배포 프로젝트 구조 (`project2-dashboard`) 구축

원본 파일(`2회/results/` 및 `3회/results/`)을 안전하게 보존한 상태에서, Cloudflare Pages 표준 배포 규격에 맞추어 `project2-dashboard/dashboard/` 디렉터리 구조를 완성했습니다.

```text
c:\Users\최지영\AI2\project2-dashboard\
└── dashboard/                                  ← Cloudflare Pages 빌드 출력 폴더 (Root)
    ├── favicon.ico                             # 브라우저 기본 파비콘 (404 방지용 16x16 ICO)
    ├── index.html                              # 통합 대시보드 포털 메인 화면 (신규 제작)
    ├── inspection-eda/
    │   └── inspection_dashboard.html           # 02회차 복사본 (339 KB)
    └── supplier-review/
        ├── supplier_review.html                # 03회차 복사본 (4.63 MB, 5개 탭 완비)
        ├── 01_기본품질현황_대시보드.png           # (툴바 캡처 다운로드 연동용)
        └── 02_협력사재계약검토_대시보드.png         # (툴바 캡처 다운로드 연동용)
```

### 🌟 신규 제작된 `index.html` (통합 대시보드 포털) 주요 특징
1. **모던 반응형 카드 레이아웃:** 02회차 대시보드와 03회차 대시보드를 한눈에 조망할 수 있는 대시보드 카드 제공
2. **품질 KPI 요약 배너:** 분석 대상(협력사 A·B·C), 전수 검사 수량(6,000건), 종합 불량률(4.95%) 상단 표시
3. **독립적 상대경로 링크:** `./inspection-eda/inspection_dashboard.html` 및 `./supplier-review/supplier_review.html`로 연결되어 로컬/웹 환경 모두 완벽 작동

---

## 4. 로컬 웹 서버 실행 및 브라우저 자동화 검증 (QA)

배포 전 실제 웹 서버 환경에서의 오작동 유무를 검증하기 위해 Python 로컬 웹 서버를 구동하고, **Selenium Headless Chrome을 활용한 전수 자동화 브라우저 인터랙션 테스트**를 수행했습니다.

* **로컬 웹 서버 구동 커맨드:** `python -m http.server 8080 --directory project2-dashboard/dashboard`
* **접속 테스트 주소:** `http://localhost:8080/`

### 📋 상세 자동화 QA 검증 결과

1. **포털 메인 화면 (`/`) 검증:**
   * 대시보드 카드 2개 및 상단 요약 지표 정상 로딩 확인 (HTTP 200 OK)
2. **02회차 대시보드 (`/inspection-eda/inspection_dashboard.html`) 인터랙티브 필터 검증:**
   * **초기 상태:** 전체 검사수량 6,000건, 불량수량 278건 (4.67%)
   * **협력사 필터 'A' 선택:** 검사수량 2,600건, 불량수량 220건 (8.51%) 실시간 카드 및 표 갱신 확인
   * **주차 필터 '1주차' 선택:** 검사수량 100건, 불량수량 6건 실시간 갱신 확인
   * **필터 초기화:** 6,000건 정상 복귀 확인
3. **03회차 대시보드 (`/supplier-review/supplier_review.html`) 5개 탭 및 차트 검증:**
   * **탭 1 (기본 품질 현황):** 협력사 드롭다운 'A' 선택 시 2,600건 실시간 통계표 갱신
   * **탭 2 (01 협력사별 품질 수준):** 협력사 'A' 버튼 클릭 시 2,600건 통계 및 반응형 SVG 히스토그램/박스플롯 단독 렌더링 확인, '초기화' 버튼 클릭 시 6,000건 복원 확인
   * **탭 3 (02 A사 집중 점검 분석):** '1주차' 버튼 클릭 시 불량수량/불량률 SVG 히트맵 실시간 갱신 확인
   * **탭 4 (03 협력사 검토 의견):** 재계약 종합 의견표 4개 행 정상 렌더링 확인
   * **탭 5 (04 추가 실습):** 중량 품질 비교 차트, 치수-중량 2차원 산점도 등 4개 Base64 내장 이미지 정상 렌더링 확인
4. **브라우저 콘솔 오류 점검:**
   * 파비콘(`favicon.ico`) 추가 조치 후 **콘솔 치명적 에러 0건 (Zero Severe Errors)** 달성

---

## 5. GitHub 원격 저장소 연동 및 푸시 내역

검증이 완료된 배포용 파일 전체를 지정된 GitHub 원격 저장소로 커밋 및 푸시 완료했습니다.

* **GitHub 저장소:** [https://github.com/ohyacintho/project2-dashboard](https://github.com/ohyacintho/project2-dashboard)
* **푸시 대상 브랜치:** `main`
* **커밋 ID (SHA):** **`9b856ea`** *(Full SHA: `9b856ea06001004e1fe9e7e64bc6d3957ff533e4`)*
* **커밋 메시지:** `feat: deploy quality inspection dashboards (02 & 03 sessions)`
* **커밋된 파일 목록 (총 6개 파일):**
  1. `dashboard/favicon.ico`
  2. `dashboard/index.html`
  3. `dashboard/inspection-eda/inspection_dashboard.html`
  4. `dashboard/supplier-review/supplier_review.html`
  5. `dashboard/supplier-review/01_기본품질현황_대시보드.png`
  6. `dashboard/supplier-review/02_협력사재계약검토_대시보드.png`

---

## 6. Cloudflare Pages 배포 가이드 및 제출 주소 안내

### 📌 타인 공유 및 과제 제출 시 주소 구분 안내

1. **코드 및 과제 검사용 주소 (GitHub Repository):**
   * **제출 주소:** `https://github.com/ohyacintho/project2-dashboard`
   * **상대방이 보는 내용:** 소스코드 폴더 구조(`dashboard/`), 커밋 기록, HTML 원본 코드
2. **실제 대시보드 웹 체험용 주소 (Cloudflare Pages):**
   * **배포 예정 주소 형태:** `https://project2-dashboard.pages.dev` (또는 지정된 고유 도메인)
   * **상대방이 보는 내용:** 브라우저에서 마우스로 필터를 누르고 차트를 인터랙티브하게 체험할 수 있는 실제 웹사이트

### 🚀 Cloudflare Pages 최종 배포 4단계

1. **[Cloudflare 대시보드](https://dash.cloudflare.com/) 로그인**
2. **`Workers 및 Pages`** → **`Create application (애플리케이션 생성)`** 클릭
3. 상단 탭에서 **`Pages`** 탭을 선택한 뒤 **`Connect to Git (Git에 연결)`** 클릭
4. **`ohyacintho/project2-dashboard`** 저장소 선택 후 빌드 설정 입력:
   * **Production branch:** `main`
   * **Framework preset:** `None`
   * **Build command:** *(공란)*
   * **Build output directory:** **`dashboard`** *(★ 가장 중요)*
5. **`Save and Deploy`** 클릭 시 10~20초 후 전 세계에 공개된 `https://*.pages.dev` 주소가 즉시 활성화됩니다.

---

## 7. [04회차 신규 분석] 생산조건과 품질의 상관관계 분석 (진행 현황)

### 7.1 분석 목적 및 데이터셋 개요 (0단계 요약)
* **분석 목적:** 협력사(A·B·C)에 추가로 확인하고 관리할 생산조건(가공시간, 공구 마모도, 설비 차이 등)을 규명하기 위함.
* **데이터셋:** `inspection_correlation.csv` (총 6,000건 전수, 20개 컬럼, 결측치 0건)
* **분석 기간:** 2026-09-01 ~ 2026-10-20 (8주간)
* **신규 추가된 생산조건 변수:**
  1. `공구교체후누적가공수`: 칼날 교체 후 몇 번째 부품인지 (공구 마모도 연계 수치형 핵심 변수)
  2. `설비번호`: 부품을 가공한 기계 식별자 (`A-L1-M1` 등 분류형 변수)
  3. `생산일시`, `생산배치번호`, `공구장착회차번호`, `원자재로트번호`

---

### 7.2 가공시간(초) vs 측정치수(mm) 산점도 및 퍼짐도 분석 (1단계)

![가공시간 vs 측정치수 산점도](file:///C:/Users/최지영/.gemini/antigravity/brain/83c7ede4-f3f1-4a32-800b-0ea0d118ab33/processing_time_vs_dimension_scatter.png)

* **저장 파일:** [processing_time_vs_dimension_scatter.png](file:///c:/Users/최지영/AI2/4회/results/graphs/processing_time_vs_dimension_scatter.png)
* **산출 지표 통계 요약:**
  * **상관계수 ($r$):** **`+0.692`** (뚜렷한 양의 상관관계)
  * **결정계수 ($R^2$):** **`47.9%`** (치수 변동의 약 48%가 가공시간으로 설명됨)
  * **선형 회귀식:** $y = 0.0057x + 9.771$ (가공시간 10초 증가 시 부품 치수 평균 +0.057mm 증가)
  * **잔차 표준편차 (직선 기준 평균 퍼짐도):** **`±0.031 mm`**
  * **95% 산포 범위:** 직선 중심 **`±0.061 mm`** 폭의 띠(Band) 형성

#### 가공시간 구간별 치수 및 퍼짐(산포) 비교표

| 가공시간 구간 | 부품 수량 | 치수 평균 | 직선 기준 퍼짐 정도 | 품질 진단 결과 |
| :---: | :---: | :---: | :---: | :--- |
| **25초 ~ 35초 (단시간)** | 201건 | 9.967 mm | $\pm 0.042$ mm | 목표치(10mm)보다 작게 깎이며 규격 하한선(9.9mm)에 근접 |
| **35초 ~ 45초 (적정시간)** | **3,315건 (55%)** | **10.004 mm** | **$\pm 0.030$ mm** | **목표값(10.000 mm)과 정확히 일치하며 퍼짐도 가장 적고 안정적** |
| **45초 ~ 55초 (지연시간)** | 2,380건 (40%) | 10.051 mm | $\pm 0.037$ mm | 치수가 10.05mm 이상으로 비대해지며 상향 편향 발생 |
| **55초 ~ 65초 (장시간)** | 104건 | 10.097 mm | $\pm 0.029$ mm | 규격 상한선(10.100 mm)을 위험하게 초과하여 불량 다발 |

#### 품질 담당자 핵심 결론 및 다음 단계
1. **가공시간 제어의 중요성:** 부품 치수는 가공시간과 강하게 연동되어 있으며, **`38초 ~ 42초`** 안팎에서 목표 치수 달성률이 극대화됨.
2. **잔여 산포(52%) 원인 규명 필요:** 직선 주변으로 퍼져 있는 52%의 잔여 산포를 설명하기 위해, 다음 단계인 **네 변수(가공시간, 치수, 중량, 공구누적가공수) 상관관계 히트맵 분석**으로 연계 필요.
