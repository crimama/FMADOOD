# Main Figure 전체 복원 목록

기존 6개 묶음에서 **11개 묶음**으로 확장했습니다. 추가로 대화 기록에 저장된 **서로 다른 이미지 26개**를 확보했습니다. 26개에는 같은 구성의 크기·압축 차이 및 문구·간격 수정도 포함되므로 26개의 독립 레이아웃이라는 뜻은 아닙니다.

[전체 비교 갤러리](gallery.html) · [전체 이미지 비교판](contact_sheet.jpg) · [출처 및 SHA256](manifest.json)

## 원본·생성 코드·재출력

| 자료 | 복구 방식 | 보기 |
| :--- | :--- | :--- |
| 초기 Method Overview | Git a73c289 원본 | [method_overview.svg](01_early_overview/method_overview.svg) |
| 초기 Method Pipeline | Git a73c289 원본 | [method_pipeline_overview.png](02_early_pipeline/method_pipeline_overview.png) |
| RSG4 · Outline v10 | 현재 보존 원본 | [fig_rsg4_method_main.png](03_rsg4_v10/fig_rsg4_method_main.png) |
| SCARF v17 · Architecture | v18 변경 전 보관 원본 | [fig_scarf_overview.png](04_scarf_v17/fig_scarf_overview.png) |
| ScopeAD v18 · Main | 현재 main figure 보존 사본 | [fig_scopead_overview.png](05_scopead_v18/fig_scopead_overview.png) |
| ScopeAD v18 · Slide Variant | 발표자료 폴더 보존 원본 | [overview_scopead.png](06_scopead_v18_slides/overview_scopead.png) |
| 초기 SCARF · 코드 수정 이력 6개 | 대화 기록에서 소스 복원 후 캐시로 재출력 | [draft_06/fig_scarf_overview_reconstructed.png](07_scarf_initial_20261001/draft_06/fig_scarf_overview_reconstructed.png) |
| SCARF v16.3 · 발표용 crop | PPTX 내 원본 이미지 바이트 복원 | [overview_crop.png](08_scarf_v16_3_slide_crop/overview_crop.png) |
| SCARF · 원고 TikZ 수평 파이프라인 | TeX 원본 보존; 미렌더링 | [TikZ source](09_scarf_tikz/scarf_core.tex) |
| SCARF v17.1 · 슬라이드 전체 흐름 | 기존 slide render 2종과 편집 가능한 PPTX 보존 | [overview_slide_02.png](10_scarf_v17_1_slides/overview_slide_02.png) |
| ScopeAD v18 · 피팅/추론 2행 | 기존 발표용 PNG와 생성 코드 원본 보존 | [two_stage.png](11_scopead_two_stage/two_stage.png) |

초기 SCARF 6개 코드 상태는 `07_scarf_initial_20261001/draft_01`부터 `draft_06`까지 각각 보존했습니다. 원래 소스는 `.original.py`, 다시 출력한 그림은 `*_reconstructed.{png,svg,pdf}`입니다. 캐시 및 출력 경로만 바꿔 실행했고 추론·실험은 새로 수행하지 않았습니다. 당시 원본 이미지 바이트와 동일하다고 주장하지 않습니다.

## 대화 기록 이미지 26개

| 번호 | 버전 / 기록 | 이미지 크기 | 보기 |
| :--- | :--- | :--- | :--- |
| 01 | 초기 Pipeline 기록 | 1800 × 1000 | [01_method_pipeline_overview.png](session_previews/01_method_pipeline_overview.png) |
| 02 | RSG4 원본 크기 기록 | 3696 × 1419 | [02_fig_rsg4_method_main.png](session_previews/02_fig_rsg4_method_main.png) |
| 03 | RSG4 2048 px 기록 | 2048 × 786 | [03_fig_rsg4_method_main.png](session_previews/03_fig_rsg4_method_main.png) |
| 04 | RSG4 2000 px 기록 | 2000 × 768 | [04_fig_rsg4_method_main.png](session_previews/04_fig_rsg4_method_main.png) |
| 05 | 초기 SCARF 첫 출력 | 2000 × 900 | [05_fig_scarf_overview.jpg](session_previews/05_fig_scarf_overview.jpg) |
| 06 | 초기 SCARF gate 수정 | 2000 × 900 | [06_fig_scarf_overview.jpg](session_previews/06_fig_scarf_overview.jpg) |
| 07 | 초기 SCARF annotation 수정 | 2000 × 900 | [07_fig_scarf_overview.jpg](session_previews/07_fig_scarf_overview.jpg) |
| 08 | 초기 SCARF 최종 | 2000 × 900 | [08_fig_scarf_overview.jpg](session_previews/08_fig_scarf_overview.jpg) |
| 09 | 초기 SCARF 후속 기록 | 2048 × 922 | [09_fig_scarf_overview.png](session_previews/09_fig_scarf_overview.png) |
| 10 | v17 정렬 초기안 | 2048 × 936 | [10_fig_scarf_overview.png](session_previews/10_fig_scarf_overview.png) |
| 11 | v17 Multi-scale 1-NN 명시 | 2048 × 936 | [11_fig_scarf_overview.png](session_previews/11_fig_scarf_overview.png) |
| 12 | v17 bank·arrow 수정 | 2048 × 936 | [12_fig_scarf_overview.png](session_previews/12_fig_scarf_overview.png) |
| 13 | v17 세부 문구 수정 | 2048 × 936 | [13_fig_scarf_overview.png](session_previews/13_fig_scarf_overview.png) |
| 14 | v17 1-NN 박스 간격 수정 | 2048 × 936 | [14_fig_scarf_overview.png](session_previews/14_fig_scarf_overview.png) |
| 15 | v17 query 위치 수정 | 2048 × 936 | [15_fig_scarf_overview.png](session_previews/15_fig_scarf_overview.png) |
| 16 | v17 compact · CQ 제거 | 2048 × 936 | [16_fig_scarf_overview.png](session_previews/16_fig_scarf_overview.png) |
| 17 | v17 detailed 초기 | 2048 × 1009 | [17_fig_scarf_overview_detailed.png](session_previews/17_fig_scarf_overview_detailed.png) |
| 18 | v17 detailed support 수정 | 2048 × 1009 | [18_fig_scarf_overview_detailed.png](session_previews/18_fig_scarf_overview_detailed.png) |
| 19 | v17 detailed legend 수정 | 2048 × 1009 | [19_fig_scarf_overview_detailed.png](session_previews/19_fig_scarf_overview_detailed.png) |
| 20 | v17 detailed 최종 | 2048 × 1009 | [20_fig_scarf_overview_detailed.png](session_previews/20_fig_scarf_overview_detailed.png) |
| 21 | v17 architecture 초기 | 2048 × 1170 | [21_fig_scarf_overview.png](session_previews/21_fig_scarf_overview.png) |
| 22 | v17 architecture 수정 | 2048 × 1170 | [22_fig_scarf_overview.png](session_previews/22_fig_scarf_overview.png) |
| 23 | v17 architecture JPEG 기록 | 2000 × 1143 | [23_fig_scarf_overview.jpg](session_previews/23_fig_scarf_overview.jpg) |
| 24 | v18 발표용 기록 | 2000 × 1143 | [24_overview_scopead.jpg](session_previews/24_overview_scopead.jpg) |
| 25 | v18 main PNG 기록 | 2048 × 1170 | [25_fig_scopead_overview.png](session_previews/25_fig_scopead_overview.png) |
| 26 | v18 main JPEG 기록 | 2000 × 1143 | [26_fig_scopead_overview.jpg](session_previews/26_fig_scopead_overview.jpg) |

각 이미지의 세션 이름, tool call/output 줄 번호, 시각, 원래 파일 경로는 [기록 인덱스](session_previews/index.json)에 보존했습니다. PNG/JPEG 데이터는 대화 기록에 남은 그대로 저장했고, 픽셀이 동일한 반복 출력만 하나로 묶었습니다.

## 복구 범위와 한계

Git 기록 및 참조가 끊긴 커밋에 대한 기존 조사에 이어, 프로젝트 관련 대화 기록, 발표자료 백업의 embedded media, slide render, 현재 원고 TikZ 소스를 추가 조사했습니다. 기록 중 이미지·코드가 남아 있지 않은 수정 상태는 복구할 수 없습니다. 일부 compact/detailed 수정안은 미리보기 이미지만 복원되어 SVG/PDF가 없습니다. TikZ는 TeX 컴파일러가 없어 소스만 보존했습니다.

과거 그림의 CQ·Noise reference·gate 등은 당시 원본의 내용입니다. 현재 ScopeAD v18 main figure는 변경하지 않았습니다. 생성 스크립트 원본은 각 폴더에 보존했으며 일부는 실행 시 cache 경로 조정이 필요합니다. 초기 SCARF 재출력은 `07_scarf_initial_20261001/render_recovered.py`로 재현할 수 있습니다.
