# SCARF 한국어 논문 초안

PaperWorks/outlines/Outline_v16.3.md를 기준으로 2026-10-03 작성했다. CVPR 2026 review 템플릿의 2단 구조와 기본 글자 크기를 유지한다. 한국어 연구 검토용 초안이며 Related Works는 요청에 따라 제목만 남겼다.

## 빌드

LuaLaTeX, BibTeX, ko.TeX 및 Noto CJK KR 글꼴이 필요하다. Ubuntu 패키지는 texlive-luatex, texlive-lang-korean, fonts-noto-cjk다.
main.tex는 이 환경에서 ko.TeX의 Lua 초기화를 위해 kotex를 cvpr.sty보다 먼저 불러온다. preamble.tex는 TeX Gyre의 라틴 글꼴과 Noto CJK KR 글꼴을 설정한다. Korean italic 모양은 사용할 글꼴에 별도 이탤릭 면이 없어 기울임을 합성한다.

    make -C PaperWorks/cvpr-2026-author-kit-template

결과는 main.pdf다. 이 세션에서는 누락된 TeX 패키지를 관리자 권한 없이 사용자 TEXMF 트리에 설치했다. 다른 환경에서는 위 의존성을 설치해야 한다. main.tex가 각 sec/*.tex를 통합하며 참고문헌은 main.bib에서 관리한다. 원본 템플릿의 예시 파일 중 본문에서 호출하지 않는 파일은 보존했다.
고정 4해상도 방법 그림은 figures/scarf_core.tex의 TikZ 벡터 소스다. 별도 래스터 그림 변환은 필요하지 않다. Makefile은 figures 디렉터리의 파일을 PDF 생성 의존성으로 추적한다.

## 한국어 서술 규칙

ASD-STE100의 명료한 기술 문장 원칙을 한국어에 적용했다. 한 문장에는 한 가지 핵심 내용을 담는다. 같은 개념에는 같은 용어를 쓴다. 연산의 목적을 설명한 뒤 수식을 제시한다. 관측 결과와 원인 가설을 구분한다. 영어 통제어휘를 따르는 표준의 정식 준수를 주장하지 않는다. 공식 소개: https://www.asd-ste100.org/about.html

## 근거와 한계

표와 주장의 출처 및 남은 검증 항목은 WRITING_AUDIT.md에 기록한다. 아웃라인의 문구가 코드나 실측 근거와 어긋나면 근거에 맞게 수정했다. 미측정 실험은 수치를 만들지 않았다. 고정 네 해상도 핵심 구성, 조건부 해상도 선택을 포함한 RSG4 벤치마크, 과거 RSG3 소거 실험을 구분한다.

## 템플릿 출처

- 원본: https://github.com/cvpr-org/author-kit/archive/refs/tags/CVPR2026-v1(latex).zip
- ZIP SHA-256: ef672bf4ad1c6801237b957d57b0087d4412e7dee01ad3b512abe82960c6466c
- cvpr.sty는 원본을 유지한다. paper ID와 저자는 익명 자리표시자다.
