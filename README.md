# 🌬️ DuctSim — 1D 덕트 압력강하 해석

덕트 시스템의 1D 준정상 압력강하를 해석하는 Streamlit 앱

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 기능

- 단일 경로 정상상태 해석 (Darcy-Weisbach + K-factor)
- Fan-System 매칭 (Brent's method)
- 6종 컴포넌트: 열교환기(ε-NTU) / 팬 / 필터 / 댐퍼 / 엘보 / 직관마찰
- 시각화: 2D 배치도, 압력·온도 프로파일, 팬곡선 매칭

## 파일 구조

| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit 메인 앱 (UI + 시각화) |
| `fluid.py` | 유체 물성 (CoolProp / fallback 간이식) |
| `geometry.py` | 단면 형상 (원형, 직사각형) |
| `components.py` | HX, Fan, Filter, Damper |
| `fittings.py` | Elbow (K-factor) |
| `duct_segment.py` | Edge (직관 마찰 + 컴포넌트 체인) |
| `network.py` | Node-Edge 그래프 |
| `solver.py` | 단일경로 솔버 + Fan 매칭 |

## 물리 모델

| 요소 | 모델 |
|------|------|
| 직관 마찰 | Darcy-Weisbach + Swamee-Jain |
| 엘보 | ASHRAE K-factor (r/D, 각도) |
| 열교환기 | Kays-London ΔP + ε-NTU |
| 팬 | 다항식 곡선 + Affinity Law |
| 필터 | ΔP = C·v^n |
| 댐퍼 | K(θ)·ρv²/2 |
