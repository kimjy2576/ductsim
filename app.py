"""
DuctSim — 1D Quasi-Steady 덕트 압력강하 해석
Streamlit MVP 메인 앱

실행: streamlit run app.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json

from fluid import FluidState
from geometry import CircularSection, RectangularSection
from components import HeatExchanger, Fan, Filter, Damper
from fittings import Elbow
from duct_segment import DuctSegment
from network import DuctNetwork, Node, NodeType
from solver import solve_single_path, find_operating_point, calc_system_curve

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  페이지 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(page_title="DuctSim", page_icon="🌬️", layout="wide",
                   initial_sidebar_state="expanded")

if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

st.markdown("""
<div style="text-align:center; padding: 0.5rem 0 1rem 0;">
    <h1 style="margin:0; letter-spacing:2px;">
        <span style="color:#63b3ed;">Duct</span><span style="color:#4fd1c5;">Sim</span>
    </h1>
    <p style="color:#718096; font-size:0.85rem; margin:0;">
        1D Quasi-Steady 덕트 압력강하 해석 · MVP
    </p>
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  사이드바 — 네트워크 구성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 🔧 덕트 네트워크 구성")

    st.markdown("### 📐 경계조건")
    col_bc1, col_bc2 = st.columns(2)
    T_inlet_C = col_bc1.number_input("입구 온도 [°C]", -20.0, 50.0, 20.0, 1.0)
    P_inlet = col_bc2.number_input("입구 압력 [Pa]", 95000, 110000, 101325, 100)
    T_inlet_K = T_inlet_C + 273.15

    st.markdown("---")
    st.markdown("### ➕ 세그먼트 추가")

    with st.expander("새 세그먼트 설정", expanded=len(st.session_state.segments) == 0):
        seg_label = st.text_input("세그먼트 이름", f"Seg_{len(st.session_state.segments)+1}")
        sec_type = st.selectbox("단면 형상", ["직사각형", "원형"])
        if sec_type == "직사각형":
            c1, c2 = st.columns(2)
            sec_W = c1.number_input("폭 W [mm]", 50, 2000, 600, 50)
            sec_H = c2.number_input("높이 H [mm]", 50, 2000, 400, 50)
        else:
            sec_D = st.number_input("내경 D [mm]", 50, 2000, 500, 50)

        seg_length = st.number_input("길이 [m]", 0.1, 100.0, 2.0, 0.5)
        seg_roughness = st.number_input("표면조도 ε [mm]", 0.01, 5.0, 0.15, 0.05)

        st.markdown("**부착 컴포넌트**")
        comp_types = st.multiselect("추가할 요소",
            ["엘보 (Elbow)", "열교환기 (HX)", "팬 (Fan)", "필터 (Filter)", "댐퍼 (Damper)"])

        comp_configs = {}
        for ct in comp_types:
            with st.container():
                st.markdown(f"**{ct}**")
                if "엘보" in ct:
                    c1, c2 = st.columns(2)
                    comp_configs['elbow'] = {
                        'angle': c1.number_input("굽힘각 [°]", 15, 180, 90, 15, key=f"elb_a_{seg_label}"),
                        'r_D': c2.number_input("r/D 비", 0.5, 3.0, 1.5, 0.25, key=f"elb_r_{seg_label}"),
                    }
                elif "열교환기" in ct:
                    c1, c2 = st.columns(2)
                    comp_configs['hx'] = {
                        'UA': c1.number_input("UA [W/K]", 50, 5000, 800, 50, key=f"hx_ua_{seg_label}"),
                        'T_fluid': c2.number_input("2차측 입구온도 [°C]", -10, 90, 7, 1, key=f"hx_tf_{seg_label}"),
                        'sigma': c1.number_input("σ (free-flow ratio)", 0.3, 0.9, 0.55, 0.05, key=f"hx_sig_{seg_label}"),
                        'f_core': c2.number_input("코어 마찰계수 f", 0.005, 0.1, 0.025, 0.005, key=f"hx_fc_{seg_label}"),
                    }
                elif "팬" in ct:
                    st.markdown("*팬 곡선: ΔP = a₀ + a₁Q + a₂Q²*")
                    c1, c2, c3 = st.columns(3)
                    comp_configs['fan'] = {
                        'a0': c1.number_input("a₀ [Pa]", 0, 2000, 600, 50, key=f"fan_a0_{seg_label}"),
                        'a1': c2.number_input("a₁", -500, 500, 0, 10, key=f"fan_a1_{seg_label}"),
                        'a2': c3.number_input("a₂", -10000, 0, -1500, 100, key=f"fan_a2_{seg_label}"),
                        'rpm': st.number_input("RPM", 100, 5000, 1450, 50, key=f"fan_rpm_{seg_label}"),
                        'eta': st.number_input("총 효율 η", 0.1, 0.95, 0.65, 0.05, key=f"fan_eta_{seg_label}"),
                    }
                elif "필터" in ct:
                    c1, c2 = st.columns(2)
                    comp_configs['filter'] = {
                        'C': c1.number_input("저항계수 C", 5.0, 500.0, 50.0, 5.0, key=f"fil_c_{seg_label}"),
                        'n': c2.number_input("속도지수 n", 1.0, 2.0, 1.8, 0.1, key=f"fil_n_{seg_label}"),
                        'loading': c1.number_input("오염도", 1.0, 5.0, 1.0, 0.5, key=f"fil_l_{seg_label}"),
                    }
                elif "댐퍼" in ct:
                    comp_configs['damper'] = {
                        'opening': st.slider("개도각 θ [°]", 0, 90, 75, 5, key=f"dmp_o_{seg_label}"),
                    }

        if st.button("✅ 세그먼트 추가", use_container_width=True):
            seg_data = {
                'label': seg_label, 'sec_type': sec_type,
                'sec_params': {'W': sec_W / 1000, 'H': sec_H / 1000} if sec_type == "직사각형" else {'D': sec_D / 1000},
                'length': seg_length, 'roughness': seg_roughness / 1000,
                'components': comp_configs,
            }
            st.session_state.segments.append(seg_data)
            st.rerun()

    # 세그먼트 목록
    if st.session_state.segments:
        st.markdown("### 📋 세그먼트 목록")
        icon_map = {'elbow': '┗', 'hx': '≋', 'fan': '◎', 'filter': '╋', 'damper': '▷◁'}
        for i, seg in enumerate(st.session_state.segments):
            icons = " ".join(icon_map.get(k, '') for k in seg['components'])
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{i+1}. {seg['label']}** {icons}  \n`{seg['length']}m · {seg['sec_type']}`")
            if col2.button("🗑", key=f"del_{i}"):
                st.session_state.segments.pop(i)
                st.rerun()
        if st.button("🗑 전체 초기화", use_container_width=True):
            st.session_state.segments = []
            st.session_state.last_result = None
            st.rerun()

    # 프리셋
    st.markdown("---")
    st.markdown("### 💡 프리셋 예제")
    if st.button("🏭 AHU 시스템 로드", use_container_width=True):
        st.session_state.segments = [
            {'label': '외기 입구 덕트', 'sec_type': '직사각형', 'sec_params': {'W': 0.6, 'H': 0.4},
             'length': 1.5, 'roughness': 0.00015, 'components': {'elbow': {'angle': 90, 'r_D': 1.5}}},
            {'label': '프리필터 (MERV-8)', 'sec_type': '직사각형', 'sec_params': {'W': 0.6, 'H': 0.4},
             'length': 0.3, 'roughness': 0.00015, 'components': {'filter': {'C': 45.0, 'n': 1.8, 'loading': 1.0}}},
            {'label': '냉각코일 (4-row)', 'sec_type': '직사각형', 'sec_params': {'W': 0.6, 'H': 0.4},
             'length': 0.15, 'roughness': 0.00015, 'components': {'hx': {'UA': 800, 'T_fluid': 7, 'sigma': 0.55, 'f_core': 0.025}}},
            {'label': '급기팬', 'sec_type': '직사각형', 'sec_params': {'W': 0.6, 'H': 0.4},
             'length': 0.5, 'roughness': 0.00015, 'components': {'fan': {'a0': 600, 'a1': 0, 'a2': -1500, 'rpm': 1450, 'eta': 0.65}}},
            {'label': '급기 주덕트', 'sec_type': '직사각형', 'sec_params': {'W': 0.5, 'H': 0.35},
             'length': 12.0, 'roughness': 0.00015, 'components': {'elbow': {'angle': 90, 'r_D': 1.5}, 'damper': {'opening': 75}}},
        ]
        st.rerun()
    if st.button("📏 단순 직관 덕트", use_container_width=True):
        st.session_state.segments = [
            {'label': '직관 덕트 10m', 'sec_type': '원형', 'sec_params': {'D': 0.4},
             'length': 10.0, 'roughness': 0.00015, 'components': {'elbow': {'angle': 90, 'r_D': 1.5}}},
        ]
        st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  네트워크 빌드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_network(segments, T_inlet_K, P_inlet):
    net = DuctNetwork()
    n = len(segments)
    for i in range(n + 1):
        ntype = NodeType.INLET if i == 0 else (NodeType.OUTLET if i == n else NodeType.JUNCTION)
        node = Node(id=f"N{i}", node_type=ntype,
                    P_boundary=P_inlet if i == 0 else None,
                    T_boundary=T_inlet_K if i == 0 else None)
        if i == 0:
            node.fluid = FluidState(T=T_inlet_K, P=P_inlet)
        net.add_node(node)

    for i, sd in enumerate(segments):
        sp = sd['sec_params']
        section = RectangularSection(W=sp['W'], H=sp['H']) if sd['sec_type'] == '직사각형' else CircularSection(D=sp['D'])
        comps = []
        cc = sd['components']
        fa = section.area
        if 'elbow' in cc:
            comps.append(Elbow(area=fa, angle_deg=cc['elbow']['angle'], r_over_D=cc['elbow']['r_D']))
        if 'filter' in cc:
            comps.append(Filter(face_area=fa, C_resistance=cc['filter']['C'],
                                n_exponent=cc['filter']['n'], loading_factor=cc['filter']['loading']))
        if 'hx' in cc:
            comps.append(HeatExchanger(face_area=fa, sigma=cc['hx']['sigma'], UA=cc['hx']['UA'],
                                       T_fluid_in=cc['hx']['T_fluid'] + 273.15, f_core=cc['hx']['f_core']))
        if 'fan' in cc:
            comps.append(Fan(curve_coeffs=[float(cc['fan']['a0']), float(cc['fan']['a1']), float(cc['fan']['a2'])],
                             rpm_rated=float(cc['fan']['rpm']), rpm=float(cc['fan']['rpm']), eta_total=cc['fan']['eta']))
        if 'damper' in cc:
            comps.append(Damper(area=fa, opening_deg=cc['damper']['opening']))

        seg = DuctSegment(id=f"E{i}", section=section, length=sd['length'],
                          roughness=sd['roughness'], components=comps, label=sd['label'])
        net.add_edge(seg, f"N{i}", f"N{i+1}")
    return net


def find_fan_in_network(net):
    for seg in net.edges.values():
        for c in seg.components:
            if isinstance(c, Fan): return c
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  메인 영역
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if not st.session_state.segments:
    st.info("👈 사이드바에서 덕트 세그먼트를 추가하거나 **프리셋 예제**를 로드하세요.")
    st.stop()

network = build_network(st.session_state.segments, T_inlet_K, P_inlet)
has_fan = find_fan_in_network(network) is not None

if not has_fan:
    Q_input = st.number_input("유량 Q [m³/s]", 0.01, 10.0, 1.0, 0.1,
                               help="팬이 없으면 직접 유량을 지정해야 합니다.")

col_run, col_status = st.columns([1, 3])
run = col_run.button("▶️ 해석 실행", type="primary", use_container_width=True)

if run:
    try:
        if has_fan:
            fan = find_fan_in_network(network)
            result = find_operating_point(network, P_inlet, T_inlet_K, (0.01, fan.max_flow * 1.2))
        else:
            result = solve_single_path(network, Q_input, P_inlet, T_inlet_K)
        st.session_state.last_result = result
        col_status.success(f"✅ 수렴 | Q = {result.Q:.4f} m³/s ({result.Q*3600:.0f} CMH) | ΔP = {result.total_dp:.1f} Pa")
    except Exception as e:
        col_status.error(f"❌ 해석 실패: {e}")
        import traceback; st.code(traceback.format_exc())
        st.stop()

result = st.session_state.last_result
if result is None:
    st.info("▶️ **해석 실행** 버튼을 눌러 결과를 확인하세요.")
    st.stop()

segments_data = st.session_state.segments
path = network.get_path_order()

tab_layout, tab_profile, tab_fan, tab_table = st.tabs([
    "🗺️ 2D 배치도", "📊 압력/온도 프로파일", "🌀 팬곡선 매칭", "📋 상세 결과"])


# ── TAB 1: 2D 배치도 ──
with tab_layout:
    fig = go.Figure()
    x_pos, node_x, node_labels = 0.0, [0.0], ["N0\n(Inlet)"]
    comp_icons = {'Elbow': '┗', 'HX': '≋', 'Fan': '◎', 'Filter': '╋', 'Damper': '▷◁'}
    max_dp = max(abs(result.edge_dp.get(eid, 0)) for eid in path) or 1

    for i, eid in enumerate(path):
        seg = network.edges[eid]
        dp = result.edge_dp.get(eid, 0)
        ratio = min(abs(dp) / max_dp, 1.0)
        color = (f"rgba(72,187,120,{0.5+0.5*ratio})" if dp < 0
                 else f"rgba(245,{int(158-120*ratio)},{int(66*(1-ratio))},{0.5+0.5*ratio})")
        x_end = x_pos + seg.length
        h = (seg.section.H if hasattr(seg.section, 'H') else seg.section.D) * 2

        fig.add_shape(type="rect", x0=x_pos, x1=x_end, y0=-h/2, y1=h/2,
                      fillcolor=color, line=dict(color="rgba(255,255,255,0.3)", width=1))
        fig.add_annotation(x=(x_pos+x_end)/2, y=h/2+0.12, text=f"<b>{segments_data[i]['label']}</b>",
                           showarrow=False, font=dict(size=10, color="white"))
        dp_txt = f"{dp:.1f} Pa" if dp >= 0 else f"+{abs(dp):.1f} Pa"
        fig.add_annotation(x=(x_pos+x_end)/2, y=-h/2-0.1, text=dp_txt, showarrow=False,
                           font=dict(size=9, color="#63b3ed" if dp < 0 else "#f59e42"))
        icons_str = "  ".join(comp_icons.get(c.name, '?') for c in seg.components)
        if icons_str:
            fig.add_annotation(x=(x_pos+x_end)/2, y=0, text=icons_str, showarrow=False,
                               font=dict(size=14, color="white", family="monospace"))
        x_pos = x_end
        node_x.append(x_pos)
        lbl = f"N{i+1}" + ("\n(Outlet)" if i+1 == len(segments_data) else "")
        node_labels.append(lbl)

    node_P = [result.node_pressures.get(f"N{i}", P_inlet) for i in range(len(node_x))]
    node_T = [result.node_temperatures.get(f"N{i}", T_inlet_K) - 273.15 for i in range(len(node_x))]
    fig.add_trace(go.Scatter(
        x=node_x, y=[0]*len(node_x), mode='markers+text',
        marker=dict(size=10, color='#63b3ed', line=dict(width=2, color='white')),
        text=node_labels, textposition='top center', textfont=dict(size=9, color='#a0aec0'),
        hovertemplate='<b>%{text}</b><br>P: %{customdata[0]:.0f} Pa<br>T: %{customdata[1]:.1f} °C',
        customdata=list(zip(node_P, node_T)), showlegend=False))
    fig.add_annotation(x=node_x[-1]+0.3, y=0, ax=node_x[-1], ay=0,
                       arrowhead=3, arrowsize=1.5, arrowcolor="#4fd1c5", showarrow=True, text="")
    fig.update_layout(height=280, plot_bgcolor='#0a0e17', paper_bgcolor='#0a0e17',
                      margin=dict(l=20, r=20, t=30, b=30),
                      xaxis=dict(title="위치 [m]", showgrid=True, gridcolor='#1e293b', color='#718096'),
                      yaxis=dict(showticklabels=False, showgrid=False, range=[-0.8, 0.8]),
                      font=dict(family="monospace", color="#a0aec0"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""<div style="display:flex;gap:20px;justify-content:center;font-size:0.8rem;color:#718096;">
        <span>┗ 엘보</span><span>≋ 열교환기</span><span>◎ 팬</span><span>╋ 필터</span><span>▷◁ 댐퍼</span>
        <span style="color:#f59e42;">■ 압력강하</span><span style="color:#48bb78;">■ 압력상승(팬)</span>
    </div>""", unsafe_allow_html=True)


# ── TAB 2: 압력/온도 프로파일 ──
with tab_profile:
    positions = [0.0]
    for eid in path:
        positions.append(positions[-1] + network.edges[eid].length)
    P_vals = [(result.node_pressures.get(f"N{i}", P_inlet) - P_inlet) for i in range(len(positions))]
    T_vals = [result.node_temperatures.get(f"N{i}", T_inlet_K) - 273.15 for i in range(len(positions))]

    figp = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("압력 프로파일 (ΔP from inlet)", "온도 프로파일"))
    figp.add_trace(go.Scatter(x=positions, y=P_vals, mode='lines+markers', name='ΔP',
                              line=dict(color='#63b3ed', width=2.5),
                              marker=dict(size=8, color='#63b3ed'), fill='tozeroy',
                              fillcolor='rgba(99,179,237,0.15)'), row=1, col=1)
    for i, eid in enumerate(path):
        dp = result.edge_dp.get(eid, 0)
        clr = 'rgba(72,187,120,0.1)' if dp < 0 else 'rgba(245,158,66,0.08)'
        figp.add_vrect(x0=positions[i], x1=positions[i+1], fillcolor=clr, line_width=0, row=1, col=1)
        figp.add_annotation(x=(positions[i]+positions[i+1])/2, y=min(P_vals)-15,
                            text=segments_data[i]['label'], showarrow=False,
                            font=dict(size=8, color='#4a5568'), row=1, col=1)
    figp.add_trace(go.Scatter(x=positions, y=T_vals, mode='lines+markers', name='T',
                              line=dict(color='#f6ad55', width=2.5),
                              marker=dict(size=8, color='#f6ad55'), fill='tozeroy',
                              fillcolor='rgba(246,173,85,0.12)'), row=2, col=1)
    figp.update_layout(height=500, plot_bgcolor='#0a0e17', paper_bgcolor='#0a0e17',
                       font=dict(family="monospace", color="#a0aec0"),
                       margin=dict(l=60, r=30, t=40, b=40), showlegend=False)
    figp.update_xaxes(title_text="위치 [m]", gridcolor='#1e293b', color='#718096', row=2, col=1)
    figp.update_yaxes(title_text="ΔP [Pa]", gridcolor='#1e293b', color='#718096', row=1, col=1)
    figp.update_yaxes(title_text="T [°C]", gridcolor='#1e293b', color='#718096', row=2, col=1)
    st.plotly_chart(figp, use_container_width=True)


# ── TAB 3: 팬곡선 매칭 ──
with tab_fan:
    fan = find_fan_in_network(network)
    if fan is None:
        st.warning("네트워크에 팬이 없습니다. 팬을 추가하면 곡선 매칭을 볼 수 있습니다.")
    else:
        Q_arr = np.linspace(0.01, fan.max_flow * 1.3, 200)
        fan_dp = np.maximum([fan.fan_dp_positive(q) for q in Q_arr], 0)
        sys_dp = calc_system_curve(network, Q_arr, T_inlet_K, P_inlet)
        Q_op = result.Q
        dp_op = fan.fan_dp_positive(Q_op)
        W_op = dp_op * Q_op / fan.eta_total if fan.eta_total > 0 else 0
        fan_power = np.array([fan.fan_dp_positive(q)*q/fan.eta_total if fan.eta_total > 0 else 0 for q in Q_arr])

        figf = go.Figure()
        figf.add_trace(go.Scatter(x=Q_arr*3600, y=fan_dp, mode='lines', name='팬 곡선',
                                  line=dict(color='#63b3ed', width=3)))
        figf.add_trace(go.Scatter(x=Q_arr*3600, y=sys_dp, mode='lines', name='시스템 곡선',
                                  line=dict(color='#f6ad55', width=3)))
        figf.add_trace(go.Scatter(x=[Q_op*3600], y=[dp_op], mode='markers+text', name='작동점',
                                  marker=dict(size=14, color='#e53e3e', symbol='star',
                                              line=dict(width=2, color='white')),
                                  text=[f'Q={Q_op*3600:.0f} CMH<br>ΔP={dp_op:.0f} Pa'],
                                  textposition='top right', textfont=dict(size=11, color='#e53e3e')))
        figf.add_shape(type="line", x0=Q_op*3600, x1=Q_op*3600, y0=0, y1=dp_op,
                       line=dict(color='#e53e3e', width=1, dash='dash'))
        figf.add_shape(type="line", x0=0, x1=Q_op*3600, y0=dp_op, y1=dp_op,
                       line=dict(color='#e53e3e', width=1, dash='dash'))
        figf.add_trace(go.Scatter(x=Q_arr*3600, y=fan_power, mode='lines', name='팬 소비전력 [W]',
                                  line=dict(color='#68d391', width=2, dash='dot'), yaxis='y2'))
        figf.update_layout(
            height=450, title=dict(text="팬곡선 - 시스템곡선 매칭", font=dict(size=14, color='#e2e8f0')),
            xaxis=dict(title="풍량 [CMH]", gridcolor='#1e293b', color='#718096'),
            yaxis=dict(title="정압 [Pa]", gridcolor='#1e293b', color='#718096'),
            yaxis2=dict(title="전력 [W]", overlaying='y', side='right', showgrid=False, color='#68d391'),
            plot_bgcolor='#0a0e17', paper_bgcolor='#0a0e17',
            font=dict(family="monospace", color="#a0aec0"),
            legend=dict(x=0.5, y=1.12, xanchor='center', orientation='h', font=dict(size=10)),
            margin=dict(l=60, r=60, t=60, b=40))
        st.plotly_chart(figf, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("작동 풍량", f"{Q_op*3600:.0f} CMH", f"{Q_op:.3f} m³/s")
        c2.metric("작동 정압", f"{dp_op:.1f} Pa")
        c3.metric("팬 소비전력", f"{W_op:.0f} W", f"{W_op/1000:.2f} kW")
        c4.metric("팬 효율", f"{fan.eta_total*100:.0f} %")


# ── TAB 4: 상세 테이블 ──
with tab_table:
    st.markdown("#### Edge 별 상세 결과")
    rows = []
    for i, eid in enumerate(path):
        seg = network.edges[eid]
        dp = result.edge_dp.get(eid, 0)
        comp_dp_str = "  ".join(f"{cd['name']}: {'+' if cd['dp']>=0 else ''}{cd['dp']:.1f} Pa"
                                for cd in result.edge_comp_dp.get(eid, []))
        dn_id = network.connectivity[eid][1]
        rows.append({
            '세그먼트': segments_data[i]['label'], '길이 [m]': seg.length,
            'Dh [mm]': f"{seg.section.Dh*1000:.1f}",
            '유속 [m/s]': f"{result.edge_velocity.get(eid,0):.2f}",
            'Re': f"{result.edge_Re.get(eid,0):.0f}",
            '마찰 ΔP [Pa]': f"{result.edge_friction_dp.get(eid,0):.1f}",
            '컴포넌트 ΔP': comp_dp_str or "-",
            '총 ΔP [Pa]': f"{dp:.1f}",
            '출구 T [°C]': f"{result.node_temperatures.get(dn_id, T_inlet_K)-273.15:.1f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Node 압력/온도")
    node_rows = []
    for i in range(len(segments_data) + 1):
        nid = f"N{i}"
        P = result.node_pressures.get(nid, P_inlet)
        T = result.node_temperatures.get(nid, T_inlet_K) - 273.15
        lbl = "Inlet" if i == 0 else ("Outlet" if i == len(segments_data) else f"Junction {i}")
        node_rows.append({'Node': f"N{i} ({lbl})", 'P [Pa]': f"{P:.1f}",
                          'ΔP from inlet [Pa]': f"{P - P_inlet:.1f}", 'T [°C]': f"{T:.1f}"})
    st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("총 ΔP", f"{result.total_dp:.1f} Pa")
    c2.metric("유량", f"{result.Q:.4f} m³/s", f"{result.Q*3600:.0f} CMH")
    T_out = result.node_temperatures.get(f"N{len(segments_data)}", T_inlet_K) - 273.15
    c3.metric("출구 온도", f"{T_out:.1f} °C", f"ΔT = {T_out - T_inlet_C:.1f} °C")

    with st.expander("📥 JSON 내보내기"):
        export = {
            'segments': st.session_state.segments,
            'boundary': {'T_inlet_C': T_inlet_C, 'P_inlet': P_inlet},
            'results': {'Q': result.Q, 'CMH': result.Q * 3600, 'total_dp': result.total_dp,
                        'edge_dp': result.edge_dp, 'node_P': result.node_pressures,
                        'node_T': {k: v - 273.15 for k, v in result.node_temperatures.items()}}
        }
        st.code(json.dumps(export, indent=2, ensure_ascii=False), language='json')
