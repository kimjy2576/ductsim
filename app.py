"""
DuctSim — 1D Duct Pressure Drop Analyzer
Drag & Drop Visual Editor + Simulation
Run: streamlit run app.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json, copy

from streamlit_flow import streamlit_flow
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge

from fluid import FluidState
from geometry import CircularSection, RectangularSection
from components import HeatExchanger, Fan, Filter, Damper
from fittings import Elbow
from duct_segment import DuctSegment
from network import DuctNetwork, Node, NodeType
from solver import solve_single_path, find_operating_point, calc_system_curve

st.set_page_config(page_title="DuctSim", page_icon="🌬️", layout="wide", initial_sidebar_state="expanded")

COMP_DEFS = {
    'duct':   {'icon': '━━', 'label': '직관 덕트', 'color': '#63b3ed', 'bg': '#1a365d'},
    'elbow':  {'icon': '┗',  'label': '엘보',     'color': '#ecc94b', 'bg': '#744210'},
    'hx':     {'icon': '≋',  'label': '열교환기',  'color': '#ed8936', 'bg': '#7b341e'},
    'fan':    {'icon': '◎',  'label': '팬',       'color': '#b794f4', 'bg': '#553c9a'},
    'filter': {'icon': '╋',  'label': '필터',     'color': '#a0aec0', 'bg': '#4a5568'},
    'damper': {'icon': '▷◁', 'label': '댐퍼',     'color': '#68d391', 'bg': '#276749'},
}
DEFAULT_PARAMS = {
    'duct':   {'length': 2.0, 'sec_type': '직사각형', 'W': 600, 'H': 400, 'D': 400, 'roughness': 0.15},
    'elbow':  {'angle': 90, 'r_D': 1.5},
    'hx':     {'UA': 800, 'T_fluid': 7, 'sigma': 0.55, 'f_core': 0.025},
    'fan':    {'a0': 600, 'a1': 0, 'a2': -1500, 'rpm': 1450, 'eta': 0.65},
    'filter': {'C': 45.0, 'n': 1.8, 'loading': 1.0},
    'damper': {'opening': 75},
}

for k in ['pipeline','next_id','selected_node','last_result']:
    if k not in st.session_state:
        st.session_state[k] = [] if k=='pipeline' else (1 if k=='next_id' else None)

st.markdown("""<div style="text-align:center;padding:0.3rem 0 0.8rem"><h1 style="margin:0;letter-spacing:2px">
<span style="color:#63b3ed">Duct</span><span style="color:#4fd1c5">Sim</span></h1>
<p style="color:#718096;font-size:0.85rem;margin:0">드래그 & 드롭 덕트 시스템 구성 → 1D 압력강하 해석</p></div>""", unsafe_allow_html=True)

# ━━ 사이드바 ━━
with st.sidebar:
    st.markdown("### 📐 경계조건")
    c1,c2=st.columns(2)
    T_inlet_C=c1.number_input("입구T[°C]",-20.0,50.0,20.0,1.0)
    P_inlet=c2.number_input("입구P[Pa]",95000,110000,101325,100)
    T_inlet_K=T_inlet_C+273.15
    st.markdown("---")
    st.markdown("### 🧩 컴포넌트 팔레트")
    st.caption("클릭 → 파이프라인 끝에 추가")
    cols=st.columns(3)
    for idx,(ct,cd) in enumerate(COMP_DEFS.items()):
        if cols[idx%3].button(f"{cd['icon']} {cd['label']}",key=f"add_{ct}",use_container_width=True):
            st.session_state.pipeline.append({'type':ct,'id':f"{ct}_{st.session_state.next_id}",'params':copy.deepcopy(DEFAULT_PARAMS[ct])})
            st.session_state.next_id+=1; st.session_state.last_result=None; st.rerun()
    st.markdown("---")
    st.markdown("### 💡 프리셋")
    if st.button("🏭 AHU 시스템",use_container_width=True):
        st.session_state.pipeline=[
            {'type':'duct','id':'duct_1','params':{'length':1.5,'sec_type':'직사각형','W':600,'H':400,'D':400,'roughness':0.15}},
            {'type':'elbow','id':'elbow_2','params':{'angle':90,'r_D':1.5}},
            {'type':'filter','id':'filter_3','params':{'C':45.0,'n':1.8,'loading':1.0}},
            {'type':'hx','id':'hx_4','params':{'UA':800,'T_fluid':7,'sigma':0.55,'f_core':0.025}},
            {'type':'fan','id':'fan_5','params':{'a0':600,'a1':0,'a2':-1500,'rpm':1450,'eta':0.65}},
            {'type':'duct','id':'duct_6','params':{'length':12.0,'sec_type':'직사각형','W':500,'H':350,'D':400,'roughness':0.15}},
            {'type':'elbow','id':'elbow_7','params':{'angle':90,'r_D':1.5}},
            {'type':'damper','id':'damper_8','params':{'opening':75}}]
        st.session_state.next_id=9;st.session_state.last_result=None;st.rerun()
    if st.button("📏 단순 덕트",use_container_width=True):
        st.session_state.pipeline=[
            {'type':'duct','id':'duct_1','params':{'length':10.0,'sec_type':'원형','W':600,'H':400,'D':400,'roughness':0.15}},
            {'type':'elbow','id':'elbow_2','params':{'angle':90,'r_D':1.5}}]
        st.session_state.next_id=3;st.session_state.last_result=None;st.rerun()
    if st.session_state.pipeline and st.button("🗑 전체 초기화",use_container_width=True):
        st.session_state.pipeline=[];st.session_state.next_id=1;st.session_state.selected_node=None;st.session_state.last_result=None;st.rerun()
    st.markdown("---")
    # ── 선택 노드 속성 편집 ──
    sel=st.session_state.selected_node
    if sel and any(it['id']==sel for it in st.session_state.pipeline):
        item=next(it for it in st.session_state.pipeline if it['id']==sel)
        cd=COMP_DEFS[item['type']]; p=item['params']
        st.markdown(f"### ✏️ {cd['icon']} {cd['label']}")
        st.caption(f"ID: {item['id']}")
        if item['type']=='duct':
            p['length']=st.number_input("길이[m]",0.1,100.0,float(p['length']),0.5,key=f"p_l_{sel}")
            p['sec_type']=st.selectbox("단면",["직사각형","원형"],index=0 if p.get('sec_type')=='직사각형' else 1,key=f"p_s_{sel}")
            if p['sec_type']=='직사각형':
                c1,c2=st.columns(2)
                p['W']=c1.number_input("W[mm]",50,2000,int(p.get('W',600)),50,key=f"p_w_{sel}")
                p['H']=c2.number_input("H[mm]",50,2000,int(p.get('H',400)),50,key=f"p_h_{sel}")
            else:
                p['D']=st.number_input("D[mm]",50,2000,int(p.get('D',400)),50,key=f"p_d_{sel}")
            p['roughness']=st.number_input("ε[mm]",0.01,5.0,float(p.get('roughness',0.15)),0.05,key=f"p_r_{sel}")
        elif item['type']=='elbow':
            c1,c2=st.columns(2)
            p['angle']=c1.number_input("각도[°]",15,180,int(p['angle']),15,key=f"p_a_{sel}")
            p['r_D']=c2.number_input("r/D",0.5,3.0,float(p['r_D']),0.25,key=f"p_rd_{sel}")
        elif item['type']=='hx':
            c1,c2=st.columns(2)
            p['UA']=c1.number_input("UA[W/K]",50,5000,int(p['UA']),50,key=f"p_ua_{sel}")
            p['T_fluid']=c2.number_input("2차측T[°C]",-10,90,int(p['T_fluid']),1,key=f"p_tf_{sel}")
            p['sigma']=c1.number_input("σ",0.3,0.9,float(p['sigma']),0.05,key=f"p_sig_{sel}")
            p['f_core']=c2.number_input("f",0.005,0.1,float(p['f_core']),0.005,key=f"p_fc_{sel}")
        elif item['type']=='fan':
            c1,c2,c3=st.columns(3)
            p['a0']=c1.number_input("a₀",0,2000,int(p['a0']),50,key=f"p_a0_{sel}")
            p['a1']=c2.number_input("a₁",-500,500,int(p['a1']),10,key=f"p_a1_{sel}")
            p['a2']=c3.number_input("a₂",-10000,0,int(p['a2']),100,key=f"p_a2_{sel}")
            p['rpm']=st.number_input("RPM",100,5000,int(p['rpm']),50,key=f"p_rpm_{sel}")
            p['eta']=st.number_input("η",0.1,0.95,float(p['eta']),0.05,key=f"p_eta_{sel}")
        elif item['type']=='filter':
            c1,c2=st.columns(2)
            p['C']=c1.number_input("C",5.0,500.0,float(p['C']),5.0,key=f"p_c_{sel}")
            p['n']=c2.number_input("n",1.0,2.0,float(p['n']),0.1,key=f"p_n_{sel}")
            p['loading']=c1.number_input("오염도",1.0,5.0,float(p['loading']),0.5,key=f"p_ld_{sel}")
        elif item['type']=='damper':
            p['opening']=st.slider("θ[°]",0,90,int(p['opening']),5,key=f"p_op_{sel}")
        st.markdown("---")
        c1,c2,c3=st.columns(3)
        idx=next(i for i,it in enumerate(st.session_state.pipeline) if it['id']==sel)
        if c1.button("⬆️",key=f"up_{sel}",disabled=idx==0):
            st.session_state.pipeline[idx],st.session_state.pipeline[idx-1]=st.session_state.pipeline[idx-1],st.session_state.pipeline[idx];st.rerun()
        if c2.button("⬇️",key=f"dn_{sel}",disabled=idx==len(st.session_state.pipeline)-1):
            st.session_state.pipeline[idx],st.session_state.pipeline[idx+1]=st.session_state.pipeline[idx+1],st.session_state.pipeline[idx];st.rerun()
        if c3.button("🗑 삭제",key=f"del_{sel}"):
            st.session_state.pipeline.pop(idx);st.session_state.selected_node=None;st.session_state.last_result=None;st.rerun()
    elif st.session_state.pipeline:
        st.info("👆 캔버스에서 노드를 클릭하면\n속성을 편집할 수 있습니다")

# ━━ 메인: 플로우 캔버스 ━━
if not st.session_state.pipeline:
    st.info("👈 **컴포넌트 팔레트**에서 요소를 추가하거나 **프리셋**을 로드하세요"); st.stop()

pipeline=st.session_state.pipeline
nodes_f,edges_f=[],[]
nodes_f.append(StreamlitFlowNode("__in__",(0,0),{"content":"🟢 **Inlet**"},node_type="input",
    source_position="right",target_position="left",draggable=True,selectable=False,
    style={"background":"#1c4532","color":"#68d391","border":"2px solid #48bb78","borderRadius":"8px","padding":"8px 16px","fontSize":"14px"}))

for i,item in enumerate(pipeline):
    cd=COMP_DEFS[item['type']]; p=item['params']
    if item['type']=='duct': detail=f"{p['length']}m"
    elif item['type']=='elbow': detail=f"{p['angle']}°"
    elif item['type']=='hx': detail=f"UA={p['UA']}"
    elif item['type']=='fan': detail=f"a₀={p['a0']}"
    elif item['type']=='filter': detail=f"C={p['C']}"
    elif item['type']=='damper': detail=f"θ={p['opening']}°"
    else: detail=""
    is_sel=item['id']==st.session_state.selected_node
    border_w="3px" if is_sel else "2px"
    nodes_f.append(StreamlitFlowNode(item['id'],((i+1)*200,0),{"content":f"{cd['icon']} **{cd['label']}**\n\n{detail}"},
        node_type="default",source_position="right",target_position="left",draggable=True,selectable=True,
        style={"background":cd['bg'],"color":cd['color'],"border":f"{border_w} solid {cd['color']}",
               "borderRadius":"10px","padding":"10px 14px","fontSize":"13px","minWidth":"120px","textAlign":"center"}))
    src="__in__" if i==0 else pipeline[i-1]['id']
    edges_f.append(StreamlitFlowEdge(f"e_{i}",src,item['id'],edge_type="smoothstep",animated=True,
        style={"stroke":"#4fd1c5","strokeWidth":2}))

nodes_f.append(StreamlitFlowNode("__out__",((len(pipeline)+1)*200,0),{"content":"🔴 **Outlet**"},node_type="output",
    source_position="right",target_position="left",draggable=True,selectable=False,
    style={"background":"#742a2a","color":"#fc8181","border":"2px solid #e53e3e","borderRadius":"8px","padding":"8px 16px","fontSize":"14px"}))
if pipeline:
    edges_f.append(StreamlitFlowEdge("e_last",pipeline[-1]['id'],"__out__",edge_type="smoothstep",animated=True,
        style={"stroke":"#4fd1c5","strokeWidth":2}))

state=StreamlitFlowState(nodes_f,edges_f)
clicked=streamlit_flow("duct_flow",state=state,height=220,fit_view=True,show_controls=True,
    get_node_on_click=True,pan_on_drag=True,allow_zoom=True,hide_watermark=True,
    style={"backgroundColor":"#0a0e17"})

if clicked and clicked not in ("__in__","__out__") and clicked!=st.session_state.selected_node:
    st.session_state.selected_node=clicked; st.rerun()

# ━━ 네트워크 빌드 ━━
def build_net():
    net=DuctNetwork(); n=len(pipeline)
    for i in range(n+1):
        nt=NodeType.INLET if i==0 else (NodeType.OUTLET if i==n else NodeType.JUNCTION)
        nd=Node(f"N{i}",nt,P_boundary=P_inlet if i==0 else None,T_boundary=T_inlet_K if i==0 else None)
        if i==0: nd.fluid=FluidState(T=T_inlet_K,P=P_inlet)
        net.add_node(nd)
    cur_sec=RectangularSection(W=0.6,H=0.4)
    for i,item in enumerate(pipeline):
        p=item['params']; comps=[]
        if item['type']=='duct':
            length=p['length']; rough=p.get('roughness',0.15)/1000
            if p.get('sec_type')=='원형': cur_sec=CircularSection(D=p.get('D',400)/1000)
            else: cur_sec=RectangularSection(W=p.get('W',600)/1000,H=p.get('H',400)/1000)
        else:
            length=0.05; rough=0.00015; fa=cur_sec.area
            if item['type']=='elbow': comps.append(Elbow(area=fa,angle_deg=p['angle'],r_over_D=p['r_D']))
            elif item['type']=='hx': comps.append(HeatExchanger(face_area=fa,sigma=p['sigma'],UA=p['UA'],T_fluid_in=p['T_fluid']+273.15,f_core=p['f_core']))
            elif item['type']=='fan': comps.append(Fan(curve_coeffs=[float(p['a0']),float(p['a1']),float(p['a2'])],rpm_rated=float(p['rpm']),rpm=float(p['rpm']),eta_total=p['eta']))
            elif item['type']=='filter': comps.append(Filter(face_area=fa,C_resistance=p['C'],n_exponent=p['n'],loading_factor=p['loading']))
            elif item['type']=='damper': comps.append(Damper(area=fa,opening_deg=p['opening']))
        seg=DuctSegment(f"E{i}",copy.deepcopy(cur_sec),length,rough,comps,item['id'])
        net.add_edge(seg,f"N{i}",f"N{i+1}")
    return net

def find_fan(net):
    for s in net.edges.values():
        for c in s.components:
            if isinstance(c,Fan): return c
    return None

network=build_net(); has_fan=find_fan(network) is not None
if not has_fan:
    Q_input=st.number_input("유량 Q [m³/s]",0.01,10.0,1.0,0.1)

c1,c2=st.columns([1,3])
if c1.button("▶️ 해석 실행",type="primary",use_container_width=True):
    try:
        if has_fan:
            fan=find_fan(network); r=find_operating_point(network,P_inlet,T_inlet_K,(0.01,fan.max_flow*1.2))
        else:
            r=solve_single_path(network,Q_input,P_inlet,T_inlet_K)
        st.session_state.last_result=r
        c2.success(f"✅ Q={r.Q:.4f} m³/s ({r.Q*3600:.0f} CMH) | ΔP={r.total_dp:.1f} Pa")
    except Exception as e:
        c2.error(f"❌ {e}"); import traceback; st.code(traceback.format_exc())

result=st.session_state.last_result
if result is None: st.stop()

# ━━ 결과 ━━
path=network.get_path_order()
tab1,tab2,tab3=st.tabs(["📊 압력/온도","🌀 팬곡선","📋 상세"])

with tab1:
    pos=[0.0]
    for eid in path: pos.append(pos[-1]+network.edges[eid].length)
    Pv=[(result.node_pressures.get(f"N{i}",P_inlet)-P_inlet) for i in range(len(pos))]
    Tv=[result.node_temperatures.get(f"N{i}",T_inlet_K)-273.15 for i in range(len(pos))]
    fp=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.08,subplot_titles=("압력 ΔP","온도"))
    fp.add_trace(go.Scatter(x=pos,y=Pv,mode='lines+markers',line=dict(color='#63b3ed',width=2.5),marker=dict(size=8),fill='tozeroy',fillcolor='rgba(99,179,237,0.15)'),row=1,col=1)
    for i,eid in enumerate(path):
        dp=result.edge_dp.get(eid,0)
        fp.add_vrect(x0=pos[i],x1=pos[i+1],fillcolor='rgba(72,187,120,0.1)' if dp<0 else 'rgba(245,158,66,0.08)',line_width=0,row=1,col=1)
    fp.add_trace(go.Scatter(x=pos,y=Tv,mode='lines+markers',line=dict(color='#f6ad55',width=2.5),marker=dict(size=8),fill='tozeroy',fillcolor='rgba(246,173,85,0.12)'),row=2,col=1)
    fp.update_layout(height=420,plot_bgcolor='#0a0e17',paper_bgcolor='#0a0e17',font=dict(family="monospace",color="#a0aec0"),margin=dict(l=60,r=30,t=40,b=40),showlegend=False)
    fp.update_xaxes(title_text="위치[m]",gridcolor='#1e293b',color='#718096',row=2,col=1)
    fp.update_yaxes(title_text="ΔP[Pa]",gridcolor='#1e293b',color='#718096',row=1,col=1)
    fp.update_yaxes(title_text="T[°C]",gridcolor='#1e293b',color='#718096',row=2,col=1)
    st.plotly_chart(fp,use_container_width=True)

with tab2:
    fan=find_fan(network)
    if not fan: st.warning("팬 없음")
    else:
        Qa=np.linspace(0.01,fan.max_flow*1.3,200)
        fd=np.maximum([fan.fan_dp_positive(q) for q in Qa],0)
        sd=calc_system_curve(network,Qa,T_inlet_K,P_inlet)
        Qo,dpo=result.Q,fan.fan_dp_positive(result.Q)
        Wo=dpo*Qo/fan.eta_total if fan.eta_total>0 else 0
        ff=go.Figure()
        ff.add_trace(go.Scatter(x=Qa*3600,y=fd,name='팬',line=dict(color='#63b3ed',width=3)))
        ff.add_trace(go.Scatter(x=Qa*3600,y=sd,name='시스템',line=dict(color='#f6ad55',width=3)))
        ff.add_trace(go.Scatter(x=[Qo*3600],y=[dpo],name='작동점',mode='markers+text',marker=dict(size=14,color='#e53e3e',symbol='star',line=dict(width=2,color='white')),text=[f'Q={Qo*3600:.0f}\nΔP={dpo:.0f}'],textposition='top right',textfont=dict(size=11,color='#e53e3e')))
        ff.add_shape(type="line",x0=Qo*3600,x1=Qo*3600,y0=0,y1=dpo,line=dict(color='#e53e3e',width=1,dash='dash'))
        ff.add_shape(type="line",x0=0,x1=Qo*3600,y0=dpo,y1=dpo,line=dict(color='#e53e3e',width=1,dash='dash'))
        pw=np.array([fan.fan_dp_positive(q)*q/fan.eta_total for q in Qa])
        ff.add_trace(go.Scatter(x=Qa*3600,y=pw,name='전력[W]',line=dict(color='#68d391',width=2,dash='dot'),yaxis='y2'))
        ff.update_layout(height=400,xaxis=dict(title="CMH",gridcolor='#1e293b',color='#718096'),yaxis=dict(title="Pa",gridcolor='#1e293b',color='#718096'),yaxis2=dict(title="W",overlaying='y',side='right',showgrid=False,color='#68d391'),plot_bgcolor='#0a0e17',paper_bgcolor='#0a0e17',font=dict(family="monospace",color="#a0aec0"),legend=dict(x=0.5,y=1.12,xanchor='center',orientation='h'),margin=dict(l=60,r=60,t=40,b=40))
        st.plotly_chart(ff,use_container_width=True)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("풍량",f"{Qo*3600:.0f} CMH")
        m2.metric("정압",f"{dpo:.1f} Pa")
        m3.metric("전력",f"{Wo:.0f} W")
        m4.metric("효율",f"{fan.eta_total*100:.0f}%")

with tab3:
    rows=[]
    for i,eid in enumerate(path):
        seg=network.edges[eid]; dp=result.edge_dp.get(eid,0)
        cs="  ".join(f"{c['name']}:{c['dp']:+.1f}" for c in result.edge_comp_dp.get(eid,[]))
        dn=network.connectivity[eid][1]; it=pipeline[i]; cd=COMP_DEFS[it['type']]
        rows.append({'요소':f"{cd['icon']} {cd['label']}",'Dh[mm]':f"{seg.section.Dh*1000:.1f}",'v[m/s]':f"{result.edge_velocity.get(eid,0):.2f}",'Re':f"{result.edge_Re.get(eid,0):.0f}",'마찰ΔP':f"{result.edge_friction_dp.get(eid,0):.1f}",'컴포넌트ΔP':cs or"-",'총ΔP[Pa]':f"{dp:.1f}",'T_out[°C]':f"{result.node_temperatures.get(dn,T_inlet_K)-273.15:.1f}"})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown("---")
    m1,m2,m3=st.columns(3)
    m1.metric("총ΔP",f"{result.total_dp:.1f} Pa")
    m2.metric("유량",f"{result.Q:.4f} m³/s",f"{result.Q*3600:.0f} CMH")
    To=result.node_temperatures.get(f"N{len(pipeline)}",T_inlet_K)-273.15
    m3.metric("출구T",f"{To:.1f}°C",f"ΔT={To-T_inlet_C:.1f}°C")
