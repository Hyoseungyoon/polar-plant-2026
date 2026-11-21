import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ── 한글 폰트 CSS ──────────────────────────────────────────────
st.markdown(
    '<style>'
    "@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');"
    'html, body, [class*="css"] {'
    "   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;"
    '}'
    '</style>'
    , unsafe_allow_html=True)

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# ── CSV 읽기 (인코딩 폴백) ──────────────────────────────────────
@st.cache_data
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 ─────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(',', '')
                .str.extract(r'(-?\d+\.?\d*)')[0],
                errors='coerce',
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df

def _label(v):
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(v)
    except Exception:
        return str(v)

# ── 사이드바 ────────────────────────────────────────────────────
st.sidebar.header("📁 CSV 업로드")

env_file      = st.sidebar.file_uploader("환경 데이터 CSV",  type="csv")
growth_file   = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

school_name = ''
if env_file is not None:
    school_name = env_file.name.split('_')[0].strip()
    if not school_name:
        school_name = env_file.name.rsplit('.', 1)[0].strip()
if growth_file is not None and not school_name:
    school_name = growth_file.name.split('_')[0].strip()
    if not school_name:
        school_name = growth_file.name.rsplit('.', 1)[0].strip()

if env_file is not None:
    with st.spinner("환경 데이터를 읽는 중..."):
        env_df = std_cols(read_csv(env_file))
else:
    env_df = pd.DataFrame()

if growth_file is not None:
    with st.spinner("생육 결과 데이터를 읽는 중..."):
        growth_df = std_cols(read_csv(growth_file))
else:
    growth_df = pd.DataFrame()

school_display = school_name if school_name else '(파일 선택 시 표시)'
st.sidebar.markdown('🏫 분석 학교: **' + school_display + '**')
st.sidebar.markdown('---')
st.sidebar.subheader("📏 생육 측정 항목")
growth_metric = st.sidebar.selectbox(
    "비교할 지표 선택",
    ["생중량", "잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량"],
    index=0,
)

st.sidebar.markdown('---')
st.sidebar.subheader("🔮 What-If 시뮬레이터")
sim_temp    = st.sidebar.slider("예측 온도 (℃)",  -10.0, 40.0, 15.0, 0.5)
sim_hum     = st.sidebar.slider("예측 습도 (%)",   0.0,  100.0, 60.0,  1.0)
sim_active  = st.sidebar.checkbox("예측 실행", value=False)

# ── 탭 선언 ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    [
        '📊 전체 요약 및 생육 현황',
        '🔍 환경-생육 상관관계 심층 분석',
        '🏆 최적 환경 조건 및 그룹 비교',
    ]
)

with tab1:
    try:
        if env_df.empty and growth_df.empty:
            st.info('좌측 사이드바에서 CSV를 업로드해 주세요.')
        else:
            st.header('📊 전체 요약 및 생육 현황 (Overview)')
            # ── KPI 카드 (4+개) ───────────────────────────────
            kpi_names = []
            kpi_vals  = []
            # ① 평균 생중량
            if '생중량' in growth_df.columns:
                wv = growth_df['생중량'].dropna()
                if len(wv) >= 2:
                    kpi_names.append('평균 생중량')
                    kpi_vals.append(f'{wv.mean():.2f} g')
                else:
                    kpi_names.append('평균 생중량')
                    kpi_vals.append('데이터 부족')
            else:
                kpi_names.append('평균 생중량')
                kpi_vals.append('측정값 없음')
            # ② 평균 초장 (지상부 길이)
            if '지상부 길이' in growth_df.columns:
                sv = growth_df['지상부 길이'].dropna()
                if len(sv) >= 2:
                    kpi_names.append('평균 초장')
                    kpi_vals.append(f'{sv.mean():.1f} mm')
                else:
                    kpi_names.append('평균 초장')
                    kpi_vals.append('데이터 부족')
            else:
                kpi_names.append('평균 초장')
                kpi_vals.append('측정값 없음')
            # ③ 평균 온도
            if 'temperature' in env_df.columns:
                tv = env_df['temperature'].dropna()
                if len(tv) >= 2:
                    kpi_names.append('평균 온도')
                    kpi_vals.append(f'{tv.mean():.1f} ℃')
                else:
                    kpi_names.append('평균 온도')
                    kpi_vals.append('데이터 부족')
            else:
                kpi_names.append('평균 온도')
                kpi_vals.append('측정값 없음')
            # ④ 평균 습도
            if 'humidity' in env_df.columns:
                hv = env_df['humidity'].dropna()
                if len(hv) >= 2:
                    kpi_names.append('평균 습도')
                    kpi_vals.append(f'{hv.mean():.1f} %')
                else:
                    kpi_names.append('평균 습도')
                    kpi_vals.append('데이터 부족')
            else:
                kpi_names.append('평균 습도')
                kpi_vals.append('측정값 없음')
            # ⑤ 목표 달성률 (평균 ÷ 최대값 × 100 → 참고 지표)
            if '생중량' in growth_df.columns:
                wv2 = growth_df['생중량'].dropna()
                if len(wv2) >= 2 and wv2.max() > 0:
                    tr = wv2.mean() / wv2.max() * 100
                    kpi_names.append('목표 달성률 (참고)')
                    kpi_vals.append(f'{tr:.1f} %')
                else:
                    kpi_names.append('목표 달성률')
                    kpi_vals.append('데이터 부족')
            else:
                kpi_names.append('목표 달성률')
                kpi_vals.append('측정값 없음')
            kc = st.columns(len(kpi_names))
            for i in range(len(kpi_names)):
                kc[i].metric(kpi_names[i], kpi_vals[i])
            st.markdown('---')
            # ── 일별 시계열 (이중 축 라인 차트) ─────────────────
            if 'time' in env_df.columns:
                daily = env_df.set_index('time').resample('D').mean().reset_index()
                tc = [c for c in ['temperature', 'humidity'] if c in daily.columns]
                if tc:
                    fig_ts = go.Figure()
                    p0 = tc[0]
                    fig_ts.add_trace(go.Scatter(
                        x=daily['time'], y=daily[p0], mode='lines',
                        name=p0,
                        line=dict(color='firebrick', width=2),
                        yaxis='y',
                    ))
                    if len(tc) >= 2:
                        p1 = tc[1]
                        fig_ts.add_trace(go.Scatter(
                            x=daily['time'], y=daily[p1], mode='lines',
                            name=p1,
                            line=dict(color='seagreen', width=2, dash='dot'),
                            yaxis='y2',
                        ))
                    fig_ts.update_layout(
                        title='📈 재배 일차별 환경 변화 (일별 평균)',
                        xaxis_title='날짜',
                        yaxis=dict(
                            title=p0,
                            tickfont=dict(color='firebrick'),
                        ),
                        font=dict(family='Malgun Gothic, Apple SD Gothic Neo, sans-serif'),
                        hovermode='x unified',
                        legend=dict(
                            orientation='h',
                            yanchor='bottom', y=1.02,
                            xanchor='center', x=0.5,
                        ),
                    )
                    if len(tc) >= 2:
                        fig_ts.update_layout(
                            yaxis2=dict(
                                title=tc[1],
                                overlaying='y', side='right',
                                tickfont=dict(color='seagreen'),
                                showgrid=False,
                            ),
                        )
                    fig_ts.update_xaxes(tickformat='%m/%d')
                    st.plotly_chart(fig_ts, width='stretch')
                else:
                    st.info('온도·습도 시계열을 그릴 열이 없어요.')
            else:
                st.info('환경 데이터에 시간 정보가 없어 시계열 차트를 그리지 못해요.')
            st.markdown('---')
            # ── 생육 분포 (히스토그램 + 박스플롯) ───────────────
            dc = growth_metric if growth_metric in growth_df.columns else '생중량'
            if dc in growth_df.columns:
                dv = growth_df[dc].dropna()
                if len(dv) >= 2:
                    nb = max(10, min(30, max(2, len(dv)//3)))
                    fig_d = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=(f'{dc} 분포 (히스토그램)',
                                        f'{dc} 분포 (박스플롯)'),
                        horizontal_spacing=0.12,
                    )
                    fig_d.add_trace(go.Histogram(
                        x=dv, nbinsx=nb,
                        name='빈도',
                        marker_color='rgba(63,81,181,0.6)',
                        marker_line_color='rgba(63,81,181,1)',
                    ), row=1, col=1)
                    fig_d.add_trace(go.Box(
                        y=dv, name=dc,
                        fillcolor='rgba(63,81,181,0.4)',
                        line=dict(color='rgba(63,81,181,1)'),
                        boxpoints='outliers',
                        marker=dict(size=6),
                    ), row=1, col=2)
                    fig_d.update_layout(
                        title=f'📦 {dc} 분포 — 개체별 차이 한눈에',
                        font=dict(family='Malgun Gothic, Apple SD Gothic Neo, sans-serif'),
                        showlegend=False,
                        height=400,
                    )
                    fig_d.update_xaxes(title_text=dc, row=1, col=1)
                    fig_d.update_yaxes(title_text='빈도', row=1, col=1)
                    fig_d.update_yaxes(title_text=dc, row=1, col=2)
                    st.plotly_chart(fig_d, width='stretch')
                    # 통계 요약 표
                    sr = []
                    sr.append(['평균',       f'{dv.mean():.3f}'])
                    sr.append(['표준편차',    f'{dv.std():.3f}'])
                    cv = dv.std()/dv.mean() if dv.mean() != 0 else 0
                    sr.append(['CV (변동계수)', f'{cv:.3f}'])
                    sr.append(['최소',       f'{dv.min():.3f}'])
                    sr.append(['최대',       f'{dv.max():.3f}'])
                    sr.append(['중앙값',     f'{dv.median():.3f}'])
                    sr.append(['데이터 수',  f'{len(dv)}개'])
                    st.dataframe(pd.DataFrame(sr, columns=['통계','값']),
                                width='stretch', hide_index=True)
                else:
                    st.info(f'{dc} 데이터가 충분하지 않아요.')
            else:
                st.info(f'{dc} 열이 없어 분포를 그리지 못해요.')
    except Exception as e:
        st.warning(f'⚠️ 전체 요약 표시 중 문제가 있었어요: {e}')

with tab2:
    try:
        if env_df.empty or growth_df.empty:
            st.info('환경 CSV와 생육 CSV를 모두 업로드해 주세요.')
        else:
            st.header('🔍 환경-생육 상관관계 심층 분석')
            # ── 상관관계 히트맵 ───────────────────────────────
            d2 = env_df.set_index('time').resample('D').mean().reset_index()
            d2 = d2.dropna(subset=['temperature', 'humidity'])
            env_stat = {}
            for col in ['temperature','humidity','ph','ec','water_temp','co2']:
                if col in d2.columns:
                    v = d2[col].dropna()
                    if len(v) >= 2:
                        env_stat[col] = {'mean':v.mean(), 'std':v.std(),
                                         'min':v.min(), 'max':v.max(),
                                         'range':v.max()-v.min(), 'cum':v.sum()}
            g_stat = {}
            for col in growth_df.columns:
                if col == '개체번호':
                    continue
                v = growth_df[col].dropna()
                if len(v) >= 2:
                    g_stat[col] = {'mean':v.mean(), 'std':v.std(),
                                   'min':v.min(), 'max':v.max(),
                                   'rng':v.max()-v.min()}
            allv = {}
            for col, s in env_stat.items():
                for sk, sv in s.items():
                    allv[f'환경_{col}_{sk}'] = sv
            for col, s in g_stat.items():
                for sk, sv in s.items():
                    allv[f'생육_{col}_{sk}'] = sv
            if len(allv) >= 3:
                mdf  = pd.DataFrame([allv])
                cmat = mdf.T.corr()
                vnames = cmat.index.tolist()
                fig_h = px.imshow(
                    cmat.values,
                    x=vnames, y=vnames,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    aspect='auto',
                )
                fig_h.update_layout(
                    title='🔥 환경-생육 변수 간 상관관계 히트맵 (Pearson r)',
                    xaxis_title='변수',
                    yaxis_title='변수',
                    font=dict(family='Malgun Gothic, Apple SD Gothic Neo, sans-serif'),
                    height=max(350, min(550, len(vnames)*22)),
                )
                fig_h.update_xaxes(tickangle=45)
                st.plotly_chart(fig_h, width='stretch')
                st.caption
            else:
                st.info('상관계수를 구할 변수가 부족해요.')
            st.markdown('---')
            # ── 인터랙티브 산점도 ─────────────────────────────
            ex = [c for c in ['temperature','humidity','ph','ec','water_temp','co2']
                  if c in env_df.columns]
            gy = [c for c in growth_df.columns if c != '개체번호']
            if ex and gy:
                cx, cy = st.columns(2)
                with cx:
                    xc = st.selectbox('X축 (환경 변수)', ex, index=0)
                with cy:
                    yc = st.selectbox('Y축 (생육 지표)', gy, index=0)
                st.markdown('---')
                xv = env_df[xc].dropna()
                yv = growth_df[yc].dropna()
                if len(xv) >= 2 and len(yv) >= 2:
                    xm, xs = xv.mean(), xv.std()
                    ym, ys = yv.mean(), yv.std()
                    n = len(yv)
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(
                        x=np.full(n, xm), y=yv.values, mode='markers',
                        name='개체별 생육값',
                        marker=dict(size=8, color='rgba(63,81,181,0.7)'),
                        hovertemplate=
                            f'{xc}: %{{x:.2f}}<br>{yc}: %{{y:.2f}}<extra></extra>',
                    ))
                    fig_s.add_vline(
                        x=xm, line_dash='dash', line_color='firebrick',
                        annotation_text='환경 평균',
                        annotation_position='top',
                    )
                    fig_s.add_vrect(
                        xm-xs, xm+xs,
                        fillcolor='rgba(255,192,0,0.15)', line_width=0,
                        annotation_text='±1σ 범위',
                        annotation_position='top',
                    )
                    fig_s.update_layout(
                        title=f'📌 {xc} vs {yc} (환경 집계값 기준)',
                        xaxis_title=f'{xc} (환경 평균값)',
                        yaxis_title=f'{yc} (개체별 값)',
                        font=dict(family='Malgun Gothic, Apple SD Gothic Neo, sans-serif'),
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(
                            orientation='h',
                            yanchor='bottom', y=1.02,
                            xanchor='center', x=0.5,
                        ),
                    )
                    st.plotly_chart(fig_s, width='stretch')
                    st.markdown('---')
                    st.subheader('📐 기초 통계 요약')
                    rr = []
                    rr.append(['환경 변수', xc])
                    rr.append([f'{xc} 평균', f'{xm:.2f}'])
                    rr.append([f'{xc} 표준편차', f'{xs:.2f}'])
                    rr.append(['생육 지표', yc])
                    rr.append([f'{yc} 평균', f'{ym:.2f}'])
                    rr.append([f'{yc} 표준편차', f'{ys:.2f}'])
                    rr.append(['개체 수 (N)', f'{n}개'])
                    st.dataframe(pd.DataFrame(rr, columns=['항목','값']),
                                width='stretch', hide_index=True)
                    st.caption
                else:
                    st.info('산점도 데이터가 충분하지 않아요.')
            else:
                st.info('산점도 선택에 필요한 열이 부족해요.')
            st.markdown('---')
            # ── 변수 중요도 (랜덤 포레스트) ───────────────────
            st.subheader('🤖 변수 중요도 — Random Forest Feature Importance')
            st.caption
            # 환경 특징: 일별 집계 통계 → 1행 × 6열 (모든 개체에 동일)
            d2 = env_df.set_index('time').resample('D').mean().reset_index()
            env_feat_names = []
            env_feat_vals  = []  # float 리스트
            for col in ['temperature','humidity','ph','ec','water_temp','co2']:
                if col in d2.columns:
                    v = d2[col].dropna()
                    if len(v) >= 5:
                        env_feat_names.append(col)
                        env_feat_vals.append([
                            float(v.mean()), float(v.std()),
                            float(v.min()),  float(v.max()),
                            float(v.max()-v.min()), float(v.sum()),
                        ])
            if env_feat_names:
                # env_feat_vals: list of list → (n_feat, 6)
                env_arr = np.array(env_feat_vals, dtype=float)  # (n_feat, 6)
                # 모든 개체에 동일 → (n_ind, n_feat, 6) → reshape → (n_ind, n_feat*6)
                # 여기서는 '환경 특징 6개 × 지표별 1세트' 구조로 바로 fit
                imp_rows = []
                for tc in ['생중량','잎 수','지상부 길이','지상부 질량','지하부 질량']:
                    if tc not in growth_df.columns:
                        continue
                    y = growth_df[tc].dropna().values
                    if len(y) < 5:
                        continue
                    # 각 개체마다 동일한 환경 특징을 반복 → (n_ind, n_feat*6)
                    X = np.tile(env_arr, (len(y), 1))  # (n_ind, n_feat*6)
                    if X.shape[0] != len(y):
                        continue
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                    # 특징 이름: '{환경변수}_{통계량}'
                    stat_names = ['mean','std','min','max','range','cum']
                    feat_full_names = [
                        f'{en}_{sn}' for en in env_feat_names for sn in stat_names
                    ]
                    for fn, iv in zip(feat_full_names, rf.feature_importances_):
                        imp_rows.append([tc, fn, float(iv)])
                if imp_rows:
                    idf = pd.DataFrame(imp_rows,
                                       columns=['생육 타겟','환경 특징','중요도'])
                    pv  = idf.pivot(index='환경 특징',
                                    columns='생육 타겟',
                                    values='중요도').fillna(0)
                    pv['평균 중요도'] = pv.mean(axis=1)
                    pv = pv.sort_values('평균 중요도', ascending=False)
                    fig_i = px.bar(
                        x=pv.index,
                        y=pv['평균 중요도'].values,
                        color=pv['평균 중요도'].values,
                        color_continuous_scale='Blues',
                    )
                    fig_i.update_layout(
                        title='📊 환경 특징별 변수 중요도 (랜덤 포레스트, 평균)',
                        xaxis_title='환경 특징',
                        yaxis_title='평균 중요도',
                        font=dict(family='Malgun Gothic, Apple SD Gothic Neo, sans-serif'),
                        showlegend=False,
                        height=400,
                    )
                    fig_i.update_xaxes(tickangle=30)
                    st.plotly_chart(fig_i, width='stretch')
                    st.caption(
                        '※ 중요도 합 ≈ 1. 값이 클수록 해당 환경 특징이 생육 예측 모형에 '
                        '기여한 비중이 큽니다. '
                        '모든 개체가 동일한 환경 값을 가지므로, '
                        '이 결과는 환경 특징 자체의 상대적 중요도를 의미합니다.'
                    )
                else:
                    st.info('변수 중요도를 계산할 타겟이 부족해요.')
            else:
                st.info('환경 특징(변수)을 만들 데이터가 부족해요.')
    except Exception as e:
        st.warning(f'⚠️ 상관관계 분석 표시 중 문제가 있었어요: {e}')

with tab3:
    try:
        if env_df.empty or growth_df.empty:
            st.info('환경 CSV와 생육 CSV를 모두 업로드해 주세요.')
        else:
            st.header('🏆 최적 환경 조건 및 그룹 비교')
            # ── 생중량 기준 상위/하위 20% ─────────────────────
            top_n = bot_n = 0
            q80 = q20 = None
            if '생중량' in growth_df.columns:
                wv = growth_df['생중량'].dropna()
                if len(wv) >= 5:
                    q80 = wv.quantile(0.80)
                    q20 = wv.quantile(0.20)
                    mask_top = growth_df['생중량'] >= q80
                    mask_bot = growth_df['생중량'] <= q20
                    top_n = int(mask_top.sum())
                    bot_n = int(mask_bot.sum())
                    st.markdown
            # ── 레이더 차트 ─────────────────────────────────────
            st.subheader('📡 상위 20% vs 하위 20% 그룹 비교 (레이더 차트)')
            r_labels  = []
            r_full    = []
            r_top     = []
            r_bot     = []
            r_max     = []
            r_min     = []
            # 환경 지표 (전체 평균이 그룹 값이기도 함 → 참고용)
            for lbl, col in [('평균 온도','temperature'), ('평균 습도','humidity')]:
                if col in env_df.columns:
                    v = env_df[col].dropna()
                    if len(v) >= 3:
                        r_labels.append(lbl)
                        r_full.append(float(v.mean()))
                        r_top.append(float(v.mean()))
                        r_bot.append(float(v.mean()))
                        r_max.append(float(v.max()))
                        r_min.append(float(v.min()))
            # 생육 지표
            for col in ['생중량','잎 수','지상부 길이','지상부 질량','지하부 질량']:
                if col in growth_df.columns:
                    v = growth_df[col].dropna()
                    if len(v) >= 3:
                        r_labels.append(col)
                        r_full.append(float(v.mean()))
                        # 상위 그룹은 실제 값이 있으면 사용
                        if top_n > 0 and col == '생중량':
                            r_top.append(float(growth_df.loc[
                                growth_df['생중량'] >= q80, col].mean()))
                            r_bot.append(float(growth_df.loc[
                                growth_df['생중량'] <= q20, col].mean()))
                        else:
                            r_top.append(float(v.mean()))
                            r_bot.append(float(v.mean()))
                        r_max.append(float(v.max()))
                        r_min.append(float(v.min()))
            if r_labels:
                mn  = np.array(r_min, dtype=float)
                mx  = np.array(r_max, dtype=float)
                rng = mx - mn
                rng[rng == 0] = 1
                nf  = (np.array(r_full) - mn) / rng
                nt  = (np.array(r_top)  - mn) / rng
                nb  = (np.array(r_bot)  - mn) / rng
                th  = r_labels + [r_labels[0]]
                nf_c = nf.tolist() + [nf[0]]
                nt_c = nt.tolist() + [nt[0]]
                nb_c = nb.tolist() + [nb[0]]
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatterpolar(
                    r=nf_c, theta=th, fill='toself',
                    name='전체 평균',
                    line=dict(color='rgba(100,100,100,0.6)'),
                    fillcolor='rgba(100,100,100,0.1)',
                ))
                if top_n > 0:
                    fig_r.add_trace(go.Scatterpolar(
                        r=nt_c, theta=th, fill='toself',
                        name=f'상위 20% (n={top_n})',
                        line=dict(color='rgb(34,139,34)'),
                        fillcolor='rgba(34,139,34,0.25)',
                    ))
                if bot_n > 0:
                    fig_r.add_trace(go.Scatterpolar(
                        r=nb_c, theta=th, fill='toself',
                        name=f'하위 20% (n={bot_n})',
                        line=dict(color='rgb(200,50,50)'),
                        fillcolor='rgba(200,50,50,0.2)',
                    ))
                fig_r.update_layout(
                    title='📡 생육 그룹별 지표 비교 (0~1 정규화 레이더)',
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0,1]),
                        angularaxis=dict(tickfont=dict(size=10)),
                    ),
                    font=dict(family='Malgun Gothic, Apple SD Gothic Neo, sans-serif'),
                    showlegend=True,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom', y=1.05,
                        xanchor='center', x=0.5,
                    ),
                    height=500,
                )
                st.plotly_chart(fig_r, width='stretch')
                rdf = pd.DataFrame({
                    '지표': r_labels,
                    '전체 평균': r_full,
                    '최대': r_max,
                    '최소': r_min,
                })
                st.dataframe(rdf, width='stretch', hide_index=True)
            else:
                st.info('레이더 차트에 사용할 지표가 없어요.')
            st.markdown('---')
            # ── 적정 환경 구간 (Optimal Range) ────────────────
            st.subheader('🎯 적정 환경 구간 (Optimal Range)')
            st.caption
            opt_rows = []
            for col in ['temperature','humidity','ph','ec','water_temp','co2']:
                if col in env_df.columns:
                    v = env_df[col].dropna()
                    if len(v) >= 3:
                        q1 = v.quantile(0.25)
                        q3 = v.quantile(0.75)
                        opt_rows.append(
                            [col, f'{v.mean():.2f}', f'{v.std():.2f}',
                             f'{q1:.2f}', f'{q3:.2f}',
                             f'{v.min():.2f}', f'{v.max():.2f}'])
            if opt_rows:
                opt_df = pd.DataFrame(opt_rows,
                    columns=['환경 변수','평균','표준편차',
                             '25% (하한)','75% (상한)',
                             '최소','최대'])
                st.dataframe(opt_df, width='stretch', hide_index=True)
            else:
                st.info('적정 구간을 계산할 환경 변수가 없어요.')
            st.markdown('---')
            # ── What-If 시뮬레이터 ────────────────────────────
            st.subheader('🔮 What-If 시뮬레이터')
            st.caption
            if sim_active and '생중량' in growth_df.columns:
                d3 = env_df.set_index('time').resample('D').mean().reset_index()
                d3 = d3.dropna(subset=['temperature','humidity'])
                if len(d3) >= 3:
                    yw = growth_df['생중량'].dropna().values
                    if len(yw) >= 3:
                        Xf  = d3[['temperature','humidity']].values
                        lr  = LinearRegression()
                        lr.fit(Xf, yw)
                        pred = float(lr.predict([[sim_temp, sim_hum]])[0])
                        r2   = lr.score(Xf, yw)
                        cr  = []
                        cr.append(['절편 (bias)',     f'{lr.intercept_:.3f}'])
                        cr.append(['온도 계수 (℃)',   f'{lr.coef_[0]:.4f} g/℃'])
                        cr.append(['습도 계수 (%공기)', f'{lr.coef_[1]:.4f} g/%'])
                        cr.append(['R² (결정계수)',    f'{r2:.3f}'])
                        c1, c2 = st.columns(2)
                        c1.metric('예상 생중량', f'{pred:.2f} g',
                                  delta=f'온도 {sim_temp}℃ · 습도 {sim_hum}% 기준')
                        c2.dataframe(pd.DataFrame(cr, columns=['항목','값']),
                                     width='stretch', hide_index=True)
                        st.caption
                    else:
                        st.info('생중량 데이터가 충분하지 않아 예측할 수 없어요.')
                else:
                    st.info('환경 데이터가 충분하지 않아 예측할 수 없어요.')
            elif not sim_active:
                st.info('사이드바에서 ✅ 예측 실행을 체크하면 예상 생육 결과가 표시됩니다.')
            else:
                st.info('What-If 예측에 필요한 데이터가 부족해요.')
            st.markdown('---')
            # ── 원본 데이터 다운로드 ──────────────────────────
            with st.expander('📁 원본 데이터 다운로드'):
                if not growth_df.empty:
                    gb = growth_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label='📥 생육 결과 데이터 CSV 다운로드',
                        data=gb,
                        file_name=(school_name or '생육')+'_결과데이터.csv',
                        mime='text/csv',
                    )
                else:
                    st.info('생육 데이터가 없습니다.')
                if not env_df.empty:
                    eb = env_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label='📥 환경 데이터 CSV 다운로드',
                        data=eb,
                        file_name=(school_name or '환경')+'_데이터.csv',
                        mime='text/csv',
                    )
                else:
                    st.info('환경 데이터가 없습니다.')
    except Exception as e:
        st.warning(f'⚠️ 최적 환경 조건 표시 중 문제가 있었어요: {e}')
