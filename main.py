import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from scipy import stats

# ── 한글 폰트 CSS ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

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
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
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

# ── 사이드바 ───────────────────────────────────────────────────
st.sidebar.header("📁 CSV 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

school_name = ""
if env_file is not None:
    school_name = env_file.name.split("_")[0].strip()
    if not school_name:
        school_name = env_file.name.rsplit(".", 1)[0].strip()
if growth_file is not None and not school_name:
    school_name = growth_file.name.split("_")[0].strip()
    if not school_name:
        school_name = growth_file.name.rsplit(".", 1)[0].strip()

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

st.sidebar.markdown(f"🏫 분석 학교: **{school_name or '(파일 선택 시 표시)'}**")

st.sidebar.markdown("---")
st.sidebar.subheader("📏 생육 측정 항목")
growth_metric = st.sidebar.selectbox(
    "비교할 지표 선택",
    ["생중량", "잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량"],
    index=0,
)

# ── 탭 ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ────────────────────────────────────────────────────────────────
# Tab 1 — 실험 개요
# ────────────────────────────────────────────────────────────────
with tab1:
    try:
        with st.expander("🔍 연구 배경 및 목적"):
            st.markdown(
                """
**연구 배경**
극지(고산·극지) 식물은 짧은 생육 기간 동안 빠르게 자라야 하므로 개체마다 생장 차이가 크게 나타납니다.
본 대시보드는 우리 학교 극지식물 실험에서 **개체별 생육 차이**를 통계적으로 분석합니다.

**분석 목적**
1. 개체별 생중량 분포와 개체 간 차이(표준편차·CV)를 이해한다.
2. 잎 수·지상부 길이와 생중량 사이의 관계를 상관분석과 산점도로 확인한다.
3. 생중량 상위 25% 개체와 하위 25% 개체를 비교하여 차이를 시각화한다.

**통계 용어 간단 설명**
- **평균**: 모든 값을 더해 개수로 나눈 대표값.
- **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 값(클수록 들쭉날쭉).
- **CV(변동계수)**: 표준편차를 평균으로 나눈 값. 단위가 다른 자료끼리도 비교 가능.
- **상관계수(r)**: 두 변수의 직선 관계 강도(-1 ~ 1). 0에 가까울수록 관계 약함.
                """
            )

        st.subheader("📋 데이터 요약")

        rows = []
        if not env_df.empty:
            rows.append(("환경 데이터", f"{len(env_df):,}행 × {len(env_df.columns):,}열"))
            if "time" in env_df.columns and env_df["time"].notna().any():
                tmin = env_df["time"].min()
                tmax = env_df["time"].max()
                rows.append(("측정 기간", f"{tmin.strftime('%Y-%m-%d')} ~ {tmax.strftime('%Y-%m-%d')}"))
            else:
                rows.append(("측정 기간", "시간 정보 없음"))
        else:
            rows.append(("환경 데이터", "업로드 필요"))
            rows.append(("측정 기간", "—"))

        if not growth_df.empty:
            rows.append(("생육 데이터", f"{len(growth_df):,}행 × {len(growth_df.columns):,}열"))
            rows.append(("개체 수", f"{len(growth_df):,}개체"))
            meas_cols = [c for c in growth_df.columns if c != "개체번호"]
            rows.append(("측정 항목", f"{len(meas_cols)}개"))
        else:
            rows.append(("생육 데이터", "업로드 필요"))
            rows.append(("개체 수", "—"))
            rows.append(("측정 항목", "—"))

        summary_df = pd.DataFrame(rows, columns=["구분", "내용"])
        st.dataframe(summary_df, width="stretch")

        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n = len(growth_df) if not growth_df.empty else 0
            st.metric("총 개체수", f"{n}개체")

        with col2:
            if "temperature" in env_df.columns:
                vals = env_df["temperature"].dropna()
                if len(vals) >= 2:
                    st.metric("평균 온도", f"{vals.mean():.1f} ℃")
                else:
                    st.metric("평균 온도", "데이터 부족")
            else:
                st.metric("평균 온도", "측정값 없음")

        with col3:
            if "humidity" in env_df.columns:
                vals = env_df["humidity"].dropna()
                if len(vals) >= 2:
                    st.metric("평균 습도", f"{vals.mean():.1f} %")
                else:
                    st.metric("평균 습도", "데이터 부족")
            else:
                st.metric("평균 습도", "측정값 없음")

        with col4:
            if "생중량" in growth_df.columns:
                vals = growth_df["생중량"].dropna()
                if len(vals) >= 2:
                    st.metric("평균 생중량", f"{vals.mean():.2f} g")
                else:
                    st.metric("평균 생중량", "데이터 부족")
            else:
                st.metric("평균 생중량", "측정값 없음")

    except Exception as e:
        st.warning(f"⚠️ 실험 개요 표시 중 문제가 있었어요: {e}")

# ────────────────────────────────────────────────────────────────
# Tab 2 — 환경 데이터
# ────────────────────────────────────────────────────────────────
with tab2:
    try:
        if env_df.empty:
            st.info("좌측 사이드바에서 환경 CSV를 업로드해 주세요.")
        else:
            st.header("🌡️ 환경 데이터 분석")

            env_cols = [c for c in ["temperature", "humidity", "ph", "ec", "water_temp", "co2"] if c in env_df.columns]
            if env_cols:
                summary_data = []
                col_label = {
                    "temperature": "온도 (℃)",
                    "humidity": "습도 (%)",
                    "ph": "pH",
                    "ec": "EC (dS/m)",
                    "water_temp": "수온 (℃)",
                    "co2": "CO₂ (ppm)",
                }
                for col in env_cols:
                    vals = env_df[col].dropna()
                    if len(vals) >= 1:
                        summary_data.append([
                            col_label.get(col, col),
                            f"{vals.mean():.2f}",
                            f"{vals.min():.2f}",
                            f"{vals.max():.2f}",
                        ])
                if summary_data:
                    env_summary_df = pd.DataFrame(summary_data, columns=["항목", "평균", "최소", "최대"])
                    st.subheader("📊 환경 요약 통계")
                    st.dataframe(env_summary_df, width="stretch")
                    st.markdown("---")

            if "time" in env_df.columns:
                daily = env_df.set_index("time").resample("D").mean().reset_index()
                temp_hum_cols = [c for c in ["temperature", "humidity"] if c in daily.columns and daily[c].notna().any()]
                if temp_hum_cols:
                    fig_th = go.Figure()
                    for col in temp_hum_cols:
                        fig_th.add_trace(go.Scatter(x=daily["time"], y=daily[col], mode="lines", name=col))
                    fig_th.update_layout(
                        title="일별 평균 온도·습도 변화",
                        xaxis_title="날짜",
                        yaxis_title="값",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    )
                    fig_th.update_xaxes(tickformat="%m/%d")
                    st.plotly_chart(fig_th, width="stretch")

                ph_ec_cols = [c for c in ["ph", "ec"] if c in daily.columns and daily[c].notna().any()]
                if ph_ec_cols:
                    fig_pe = go.Figure()
                    for col in ph_ec_cols:
                        fig_pe.add_trace(go.Scatter(x=daily["time"], y=daily[col], mode="lines", name=col))
                    fig_pe.update_layout(
                        title="일별 평균 pH·EC 변화",
                        xaxis_title="날짜",
                        yaxis_title="값",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    )
                    fig_pe.update_xaxes(tickformat="%m/%d")
                    st.plotly_chart(fig_pe, width="stretch")

            st.markdown("---")

            with st.expander("📁 환경 데이터 원본 테이블 + CSV 다운로드"):
                if not env_df.empty:
                    st.dataframe(env_df, width="stretch")
                    csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 환경 데이터 CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{school_name or '환경'}_데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("업로드된 환경 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 환경 데이터 표시 중 문제가 있었어요: {e}")

# ────────────────────────────────────────────────────────────────
# Tab 3 — 생육 결과
# ────────────────────────────────────────────────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info("좌측 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            st.header("📊 생육 결과 분석")
            st.caption("※ 개체 간 비교만 진행합니다 (개체번호는 분석 항목에서 제외).")

            # ── 🥇 핵심 결과 카드 ──────────────────────────────
            if "생중량" in growth_df.columns:
                w = growth_df["생중량"].dropna()
                if len(w) >= 1:
                    mean_w = w.mean()
                    max_idx = w.idxmax()
                    min_idx = w.idxmin()
                    max_val = w.max()
                    min_val = w.min()

                    id_col = growth_df["개체번호"] if "개체번호" in growth_df.columns else None

                    def make_label(i):
                        if id_col is not None:
                            v = id_col.loc[i]
                            return str(int(v)) if float(v).is_integer() else str(v)
                        return f"개체 {i}"

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("평균 생중량", f"{mean_w:.2f} g")
                    c2.metric("최대 생중량", f"{max_val:.2f} g", delta=f"({make_label(max_idx)})")
                    c3.metric("최소 생중량", f"{min_val:.2f} g", delta=f"({make_label(min_idx)})")
                    c4.metric("개체 수", f"{len(w)}개")

                    st.markdown("---")

            # ── 막대 그래프 함수 (재사용) ──────────────────────
            def plot_bar(metric_col, title, ylabel):
                if metric_col not in growth_df.columns:
                    st.info(f"{title} 데이터가 없어요 (열이 없습니다).")
                    return
                vals = growth_df[metric_col].dropna()
                if len(vals) < 2:
                    st.info(f"{title} 데이터가 충분하지 않아요.")
                    return
                sub = growth_df.loc[vals.index].copy()
                sub = sub.sort_values(metric_col, ascending=False).reset_index(drop=True)

                id_col = growth_df["개체번호"] if "개체번호" in growth_df.columns else None
                if id_col is not None:
                    sub["label"] = sub["개체번호"].apply(lambda v: str(int(v)) if float(v).is_integer() else str(v))
                else:
                    sub["label"] = [f"개체 {i+1}" for i in range(len(sub))]

                fig = go.Figure()
                fig.add_trace(go.Bar(x=sub["label"], y=sub[metric_col], name=metric_col))
                fig.update_layout(
                    title=title,
                    xaxis_title="개체",
                    yaxis_title=ylabel,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    xaxis=dict(tickangle=45),
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")

            # ── 개체별 생육 비교 ───────────────────────────────
            st.subheader("🌿 개체별 생육 비교 (막대 그래프)")
            metric_map = {
                "생중량": ("생중량 (g)", "생중량"),
                "잎 수": ("잎 수", "잎 수"),
                "지상부 길이": ("지상부 길이 (mm)", "지상부 길이"),
                "지하부 길이": ("지하부 길이 (mm)", "지하부 길이"),
            }
            if growth_metric in metric_map:
                ylbl, mcol = metric_map[growth_metric]
                plot_bar(mcol, f"개체별 평균 {growth_metric}", ylbl)

            # ── 생중량 분포 ────────────────────────────────────
            st.markdown("---")
            st.subheader("📦 생중량 분포")
            if "생중량" in growth_df.columns:
                w = growth_df["생중량"].dropna()
                if len(w) >= 2:
                    use_violin = len(w) >= 5
                    fig_w = go.Figure()
                    if use_violin:
                        fig_w.add_trace(go.Violin(
                            y=w,
                            name="생중량",
                            fillcolor="rgba(63, 81, 181, 0.5)",
                            line=dict(color="rgba(63, 81, 181, 1)"),
                            points="outliers",
                        ))
                    else:
                        fig_w.add_trace(go.Box(
                            y=w,
                            name="생중량",
                            fillcolor="rgba(63, 81, 181, 0.5)",
                            line=dict(color="rgba(63, 81, 181, 1)"),
                            boxpoints="outliers",
                        ))
                    fig_w.update_layout(
                        title="생중량 분포 (박스플롯 / 바이올린 플롯)",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_w, width="stretch")
                else:
                    st.info("생중량 데이터가 충분하지 않아요.")

            # ── 상관관계 분석 ───────────────────────────────────
            st.markdown("---")
            st.subheader("🔗 상관관계 분석")

            def plot_corr(x_col, y_col, x_label, y_label, title):
                if x_col not in growth_df.columns or y_col not in growth_df.columns:
                    st.info(f"{title} — 필요한 열이 없어요.")
                    return
                tmp = growth_df[[x_col, y_col]].dropna()
                if len(tmp) < 2:
                    st.info(f"{title} — 데이터가 충분하지 않아요.")
                    return
                x = tmp[x_col].values
                y = tmp[y_col].values
                r, p = stats.pearsonr(x, y)
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept

                fig = make_subplots(rows=1, cols=1)
                fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="개체", marker=dict(size=8)))
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"추세선 (r={r:.2f})",
                    line=dict(color="red", dash="dash"),
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    f"상관계수 r = {r:.3f}  |  p = {p:.4f}  |  추세선: y = {slope:.3f}x + {intercept:.3f}"
                )

            plot_corr("잎 수", "생중량", "잎 수 (장)", "생중량 (g)", "잎 수 vs 생중량")
            plot_corr("지상부 길이", "생중량", "지상부 길이 (mm)", "생중량 (g)", "지상부 길이 vs 생중량")

            # ── 원본 테이블 + 다운로드 ─────────────────────────
            st.markdown("---")
            with st.expander("📁 생육 데이터 원본 테이블 + CSV 다운로드"):
                if not growth_df.empty:
                    st.dataframe(growth_df, width="stretch")
                    csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 생육 데이터 CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{school_name or '생육'}_결과데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("업로드된 생육 데이터가 없습니다.")
    except Exception as e:
        st.warning(f"⚠️ 생육 결과 표시 중 문제가 있었어요: {e}")
