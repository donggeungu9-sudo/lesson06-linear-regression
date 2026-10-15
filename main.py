import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균 기온 분석 및 예측기")
st.markdown(
    "서울 기온 데이터를 바탕으로 연도별 평균기온 추세를 살펴보고, "
    "전체 기간과 최근 20년의 기온 상승 속도를 비교해 보세요."
)

# 데이터 로드 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    try:
        df = pd.read_csv(url, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(url, encoding="cp949")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 데이터 전처리
df.columns = df.columns.str.strip()
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["연도"] = df["날짜"].dt.year

# 2025년까지만 필터링 (수업 기준 기간)
df = df[df["연도"] <= 2025]
df_clean = df.dropna(subset=["평균기온"])

# 연도별 데이터 수(관측일수)와 연평균기온 계산
yearly_stats = df_clean.groupby("연도").agg(
    관측일수=("평균기온", "count"),
    연평균기온=("평균기온", "mean")
).reset_index()

# 관측일이 300일 이상인 해만 필터링
filtered_yearly = yearly_stats[yearly_stats["관측일수"] >= 300].copy()

if filtered_yearly.empty:
    st.warning("조건을 만족하는 연도별 데이터가 없습니다.")
    st.stop()

# --- 1. 전체 기간 분석 ---
years_all = filtered_yearly["연도"].values
temps_all = filtered_yearly["연평균기온"].values

slope_all, intercept_all = np.polyfit(years_all, temps_all, 1)
correlation_all = np.corrcoef(years_all, temps_all)[0, 1]

# 100년당 상승 온도 (기울기 * 100)
rate_100y_all = slope_all * 100

# --- 2. 최근 20년 분석 ---
max_year = int(years_all.max())
min_year_recent = max_year - 19  # 최근 20년 (예: max가 2025년이면 2006~2025년)

recent_yearly = filtered_yearly[filtered_yearly["연도"] >= min_year_recent]
years_recent = recent_yearly["연도"].values
temps_recent = recent_yearly["연평균기온"].values

slope_recent, intercept_recent = np.polyfit(years_recent, temps_recent, 1)
correlation_recent = np.corrcoef(years_recent, temps_recent)[0, 1]

rate_100y_recent = slope_recent * 100


# 화면 상단 요약 정보 표시
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("사용한 해의 개수", f"{len(filtered_yearly)}개")
with col2:
    st.metric("시작 연도", f"{int(years_all.min())}년")
with col3:
    st.metric("끝 연도", f"{max_year}년")
with col4:
    st.metric("전체 상관계수 (r)", f"{correlation_all:.2f}")
st.markdown("---")


# --- 기울기 비교 섹션 (전체 vs 최근 20년) ---
st.subheader("📈 기온 상승 속도 비교 (100년 기준)")
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
            <h4 style="margin: 0; color: #31333F;">전체 기간 ({int(years_all.min())}~{max_year}년)</h4>
            <p style="font-size: 38px; font-weight: bold; color: #1f77b4; margin: 10px 0 0 0;">
                +{rate_100y_all:.2f} °C / 100년
            </p>
            <p style="font-size: 14px; color: gray; margin-top: 5px;">연간 증가율: +{slope_all:.4f} °C/년</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with comp_col2:
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #fff3f3; text-align: center;">
            <h4 style="margin: 0; color: #31333F;">최근 20년 ({min_year_recent}~{max_year}년)</h4>
            <p style="font-size: 38px; font-weight: bold; color: #ff4b4b; margin: 10px 0 0 0;">
                +{rate_100y_recent:.2f} °C / 100년
            </p>
            <p style="font-size: 14px; color: gray; margin-top: 5px;">연간 증가율: +{slope_recent:.4f} °C/년</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("")


# Plotly 그래프 생성 (전체 추세선 표시)
filtered_yearly["추세선_전체"] = slope_all * years_all + intercept_all

fig = px.scatter(
    filtered_yearly,
    x="연도",
    y="연평균기온",
    title="서울 연도별 평균기온 추이 및 전체 회귀 직선",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
    template="plotly_white"
)

# 전체 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=filtered_yearly["연도"],
        y=filtered_yearly["추세선_전체"],
        mode="lines",
        name=f"전체 회귀 직선 (100년당 +{rate_100y_all:.2f}°C)",
        line=dict(color="blue", width=2)
    )
)

fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)


# 슬라이더를 통한 연도별 예상 기온 확인 (전체 회귀 기준)
st.subheader("🔮 연도별 예상 기온 확인하기")
selected_year = st.slider(
    "조회할 연도를 선택하세요 (1900 ~ 2100)",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

predicted_temp = slope_all * selected_year + intercept_all

st.markdown(
    f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
        <h3 style="margin: 0; color: #31333F;">{selected_year}년 서울 예상 연평균기온 (전체 회귀 모델 기준)</h3>
        <p style="font-size: 48px; font-weight: bold; color: #ff4b4b; margin: 10px 0 0 0;">
            {predicted_temp:.2f} °C
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
