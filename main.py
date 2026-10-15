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
    "선형 회귀 모델을 통해 특정 연도의 예상 기온을 확인해보세요."
)

# 데이터 로드 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    # 인코딩 오류 방지를 위해 utf-8 또는 cp949 시도
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
# 열 이름 공백 제거
df.columns = df.columns.str.strip()

# 날짜를 datetime 형식으로 변환
df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["연도"] = df["날짜"].dt.year

# 2025년까지만 필터링 (수업 기준 기간)
df = df[df["연도"] <= 2025]

# 결측치 제거 (평균기온 기준)
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

# 회귀 분석 수행 (Numpy polyfit 사용, 1차 직선)
years = filtered_yearly["연도"].values
temps = filtered_yearly["연평균기온"].values

slope, intercept = np.polyfit(years, temps, 1)

# 상관계수 계산
correlation = np.corrcoef(years, temps)[0, 1]

# 화면 상단 요약 정보 표시
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("사용한 해의 개수", f"{len(filtered_yearly)}개")
with col2:
    st.metric("시작 연도", f"{int(years.min())}년")
with col3:
    st.metric("끝 연도", f"{int(years.max())}년")
with col4:
    st.metric("상관계수 (r)", f"{correlation:.2f}")
st.markdown("---")

# Plotly 그래프 생성
filtered_yearly["추세선"] = slope * years + intercept

fig = px.scatter(
    filtered_yearly,
    x="연도",
    y="연평균기온",
    title="서울 연도별 평균기온 추이 및 회귀 직선",
    labels={"연도": "연도", "연평균기온": "연평균기온 (°C)"},
    template="plotly_white"
)

# 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=filtered_yearly["연도"],
        y=filtered_yearly["추세선"],
        mode="lines",
        name=f"회귀 직선 (y = {slope:.4f}x + {intercept:.2f})",
        line=dict(color="red", width=2)
    )
)

fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# 슬라이더를 통한 연도별 예상 기온 확인
st.subheader("🔮 연도별 예상 기온 확인하기")
selected_year = st.slider(
    "조회할 연도를 선택하세요 (1900 ~ 2100)",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

predicted_temp = slope * selected_year + intercept

st.markdown(
    f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
        <h3 style="margin: 0; color: #31333F;">{selected_year}년 서울 예상 연평균기온</h3>
        <p style="font-size: 48px; font-weight: bold; color: #ff4b4b; margin: 10px 0 0 0;">
            {predicted_temp:.2f} °C
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
