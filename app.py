"""
HKDSE 數學全方位智能練習網
============================
單一檔案 Streamlit 應用程式 (app.py)
使用 Google Gemini API 智能生成 DSE 數學練習題

前置準備：
  1. pip install streamlit google-generativeai
  2. 建立 .streamlit/secrets.toml，內容：
     GEMINI_API_KEY = "你的-Gemini-API-Key"
  3. streamlit run app.py
"""

import streamlit as st
import google.generativeai as genai
import json
import re

# ============================================================
# 1. 頁面基本設定
# ============================================================
st.set_page_config(
    page_title="HKDSE 數學全方位智能練習網",
    page_icon="📐",
    layout="wide",
)

# ============================================================
# 2. 自訂 CSS 美化介面
# ============================================================
st.markdown(
    """
    <style>
    /* 限制主內容區寬度，提升閱讀體驗 */
    .block-container {
        max-width: 920px;
        padding-top: 1.5rem;
    }

    /* 主標題漸層色 */
    .hero-title {
        text-align: center;
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #9333EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .hero-sub {
        text-align: center;
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }

    /* 歡迎區塊 */
    .welcome-box {
        text-align: center;
        padding: 4rem 1rem 2rem;
    }
    .welcome-box .icon { font-size: 4rem; margin-bottom: 0.8rem; }
    .welcome-box h3 { color: #1F2937; }
    .welcome-box p  { color: #6B7280; font-size: 1.05rem; line-height: 1.8; }

    /* 側邊欄底部小字 */
    .sidebar-footer {
        color: #9CA3AF;
        font-size: 0.78rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. 安全讀取 Gemini API Key（從 Streamlit Secrets）
# ============================================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except (KeyError, FileNotFoundError):
    st.error("⚠️ 系統設定中 (Missing Secrets)，請聯絡老師")
    st.stop()

# ============================================================
# 4. Session State 初始化
#    —— 確保按鈕互動或頁面重繪時，題目與顯示狀態不會消失
# ============================================================
STATE_DEFAULTS: dict = {
    "current_question": None,    # 目前題目文字
    "current_hint": None,        # 目前提示文字
    "current_solution": None,    # 目前詳解文字
    "show_hint": False,          # 是否顯示提示
    "show_solution": False,      # 是否顯示詳解
    "display_section": "",       # 題目所屬部份（供主畫面標題顯示）
    "display_topic": "",         # 題目所屬課題（供主畫面標題顯示）
}
for key, default_value in STATE_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ============================================================
# 5. 選項常數
# ============================================================
SECTIONS: list[str] = [
    "甲部(一) Section A1",
    "甲部(二) Section A2",
    "乙部 Section B",
]

TOPICS: list[str] = [
    "基礎代數與百分數",
    "幾何與坐標 (Geometry)",
    "統計學 (Statistics)",
    "多項式與變分",
    "圓的性質",
    "等差與等比數列 (AS/GS)",
    "三角學 (2D/3D)",
    "概率 (Probability)",
]

# ============================================================
# 6. Gemini System Prompt — DSE 出題規則
# ============================================================
SYSTEM_PROMPT: str = """\
你是一位經驗豐富的香港中學文憑試 (HKDSE) 數學科出題老師。
請根據使用者選擇的「試卷部份 (Section)」和「課題 (Topic)」出一道全新的數學練習題。

■ 難度與題型規則：

1. 若選「甲部(一) Section A1」：
   - 難度：基礎。題目簡短，步驟少。
   - 常見題型：簡易百分數運算、基礎代數化簡 / 方程、
     簡易坐標幾何（距離 / 斜率 / 中點）、基礎統計（平均值 / 中位數 / 眾數）。
   - 配分：約 3–4 分。

2. 若選「甲部(二) Section A2」：
   - 難度：進階。需要較多步驟或概念結合。
   - 常見題型：多項式除法與因式分解、變分 (variation)、
     圓的幾何性質（圓心角 / 弧 / 切線）、對數 (logarithm)、圖像變換 (transformation)。
   - 配分：約 5–7 分。

3. 若選「乙部 Section B」：
   - 難度：高階 / 複雜。需要綜合應用多個概念。
   - 常見題型：3D 三角學（角度 / 最短距離）、等差等比數列與級數 (AS/GS)、
     複雜概率（排列組合 nCr / nPr、條件概率）、圓的方程與切線。
   - 配分：約 10–12 分。必須生成包含 (a)、(b) 甚至 (c) 子題的結構。

■ 輸出格式（嚴格 JSON）：

回傳一個 JSON 物件，包含以下三個欄位：
{
  "question": "題目內容",
  "hint": "解題提示（僅提供思考方向，不直接給出答案）",
  "solution": "完整的逐步解題過程與最終答案"
}

■ 重要注意事項：
- 全部文字使用繁體中文。
- 數學公式使用 LaTeX 語法：行內公式用 $...$ 包裹，獨立公式用 $$...$$ 包裹。
- 題目風格必須貼近 DSE 真實試卷用語（如「化簡」「求⋯的值」「以 surd form 表示」
  「證明」「Express ... in terms of ...」等中英夾雜風格）。
- 每次必須生成全新且不重複的題目，題目數值也要有變化。
"""

# ============================================================
# 7. Gemini API 呼叫函式
# ============================================================
def call_gemini(section: str, topic: str) -> dict:
    """
    呼叫 Gemini API 生成一道 DSE 數學練習題。

    Args:
        section: 試卷部份，例如 '甲部(一) Section A1'
        topic:   課題名稱，例如 '三角學 (2D/3D)'

    Returns:
        dict 包含 "question", "hint", "solution" 三個鍵。
    """
    model = genai.GenerativeModel(
        model_name="gemini-pro",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.9,
            response_mime_type="application/json",   # 強制 JSON 輸出
        ),
    )

    user_message = (
        f"請出一道全新的 DSE 數學練習題：\n"
        f"- 試卷部份 (Section)：{section}\n"
        f"- 課題 (Topic)：{topic}\n"
    )

    response = model.generate_content(user_message)
    raw_text: str = response.text.strip()

    # 安全清理：若 API 回傳仍帶有 Markdown code block 標記
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
        raw_text = re.sub(r"\n?\s*```$", "", raw_text)

    return json.loads(raw_text)

# ============================================================
# 8. 側邊欄 UI
# ============================================================
with st.sidebar:
    st.header("📐 DSE 數學練習設定")
    st.divider()

    # 試卷部份選單
    section = st.selectbox(
        "📋 試卷部份 (Section)",
        SECTIONS,
        help="甲部(一) 最簡單，乙部最深",
    )

    # 課題選單
    topic = st.selectbox(
        "📚 課題選擇 (Topic)",
        TOPICS,
        help="選擇你想練習的數學範疇",
    )

    st.divider()

    # 生成按鈕
    generate_btn = st.button(
        "🔄 生成新題目",
        use_container_width=True,
        type="primary",
    )

    # 側邊欄底部資訊
    st.divider()
    st.markdown(
        '<p class="sidebar-footer">Powered by Google Gemini<br>Built with Streamlit</p>',
        unsafe_allow_html=True,
    )

# ============================================================
# 9. 處理「生成新題目」按鈕事件
# ============================================================
if generate_btn:
    with st.spinner("🤖 AI 老師正在出題，請稍候……"):
        try:
            data = call_gemini(section, topic)

            # 將結果存入 Session State
            st.session_state.current_question = data.get(
                "question", "（題目生成失敗，請重試）"
            )
            st.session_state.current_hint = data.get(
                "hint", "（提示未生成）"
            )
            st.session_state.current_solution = data.get(
                "solution", "（詳解未生成）"
            )
            # 重設顯示狀態
            st.session_state.show_hint = False
            st.session_state.show_solution = False
            # 記錄當前題目的部份與課題
            st.session_state.display_section = section
            st.session_state.display_topic = topic

        except json.JSONDecodeError:
            st.error("❌ AI 回應格式異常，請再按一次「🔄 生成新題目」重試。")
        except Exception as e:
            st.error(f"❌ 生成題目時發生錯誤：{e}")

# ============================================================
# 10. 主畫面
# ============================================================

# ----- 頁首標題 -----
st.markdown(
    '<h1 class="hero-title">📐 HKDSE 數學全方位智能練習網</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">根據 DSE 考試大綱 · AI 智能出題 · 助你輕鬆備戰</p>',
    unsafe_allow_html=True,
)
st.divider()

# ----- 判斷是否已有題目 -----
if st.session_state.current_question is not None:

    # 當前練習標題
    st.subheader(
        f"📝 當前練習：{st.session_state.display_section} — "
        f"{st.session_state.display_topic}"
    )
    st.write("")  # 小間距

    # ---- 題目區 ----
    st.markdown("#### 📖 題目")
    with st.container(border=True):
        st.markdown(st.session_state.current_question)

    st.write("")  # 小間距

    # ---- 作答區 ----
    st.markdown("#### ✏️ 你的作答")
    st.text_area(
        label="answer_input",
        height=150,
        placeholder="在此輸入你的解題過程和答案……",
        label_visibility="collapsed",
    )

    st.divider()

    # ---- 互動按鈕（並排）----
    col_hint, col_solution = st.columns(2)

    with col_hint:
        if st.button("💡 提示 (Hint)", use_container_width=True):
            st.session_state.show_hint = True

    with col_solution:
        if st.button("✅ 核對答案 (Solution)", use_container_width=True):
            st.session_state.show_solution = True

    # ---- 顯示提示 ----
    if st.session_state.show_hint and st.session_state.current_hint:
        st.info(f"💡 **解題提示**\n\n{st.session_state.current_hint}")

    # ---- 顯示詳解 ----
    if st.session_state.show_solution and st.session_state.current_solution:
        st.success(f"✅ **完整解答**\n\n{st.session_state.current_solution}")

else:
    # ----- 尚未生成題目：歡迎畫面 -----
    st.markdown(
        """
        <div class="welcome-box">
            <div class="icon">🎯</div>
            <h3>歡迎使用 HKDSE 數學智能練習網！</h3>
            <p>
                請在左側邊欄選擇<b>試卷部份</b>和<b>課題</b>，<br>
                然後按下「🔄 生成新題目」即可開始練習。
            </p>
        </div>
        """,
        unsafe_allow_html=True,

    )
