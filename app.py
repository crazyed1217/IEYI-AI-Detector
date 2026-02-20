import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import io
import base64

# 1. 網頁頁面設定
st.set_page_config(page_title="IEYI AI語音偵測系統", page_icon="🛡️", layout="wide")

# --- 音效處理函式 ---
def play_sound(sound_type):
    # 使用 Base64 編碼播放內建音效網址 (使用公共音效庫確保穩定)
    if sound_type == "success":
        url = "https://www.soundjay.com/buttons/sounds/button-37.mp3" # 叮咚聲
    else:
        url = "https://www.soundjay.com/buttons/sounds/button-10.mp3" # 警報聲
    
    sound_html = f"""
        <audio autoplay>
            <source src="{url}" type="audio/mp3">
        </audio>
    """
    st.components.v1.html(sound_html, height=0)

# 2. 自定義 CSS 美化 (包含狼頭動畫與資訊卡)
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .result-card { padding: 30px; border-radius: 20px; margin-bottom: 25px; color: white; text-align: center; }
    .success-bg { background-color: #28a745; border: 5px solid #1e7e34; }
    .error-bg { background-color: #dc3545; border: 5px solid #a71d2a; animation: shake 0.5s infinite; }
    @keyframes shake { 0% { transform: translate(1px, 1px) rotate(0deg); } 10% { transform: translate(-1px, -2px) rotate(-1deg); } 20% { transform: translate(-3px, 0px) rotate(1deg); } 30% { transform: translate(3px, 2px) rotate(0deg); } 40% { transform: translate(1px, -1px) rotate(1deg); } 50% { transform: translate(-1px, 2px) rotate(-1deg); } }
    .wolf-icon { font-size: 100px; }
    .team-info { background: #1e3a8a; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# 3. 顯示學生資訊與學校
st.markdown("""
    <div class="team-info">
        <h2>🛡️ AI 語音防詐騙即時偵測系統</h2>
        <p style="font-size: 18px;">2026 IEYI 世界青少年發明展 - 參賽作品</p>
        <hr>
        <div style="display: flex; justify-content: space-around; font-size: 16px;">
            <div><b>林口康橋國際學校</b><br>范懿飛 George</div>
            <div><b>台北市私立延平中學</b><br>范坤翔 Charles</div>
            <div><b>新北市私立衛理女中</b><br>范瑀媗 Rose</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 核心分析功能
def process_audio(audio_bytes, title):
    if audio_bytes:
        audio_segment = io.BytesIO(audio_bytes)
        try:
            y, sr = librosa.load(audio_segment, sr=16000)
            duration = len(y) / sr
            
            if duration < 0.5:
                st.warning("⚠️ 錄音過短，請重新嘗試。")
                return
            
            # 聲學運算
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = np.var(mfccs) 

            # 判定邏輯
            ai_score = 0
            if zcr < 0.115: ai_score += 1
            if mfcc_var < 10400: ai_score += 1
            if zcr < 0.095: ai_score += 1

            # 4. 判定結果與音效產生
            if ai_score >= 2:
                play_sound("error") # 播放警報音效
                st.markdown(f"""
                <div class="result-card error-bg">
                    <div class="wolf-icon">🐺</div>
                    <h1>🚨 警報：偵測到偽造語音！</h1>
                    <p style="font-size: 22px;">AI 判定分數：{ai_score}/3 (高風險詐騙)</p>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("🔍 點擊查看技術分析依據"):
                    st.write(f"1. 音訊長度：{duration:.2f} 秒")
                    st.write(f"2. 判定理由：偵測到數位合成頻率特徵與低變異音色指紋。")
            else:
                play_sound("success") # 播放成功音效
                st.markdown(f"""
                <div class="result-card success-bg">
                    <div style="font-size: 100px;">🛡️</div>
                    <h1>✅ 安全：確認為真人語音</h1>
                    <p style="font-size: 22px;">判定結果：符合自然人聲特徵</p>
                </div>
                """, unsafe_allow_html=True)

            # 5. 數據指標
            st.markdown("### 📊 聲學關鍵指標")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMS (能量強度)", f"{rms:.4f}")
            c2.metric("ZCR (頻率隨機性)", f"{zcr:.4f}")
            c3.metric("MFCC Var (音色豐富度)", f"{mfcc_var:.1f}")

            # 6. 視覺化圖表
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#007bff')
                ax1.set_title("時間域波形 (Waveform)")
                st.pyplot(fig1)
            with col2:
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr, ax=ax2, x_axis='time', y_axis='hz')
                ax2.set_title("頻譜圖特徵 (Spectrogram)")
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"分析錯誤: {e}")

# 分頁區
tab1, tab2 = st.tabs(["🎙️ 現場錄音偵測", "📂 檔案上傳分析"])

with tab1:
    recorded_audio = audio_recorder(text="點擊開始錄音", recording_color="#dc3545", icon_size="3x")
    if recorded_audio:
        process_audio(recorded_audio, "現場錄音")

with tab2:
    uploaded_file = st.file_uploader("請選擇音訊檔案", type=['wav', 'mp3', 'm4a'])
    if uploaded_file:
        process_audio(uploaded_file.read(), "檔案上傳分析")