import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import io

# 1. 網頁頁面設定
st.set_page_config(page_title="AI語音偵測系統", page_icon="🛡️", layout="wide")

# 2. 自定義 CSS 美化
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white;
    }
    .success-bg { background-color: #28a745; }
    .error-bg { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# 3. 標題與團隊資訊
st.title("🛡️ AI 語音防詐騙即時偵測系統")
st.markdown("##### 2026 IEYI 世界青少年發明展 | 技術展示版")

# 團隊成員介紹
col_team1, col_team2, col_team3 = st.columns(3)
col_team1.caption("林口康橋 范懿飛 George")
col_team2.caption("延平中學 范坤翔 Charles")
col_team3.caption("衛理女中 范瑀媗 Rose")
st.markdown("---")

# 核心分析功能
def process_audio(audio_bytes, title):
    if audio_bytes:
        audio_segment = io.BytesIO(audio_bytes)
        try:
            y, sr = librosa.load(audio_segment, sr=16000)
            
            if len(y) < 1024:
                st.warning("⚠️ 錄音過短，請再試一次。")
                return
            
            # 聲學運算
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = np.var(mfccs) 

            # IEYI 現場穩定版邏輯
            ai_score = 0
            if zcr < 0.085: ai_score += 1
            if mfcc_var < 10400: ai_score += 1
            if zcr < 0.10 and mfcc_var < 10800: ai_score += 1

            # 4. 判定結果美化顯示
            if ai_score >= 2:
                st.markdown(f"""
                <div class="result-card error-bg">
                    <h2>🚨 偵測結果：高風險 AI 語音 (評分: {ai_score}/3)</h2>
                    <p>偵測到數位合成特徵，請警惕該音訊來源。</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card success-bg">
                    <h2>✅ 偵測結果：安全真人語音 (評分: {ai_score}/3)</h2>
                    <p>聲波具備自然人聲諧波，未偵測到數位合成痕跡。</p>
                </div>
                """, unsafe_allow_html=True)

            # 5. 數據儀表板 (Metric Cards)
            st.markdown("### 📊 關鍵聲學數據指標")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMS 能量起伏", f"{rms:.4f}")
            c2.metric("ZCR 頻率隨機性", f"{zcr:.4f}")
            c3.metric("MFCC 音色指紋", f"{mfcc_var:.1f}")

            # 6. 視覺化圖表美化
            st.markdown("---")
            col_plot1, col_plot2 = st.columns(2)
            
            with col_plot1:
                st.write("📈 **時間域波形 (Waveform)**")
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#007bff')
                ax1.set_axis_off()
                st.pyplot(fig1)
                
            with col_plot2:
                st.write("🌈 **頻譜圖特徵 (Spectrogram)**")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr, ax=ax2, x_axis='time', y_axis='hz')
                ax2.set_axis_off()
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"分析失敗，請重試。錯誤碼: {e}")

# 分頁區
tab1, tab2 = st.tabs(["🎙️ 現場偵測", "📂 檔案上傳"])

with tab1:
    st.write("請點擊麥克風並開始說話：")
    recorded_audio = audio_recorder(text="", recording_color="#dc3545", neutral_color="#6c757d", icon_size="3x")
    if recorded_audio:
        process_audio(recorded_audio, "現場測試")

with tab2:
    uploaded_file = st.file_uploader("上傳 .wav 或 .mp3 檔案", type=['wav', 'mp3'])
    if uploaded_file:
        process_audio(uploaded_file.read(), "檔案分析")