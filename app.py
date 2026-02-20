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
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .result-card { padding: 30px; border-radius: 20px; margin-bottom: 25px; color: white; text-align: center; }
    .success-bg { background-color: #28a745; border: 5px solid #1e7e34; }
    .error-bg { background-color: #dc3545; border: 5px solid #a71d2a; animation: pulse 2s infinite; }
    @keyframes pulse { 0% {box-shadow: 0 0 0 0px rgba(220, 53, 69, 0.7);} 70% {box-shadow: 0 0 0 20px rgba(220, 53, 69, 0);} 100% {box-shadow: 0 0 0 0px rgba(220, 53, 69, 0);} }
    .wolf-icon { font-size: 80px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 3. 標題與團隊資訊
st.title("🛡️ AI 語音防詐騙即時偵測系統")
st.markdown("##### 2026 IEYI 世界青少年發明展 | George, Charles, Rose 聯合研發")
st.markdown("---")

# 核心分析功能
def process_audio(audio_bytes, title):
    if audio_bytes:
        audio_segment = io.BytesIO(audio_bytes)
        try:
            # 讀取音訊並計算秒數
            y, sr = librosa.load(audio_segment, sr=16000)
            duration = len(y) / sr
            
            if duration < 0.5:
                st.warning("⚠️ 錄音過短，請至少錄製 1 秒。")
                return
            
            # 顯示秒數資訊
            st.write(f"⏱️ **偵測音訊長度：{duration:.2f} 秒**")
            
            # 聲學運算
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = np.var(mfccs) 

            # IEYI 現場穩定版邏輯
            ai_score = 0
            reasons = []
            
            if zcr < 0.115: 
                ai_score += 1
                reasons.append("頻率變化過於平滑 (ZCR)")
            if mfcc_var < 10400: 
                ai_score += 1
                reasons.append("音色特徵單一 (MFCC)")
            if zcr < 0.095: 
                ai_score += 1
                reasons.append("數位合成痕跡明顯")

            # 4. 判定結果視覺化 (加入狼頭與驚悚效果)
            if ai_score >= 2:
                st.markdown(f"""
                <div class="result-card error-bg">
                    <div class="wolf-icon">🐺</div>
                    <h2>🚨 警報：偵測到偽造語音！ (得分: {ai_score}/3)</h2>
                    <p style="font-size: 20px;">這段音訊極可能是由 AI 合成，並非真人說話。</p>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📝 為什麼系統判定這是 AI？"):
                    st.write(f"系統分析發現此音訊具備以下特徵：**{', '.join(reasons)}**。")
                    st.write("這代表聲音缺乏真人說話時喉嚨共振產生的隨機性與豐富度。")
            else:
                st.markdown(f"""
                <div class="result-card success-bg">
                    <div style="font-size: 80px;">🛡️</div>
                    <h2>✅ 偵測通過：確認為真人語音</h2>
                    <p style="font-size: 20px;">音訊具備自然的諧波與頻率隨機性。</p>
                </div>
                """, unsafe_allow_html=True)

            # 5. 數據儀表板
            st.markdown("### 📊 科學分析數據")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMS (能量強度)", f"{rms:.4f}")
            c2.metric("ZCR (頻率隨機性)", f"{zcr:.4f}")
            c3.metric("MFCC Var (音色豐富度)", f"{mfcc_var:.1f}")

            # 6. 視覺化圖表
            st.markdown("---")
            col_plot1, col_plot2 = st.columns(2)
            with col_plot1:
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax1, color='#007bff')
                ax1.set_title("Waveform (觀察能量波動)")
                st.pyplot(fig1)
            with col_plot2:
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr, ax=ax2, x_axis='time', y_axis='hz')
                ax2.set_title("Spectrogram (觀察諧波指紋)")
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"分析失敗。錯誤碼: {e}")

# 分頁區
tab1, tab2 = st.tabs(["🎙️ 現場測試 (Live)", "📂 檔案上傳 (Upload)"])

with tab1:
    recorded_audio = audio_recorder(text="點擊麥克風開始錄音", recording_color="#dc3545", icon_size="3x")
    if recorded_audio:
        process_audio(recorded_audio, "現場錄音")

with tab2:
    uploaded_file = st.file_uploader("請選擇音訊檔案", type=['wav', 'mp3', 'm4a'])
    if uploaded_file:
        process_audio(uploaded_file.read(), "檔案上傳")