import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import io

# 1. 網頁頁面與團隊資訊設定
st.set_page_config(page_title="IEYI AI語音防詐騙系統", page_icon="🛡️", layout="wide")

st.title("🛡️ AI 語音防詐騙即時偵測系統")
st.markdown("### 2026 IEYI 世界青少年發明展 - 參賽作品展示")
st.markdown("#### 團隊成員：林口康橋 范懿飛 George | 延平中學 范坤翔 Charles | 衛理女中 范瑀媗 Rose")

# 側邊欄：科學原理說明
with st.sidebar:
    st.header("🔬 技術偵測原理")
    st.info("""
    **本系統監測三大關鍵指標：**
    1. **ZCR (過零率)**：偵測頻率變化的隨機性。AI 語音通常變化率低於 0.115。
    2. **MFCC Var (音色變異數)**：分析聲音的諧波豐富度。AI 的音色指紋通常低於 10400。
    3. **RMS (能量)**：觀測聲音的物理動力。
    """)
    st.warning("⚠️ 提醒：現場環境吵雜時，建議使用外接麥克風以確保分析精準。")

# 核心分析功能
def process_audio(audio_bytes, title):
    if audio_bytes:
        # 讀取音訊
        audio_segment = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_segment, sr=16000)
        
        # --- 數據運算 ---
        # A. RMS 能量
        rms = np.mean(librosa.feature.rms(y=y))
        # B. ZCR 過零率
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        # C. MFCC 音色分析 (針對高階 AI 的關鍵特徵)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = np.var(mfccs) 

        # --- 最終優化判定邏輯 (針對 Deevid AI 全系列樣本校準) ---
        ai_score = 0
        
        # 門檻 1：ZCR 判定 (AI 樣本實測 0.088 vs 真人 0.127)
        if zcr < 0.115: 
            ai_score += 1
        
        # 門檻 2：MFCC Var 判定 (AI 樣本實測 9245 vs 真人 10943)
        if mfcc_var < 10400: 
            ai_score += 1
            
        # 門檻 3：極端特徵判定 (如果 ZCR 低於 0.095，通常是數位合成的鐵證)
        if zcr < 0.095:
            ai_score += 1

        # --- 結果顯示 ---
        st.markdown(f"### 🔍 分析來源: {title}")
        
        # 綜合評分判定
        if ai_score >= 2:
            st.error(f"🚨 偵測結果：高風險！可能是 AI 模擬語音 (AI 評分: {ai_score}/3)")
            st.write(f"【判定依據】系統偵測到音色豐富度較低 ({mfcc_var:.1f}) 且頻率變換率過低 ({zcr:.4f})，符合數位合成特徵。")
        else:
            st.success(f"✅ 偵測結果：極可能是真人語音 (AI 評分: {ai_score}/3)")
            st.write(f"【判定依據】聲波具備自然的動態範圍、諧波指紋豐富度以及自然的頻率變化。")

        # --- 數據儀表板 ---
        st.markdown("#### 📊 聲學關鍵指標數據庫")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMS (能量)", f"{rms:.5f}")
        c2.metric("ZCR (頻率變化率)", f"{zcr:.5f}")
        c3.metric("MFCC Var (音色豐富度)", f"{mfcc_var:.1f}")

        # --- 視覺化圖表 ---
        st.markdown("---")
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            fig1, ax1 = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax1, color='#1f77b4')
            ax1.set_title("Waveform (時間域波形 - 觀察能量起伏)")
            st.pyplot(fig1)
            
        with col_plot2:
            fig2, ax2 = plt.subplots()
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, ax=ax2, x_axis='time', y_axis='hz')
            ax2.set_title("Spectrogram (頻譜圖 - 觀察諧波指紋)")
            st.pyplot(fig2)

# 網頁介面導覽
tab1, tab2 = st.tabs(["🎙️ 現場錄音測試 (Live Record)", "📂 上傳音訊檔案 (Upload File)"])

with tab1:
    st.write("請點擊麥克風後開始說話（建議 3-5 秒），結束請再按一次麥克風：")
    recorded_audio = audio_recorder(
        text="點擊錄音",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_size="3x",
    )
    if recorded_audio:
        process_audio(recorded_audio, "現場錄音")

with tab2:
    uploaded_file = st.file_uploader("請選擇音訊檔案 (.wav / .mp3 / .m4a)", type=['wav', 'mp3', 'm4a'])
    if uploaded_file is not None:
        process_audio(uploaded_file.read(), "檔案分析")