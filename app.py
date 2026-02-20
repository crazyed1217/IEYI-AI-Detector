import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import io

# 1. 網頁頁面設定 (設定瀏覽器分頁標題與圖示)
st.set_page_config(page_title="IEYI AI Voice Detector", page_icon="🛡️", layout="wide")

# --- 音效播放函式 ---
def play_audio_effect(is_ai):
    # 判定為 AI 時播放警報音，真人則播放叮咚聲
    sound_url = "https://www.soundjay.com/buttons/sounds/button-10.mp3" if is_ai else "https://www.soundjay.com/buttons/sounds/button-37.mp3"
    sound_html = f"""
        <audio autoplay>
            <source src="{sound_url}" type="audio/mp3">
        </audio>
    """
    st.components.v1.html(sound_html, height=0)

# 2. 自定義極致美化 CSS (包含動態狼頭與名牌樣式)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap');
    html, body, [data-testid="stSidebar"] { font-family: 'Noto Sans TC', sans-serif; }
    
    /* 團隊名牌樣式 */
    .team-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; padding: 30px; border-radius: 20px;
        text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        margin-bottom: 35px;
    }
    .member-box {
        display: inline-block; background: rgba(255,255,255,0.15);
        padding: 12px 25px; border-radius: 12px; margin: 8px;
        border: 1px solid rgba(255,255,255,0.4);
        line-height: 1.6;
    }

    /* 結果顯示卡片 */
    .result-container {
        padding: 40px; border-radius: 25px; text-align: center;
        margin: 25px 0; color: white; transition: 0.5s;
    }
    .safe-card {
        background: linear-gradient(145deg, #166534, #22c55e);
        box-shadow: 0 0 35px rgba(34, 197, 94, 0.4);
    }
    .warning-card {
        background: linear-gradient(145deg, #991b1b, #ef4444);
        box-shadow: 0 0 55px rgba(239, 68, 68, 0.7);
        animation: wolf-shake 0.4s infinite;
    }
    @keyframes wolf-shake {
        0% { transform: scale(1) rotate(0deg); }
        25% { transform: scale(1.03) rotate(-1deg); }
        75% { transform: scale(1.03) rotate(1deg); }
        100% { transform: scale(1) rotate(0deg); }
    }
    .wolf-head { font-size: 130px; filter: drop-shadow(0 0 15px black); margin-bottom: 10px; }
    
    /* 指標數據美化 */
    .stMetric { background: white; border-radius: 15px !important; box-shadow: 0 6px 12px rgba(0,0,0,0.08) !important; padding: 20px !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. 團隊名牌 (完整呈現學校全稱與學生姓名)
st.markdown("""
    <div class="team-header">
        <h1 style='margin-bottom:5px; font-size: 40px;'>🛡️ AI 語音防詐騙即時偵測系統</h1>
        <p style='opacity:0.9; font-size:20px; letter-spacing: 2px;'>2026 IEYI 世界青少年發明展 - 競賽展示版</p>
        <div style='margin-top:20px;'>
            <div class="member-box">🏫 <b>新北市私立林口康橋國際學校</b><br>范懿飛 George</div>
            <div class="member-box">🏫 <b>台北市私立延平中學</b><br>范坤翔 Charles</div>
            <div class="member-box">🏫 <b>台北市私立衛理女中</b><br>范瑀媗 Rose</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 核心分析功能
def process_audio(audio_bytes, title):
    if audio_bytes:
        audio_segment = io.BytesIO(audio_bytes)
        try:
            # 讀取音訊並計算秒數
            y, sr = librosa.load(audio_segment, sr=16000)
            duration = len(y) / sr
            
            if duration < 0.5:
                st.warning("⚠️ 錄音長度不足，請至少錄製 1 秒以上。")
                return

            # --- 聲學特徵計算 ---
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
            mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = np.var(mfcc_feat)

            # --- 多維度判定邏輯 (針對實測數據優化) ---
            ai_score = 0
            reasons = []
            if zcr < 0.115: 
                ai_score += 1
                reasons.append("頻率隨機性偏低 (ZCR 低於安全門檻)")
            if mfcc_var < 10400: 
                ai_score += 1
                reasons.append("音色特徵變異數不足 (MFCC 指紋過於單一)")
            if zcr < 0.095: 
                ai_score += 1
                reasons.append("偵測到明顯數位合成痕跡")

            # --- 結果顯示區 ---
            st.info(f"⏱️ **分析完成！音訊總長度：{duration:.2f} 秒**")
            
            if ai_score >= 2:
                play_audio_effect(True) # 播放 AI 警報音
                st.markdown(f"""
                <div class="result-container warning-card">
                    <div class="wolf-head">🐺</div>
                    <h1 style='font-size:48px; margin:0;'>DANGER: AI VOICE DETECTED</h1>
                    <p style='font-size:26px;'>偵測到高度詐騙風險！(判定得分: {ai_score}/3)</p>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("📝 詳細科學判定依據"):
                    st.write(f"系統偵測到以下異常特徵：**{', '.join(reasons)}**。")
                    st.write("這代表該音訊缺乏真人聲帶在說話時自然產生的『物理隨機性』與『諧波豐富度』。")
            else:
                play_audio_effect(False) # 播放真人成功音
                st.markdown(f"""
                <div class="result-container safe-card">
                    <div style="font-size:110px;">🛡️</div>
                    <h1 style='font-size:48px; margin:0;'>SAFE: HUMAN VOICE</h1>
                    <p style='font-size:26px;'>判定為真實人聲。未偵測到數位合成跡象。</p>
                </div>
                """, unsafe_allow_html=True)

            # --- 數據儀表板 ---
            st.markdown("### 📊 關鍵聲學數據庫")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMS (能量強度)", f"{rms:.4f}")
            c2.metric("ZCR (頻率隨機性)", f"{zcr:.4f}")
            c3.metric("MFCC Var (音色豐富度)", f"{mfcc_var:.1f}")

            # --- 視覺化圖表 ---
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("📈 **Time Domain (時間域波形 - 能量分佈)**")
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax, color='#3b82f6')
                ax.set_axis_off()
                st.pyplot(fig)
            with col_b:
                st.caption("🌈 **Spectrogram (頻譜圖 - 諧波指紋)**")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(S, sr=sr, ax=ax2, x_axis='time', y_axis='hz', cmap='magma')
                ax2.set_axis_off()
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"分析發生不可預期的錯誤: {e}")

# 5. 分頁主介面
tab_rec, tab_file = st.tabs(["🎙️ 現場偵測 (Live Record)", "📂 檔案分析 (Upload File)"])

with tab_rec:
    st.write("請點擊麥克風後開始說話（建議錄製 3-5 秒）：")
    audio_data = audio_recorder(text="", recording_color="#ef4444", icon_size="3x")
    if audio_data:
        process_audio(audio_data, "現場錄音")

with tab_file:
    file = st.file_uploader("選擇欲分析的音訊檔案 (.wav/mp3/m4a)", type=['wav','mp3','m4a'])
    if file:
        process_audio(file.read(), "檔案上傳分析")