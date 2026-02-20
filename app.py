import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import io

# 1. 網頁頁面設定
st.set_page_config(page_title="IEYI AI Voice Detector", page_icon="🛡️", layout="wide")

# --- 音效播放函式 ---
def play_audio_effect(is_ai):
    # 使用公共音效庫：叮咚聲 (Success) vs 警報聲 (Alarm)
    sound_url = "https://www.soundjay.com/buttons/sounds/button-10.mp3" if is_ai else "https://www.soundjay.com/buttons/sounds/button-37.mp3"
    sound_html = f"""
        <audio autoplay>
            <source src="{sound_url}" type="audio/mp3">
        </audio>
    """
    st.components.v1.html(sound_html, height=0)

# 2. 自定義極致美化 CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;700&display=swap');
    html, body, [data-testid="stSidebar"] { font-family: 'Noto Sans TC', sans-serif; }
    
    /* 團隊名牌樣式 */
    .team-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white; padding: 25px; border-radius: 20px;
        text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        margin-bottom: 30px;
    }
    .member-box {
        display: inline-block; background: rgba(255,255,255,0.1);
        padding: 10px 20px; border-radius: 10px; margin: 5px;
        border: 1px solid rgba(255,255,255,0.3);
    }

    /* 結果卡片樣式 */
    .result-container {
        padding: 40px; border-radius: 25px; text-align: center;
        margin: 20px 0; color: white; transition: 0.5s;
    }
    .safe-card {
        background: linear-gradient(145deg, #166534, #22c55e);
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.4);
    }
    .warning-card {
        background: linear-gradient(145deg, #991b1b, #ef4444);
        box-shadow: 0 0 50px rgba(239, 68, 68, 0.6);
        animation: wolf-shake 0.5s infinite;
    }
    @keyframes wolf-shake {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .wolf-head { font-size: 120px; filter: drop-shadow(0 0 10px black); }
    
    /* 數據顯示樣式 */
    .stMetric { background: white; border-radius: 15px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. 團隊名牌 (學校 + 姓名)
st.markdown("""
    <div class="team-header">
        <h1 style='margin-bottom:0;'>🛡️ AI 語音防詐騙即時偵測系統</h1>
        <p style='opacity:0.9; font-size:18px;'>2026 IEYI 世界青少年發明展 - 參賽作品展示</p>
        <div style='margin-top:15px;'>
            <div class="member-box">🏫 <b>林口康橋國際學校</b><br>范懿飛 George</div>
            <div class="member-box">🏫 <b>私立延平中學</b><br>范坤翔 Charles</div>
            <div class="member-box">🏫 <b>私立衛理女中</b><br>范瑀媗 Rose</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 核心分析函式
def process_audio(audio_bytes, title):
    if audio_bytes:
        audio_segment = io.BytesIO(audio_bytes)
        try:
            y, sr = librosa.load(audio_segment, sr=16000)
            duration = len(y) / sr
            
            if duration < 0.5:
                st.warning("⚠️ 錄音過短，請再試一次。")
                return

            # 計算特徵
            rms = np.mean(librosa.feature.rms(y=y))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_var = np.var(mfcc)

            # 判定邏輯
            ai_score = 0
            reasons = []
            if zcr < 0.115: 
                ai_score += 1
                reasons.append("頻率隨機性偏低 (ZCR 低於門檻)")
            if mfcc_var < 10400: 
                ai_score += 1
                reasons.append("音色特徵單一 (MFCC 變異數不足)")
            if zcr < 0.095: 
                ai_score += 1
                reasons.append("偵測到明顯數位合成痕跡")

            # --- 結果顯示區 ---
            st.write(f"⏱️ **分析音訊長度：{duration:.2f} 秒**")
            
            if ai_score >= 2:
                play_audio_effect(True) # 播放警報音
                st.markdown(f"""
                <div class="result-container warning-card">
                    <div class="wolf-head">🐺</div>
                    <h1 style='font-size:45px; margin:0;'>DANGER: AI VOICE DETECTED</h1>
                    <p style='font-size:24px;'>偵測到高度詐騙風險！(AI 指標得分: {ai_score}/3)</p>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("📝 詳細判定依據"):
                    st.write(f"系統偵測到以下異常：**{', '.join(reasons)}**。這種特徵常見於 AI 模擬出的『平滑』語音，缺乏真人聲帶物理震動的豐富性。")
            else:
                play_audio_effect(False) # 播放叮咚聲
                st.markdown(f"""
                <div class="result-container safe-card">
                    <div style="font-size:100px;">🛡️</div>
                    <h1 style='font-size:45px; margin:0;'>SAFE: HUMAN VOICE</h1>
                    <p style='font-size:24px;'>判定為真實人聲。未偵測到數位偽造痕跡。</p>
                </div>
                """, unsafe_allow_html=True)

            # --- 數據儀表板 ---
            st.markdown("### 📊 科學分析數據指標")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMS (能量強度)", f"{rms:.4f}")
            c2.metric("ZCR (頻率隨機性)", f"{zcr:.4f}")
            c3.metric("MFCC Var (音色豐富度)", f"{mfcc_var:.1f}")

            # --- 圖表區 ---
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("📈 **Waveform (時間域波形)**")
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax, color='#3b82f6')
                ax.set_axis_off()
                st.pyplot(fig)
            with col_b:
                st.caption("🌈 **Spectrogram (頻譜圖特徵)**")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(S, sr=sr, ax=ax2, x_axis='time', y_axis='hz', cmap='magma')
                ax2.set_axis_off()
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"分析失敗: {e}")

# 分頁介面
tab_rec, tab_file = st.tabs(["🎙️ 現場錄音分析", "📂 檔案上傳分析"])

with tab_rec:
    st.write("請點擊麥克風開始錄製 3-5 秒內容：")
    audio_data = audio_recorder(text="", recording_color="#ef4444", icon_size="3x")
    if audio_data:
        process_audio(audio_data, "現場錄音")

with tab_file:
    file = st.file_uploader("選擇音訊檔案 (.wav/mp3)", type=['wav','mp3','m4a'])
    if file:
        process_audio(file.read(), "上傳檔案")