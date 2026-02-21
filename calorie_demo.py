import streamlit as st
from PIL import Image
from openai import OpenAI
import base64
import json
import os
from datetime import date, datetime
import pandas as pd

# ====================== 設定 ======================
st.set_page_config(page_title="AI 食物熱量 Demo", layout="wide")
st.title("🍔 AI 食物熱量計算器 - 個人一天 Demo (使用 Groq API)")

# API Key (Groq)
api_key = st.sidebar.text_input("你的 Groq API Key", type="password", value="")
if api_key:
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

# 個人設定（存到 session）
if "settings" not in st.session_state:
    st.session_state.settings = {
        "daily_goal": 2000,
        "weight_kg": 70.0
    }

# 個人設定區塊（擴充版）
st.sidebar.subheader("個人資料（用來自動計算熱量目標）")

# 新增欄位
gender = st.sidebar.radio("性別", ["男", "女"], index=0)
age = st.sidebar.number_input("年齡 (歲)", min_value=10, max_value=100, value=30, step=1)
height_cm = st.sidebar.number_input("身高 (cm)", min_value=100, max_value=250, value=170, step=1)
current_weight_kg = st.sidebar.number_input("目前體重 (kg)", min_value=30.0, value=70.0, step=0.1, format="%.1f")

# 活動水平（下拉選單）
activity_level = st.sidebar.selectbox(
    "活動水平",
    [
        "久坐（辦公室，少運動）",
        "輕度活動（每周運動1-3天）",
        "中度活動（每周運動3-5天）",
        "重度活動（每周運動6-7天）",
        "極重度（體力勞動或專業運動員）"
    ],
    index=1  # 預設輕度
)

# 目標類型與期望變化
goal_type = st.sidebar.selectbox(
    "目標",
    ["維持體重", "減重", "增重"],
    index=0
)
weekly_change_kg = st.sidebar.number_input(
    "期望每週變化 (kg)",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.1,
    format="%.1f",
    help="減重建議 0.5-1kg/週，增重建議 0.25-0.5kg/週"
)

# 活動係數對應字典
activity_multipliers = {
    "久坐（辦公室，少運動）": 1.2,
    "輕度活動（每周運動1-3天）": 1.375,
    "中度活動（每周運動3-5天）": 1.55,
    "重度活動（每周運動6-7天）": 1.725,
    "極重度（體力勞動或專業運動員）": 1.9
}

# 計算 BMR
if gender == "男":
    bmr = 10 * current_weight_kg + 6.25 * height_cm - 5 * age + 5
else:
    bmr = 10 * current_weight_kg + 6.25 * height_cm - 5 * age - 161

# 計算 TDEE (維持熱量)
multiplier = activity_multipliers[activity_level]
tdee = bmr * multiplier

# 根據目標調整
calorie_adjustment = 0
if goal_type == "減重":
    calorie_adjustment = -weekly_change_kg * 7700 / 7  # 1kg ≈ 7700 kcal
elif goal_type == "增重":
    calorie_adjustment = weekly_change_kg * 7700 / 7

daily_goal = round(tdee + calorie_adjustment)

# 顯示結果
st.sidebar.markdown("### 自動計算結果")
st.sidebar.metric("估計基礎代謝率 (BMR)", f"{round(bmr)} kcal/天")
st.sidebar.metric("維持體重熱量 (TDEE)", f"{round(tdee)} kcal/天")
st.sidebar.metric("**建議每日熱量目標**", f"{daily_goal} kcal/天", delta=f"{calorie_adjustment:+.0f} kcal")

# 讓用戶可手動覆蓋自動值
manual_goal = st.sidebar.checkbox("手動設定目標（不使用自動計算）", value=False)
if manual_goal:
    st.session_state.settings["daily_goal"] = st.sidebar.number_input(
        "自訂每日熱量目標 (kcal)",
        value=daily_goal,
        step=50
    )
else:
    st.session_state.settings["daily_goal"] = daily_goal


# ====================== 資料儲存 ======================
DATA_FILE = "my_calorie_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

data = load_data()
today = str(date.today())
if today not in data:
    data[today] = {"meals": [], "total_calories": 0}
today_data = data[today]

# ====================== AI 分析 ======================
st.header("📸 上傳食物照片")

uploaded_file = st.file_uploader("選擇或拍攝食物照片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="你的食物", use_column_width=True)
    
    if st.button("🚀 讓 Groq AI 分析熱量"):
        if not api_key:
            st.error("請先在左側輸入 Groq API Key")
        else:
            with st.spinner("AI 正在分析中...（通常 5-15 秒）"):
                try:
                    base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                    
                    # 目前推薦的 Groq vision 模型（2026/2 可用）
                    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
                    # 如果不行，換成： "meta-llama/llama-4-maverick-17b-128e-instruct"
                    # 請去 https://console.groq.com/docs/vision 確認最新可用
                    
                    prompt = """
你是一位專業營養師。請仔細分析這張食物照片，並以嚴格的 JSON 格式輸出結果，不要添加任何額外文字、解釋或 markdown 符號。

輸出必須是純 JSON，格式如下：
{
  "foods": [
    {
      "name": "食物名稱（中文）",
      "quantity": "份量（如 1個、150g、一碗）",
      "weight_g": 估計克數（整數）,
      "calories": 熱量(kcal，整數),
      "protein_g": 蛋白質(g，整數或一位小數),
      "carbs_g": 碳水(g，整數或一位小數),
      "fat_g": 脂肪(g，整數或一位小數)
    }
  ],
  "total_calories": 總熱量（整數）,
  "notes": "額外觀察（可留空）"
}

請盡量估計真實份量，確保所有數值合理，並嚴格遵守 JSON 格式。
"""
                    
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                    }
                                ]
                            }
                        ],
                        max_tokens=1024,
                        temperature=0.2
                    )
                    
                    # 強制 UTF-8 處理
                    raw_content = response.choices[0].message.content
                    text = raw_content.encode('utf-8', errors='replace').decode('utf-8', errors='replace').strip()
                    
                    # 清理 markdown
                    if text.startswith("```json"):
                        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
                    elif text.startswith("```"):
                        text = text.split("```", 1)[1].split("```", 1)[0].strip()
                    
                    result = json.loads(text)
                    st.session_state.analysis = result
                    
                    st.success("✅ 分析完成！請確認後加入")
                    
                except json.JSONDecodeError as je:
                    st.error(f"JSON 解析失敗：{str(je)}")
                    st.info("AI 原始回應（供檢查）：")
                    st.code(text, language="json")
                except Exception as e:
                    st.error(f"分析出錯：{str(e)}")
                    if 'text' in locals():
                        st.info("原始回應文字：")
                        st.code(text)

# ====================== 確認與加入 ======================
if "analysis" in st.session_state:
    result = st.session_state.analysis
    
    st.subheader("AI 辨識結果（可直接編輯）")
    df = pd.DataFrame(result.get("foods", []))
    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    
    total_cal = int(edited_df["calories"].sum()) if not edited_df.empty else 0
    st.metric("總熱量", f"{total_cal} kcal")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 確認加入今日餐食", type="primary"):
            meal = {
                "time": datetime.now().strftime("%H:%M"),
                "foods": edited_df.to_dict("records"),
                "total_calories": total_cal
            }
            today_data["meals"].append(meal)
            today_data["total_calories"] = sum(m["total_calories"] for m in today_data["meals"])
            save_data(data)
            st.success("已加入！")
            if "analysis" in st.session_state:
                del st.session_state.analysis
            st.rerun()
    
    with col2:
        if st.button("❌ 取消"):
            if "analysis" in st.session_state:
                del st.session_state.analysis
            st.rerun()

# ====================== 今日總結 + 運動建議 ======================
st.header("📊 今日熱量總結")

total_today = today_data.get("total_calories", 0)
goal = st.session_state.settings["daily_goal"]
gap = goal - total_today   # 正：還可以吃；負：超標

col1, col2, col3 = st.columns(3)
col1.metric("已攝取", f"{total_today} kcal")
col2.metric("目標", f"{goal} kcal")

if gap > 0:
    col3.metric("還可以吃", f"{gap} kcal", delta_color="normal")
    st.success(f"🎯 很好！你今天還有 **{gap} kcal** 的空間，可以再吃點健康的東西～")
elif gap < 0:
    col3.metric("已超標", f"{-gap} kcal", delta_color="inverse")
    st.error(f"⚠️ 你今天已經多吃了 **{-gap} kcal**！")
else:
    col3.metric("達成目標", "剛好！", delta_color="off")
    st.balloons()  # 慶祝氣球效果（Streamlit 內建）
    st.success("完美！今天熱量剛好達到目標，繼續保持～")

if goal > 0:
    progress_value = min(total_today / goal, 1.0)
    st.progress(progress_value)
    if progress_value > 1.0:
        st.progress(1.0)  # 不要超過 100%
if today_data["meals"]:
    st.subheader("今日已記錄餐食")
    for i, meal in enumerate(today_data["meals"]):
        with st.expander(f"🕒 {meal['time']} - {meal['total_calories']} kcal"):
            st.dataframe(pd.DataFrame(meal["foods"]), use_container_width=True)
            # 刪除按鈕
            if st.button("🗑️ 刪除這筆餐食", key=f"delete_meal_{i}"):
                # 移除該筆 meal
                del today_data["meals"][i]
                # 重新計算總熱量
                today_data["total_calories"] = sum(m["total_calories"] for m in today_data["meals"])
                # 存檔
                save_data(data)
                st.success("已刪除這筆餐食")
                st.rerun()  # 立即重新渲染頁面

# ====================== 運動 / 調整建議（根據差距） ======================
st.subheader("要達到每日目標的建議")

weight = st.session_state.settings["weight_kg"]

if gap >= 0:
    # 還沒達到目標
    if gap > 200:
        st.info(f"你今天還可以再攝取 **{gap} kcal**。建議選擇高蛋白、低GI的食物（如雞胸肉、希臘優格、堅果）來補充。")
    else:
        st.info("已經很接近目標了！可以小量加一點碳水或蛋白質，讓身體更有飽足感。")
    
    # 可選：建議輕鬆活動（消耗少一點）
    st.caption("如果想多消耗一點熱量，可以選擇以下輕鬆運動：")
    light_exercises = {
        "散步（輕鬆）": 3.0,
        "瑜伽（輕度）": 3.5,
        "家務勞動": 3.0
    }
    for name, met in light_exercises.items():
        if weight > 0:
            minutes = gap / (met * weight * 0.0175)  # 消耗 gap 熱量的時間
            if minutes > 5 and minutes < 30:  # 只顯示合理時間
                st.write(f"- {name} → 約 {minutes:.0f} 分鐘（可消耗 {gap} kcal）")

elif gap < 0:
    # 超標，需要運動消耗
    excess = -gap
    st.error(f"建議透過運動消耗 **{excess} kcal**，才能回到每日目標。")
    
    exercises = {
        "快走（輕鬆）": 4.0,
        "快走（快速）": 5.0,
        "慢跑": 7.0,
        "騎腳踏車（中速）": 6.5,
        "游泳": 8.0,
        "HIIT / 跳繩": 10.0,
        "重量訓練": 5.0
    }
    
    for name, met in exercises.items():
        if weight > 0:
            minutes = excess / (met * weight * 0.0175)
            if minutes > 5:  # 避免顯示太短的建議
                st.write(f"- **{name}** → 約 **{minutes:.0f} 分鐘**（可消耗 {excess} kcal）")
        else:
            st.warning("請先在側邊欄設定正確體重")

# ====================== 手動補充 ======================
st.header("✍️ 手動快速新增（AI 認不出時用）")
col_a, col_b = st.columns(2)
with col_a:
    name = st.text_input("食物名稱")
with col_b:
    cal = st.number_input("熱量 (kcal)", min_value=0)
    
if st.button("加入手動餐食") and name:
    meal = {
        "time": datetime.now().strftime("%H:%M"),
        "foods": [{"name": name, "calories": cal}],
        "total_calories": cal
    }
    today_data["meals"].append(meal)
    today_data["total_calories"] += cal
    save_data(data)
    st.success("手動加入成功！")
    st.rerun()

st.caption("資料會自動儲存在同資料夾的 my_calorie_data.json，重啟也不會不見")
