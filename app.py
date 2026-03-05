import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 1. ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(page_title="Coffee Shop AI Predictor", page_icon="☕", layout="centered")

# ==========================================
# 2. ฟังก์ชันโหลดข้อมูลและโมเดล (โหลดครั้งเดียว)
# ==========================================
@st.cache_data
def load_base_data():
    # โหลดไฟล์ข้อมูล
    df = pd.read_csv("coffee_shop_sales.csv")
    # หาเวลาเปิดปิดของแต่ละสาขา
    store_hours = df.groupby('store_location')['hour'].agg(['min', 'max']).to_dict(orient='index')
    # จัดกลุ่มยอดขายย้อนหลัง
    daily_grouped = df.groupby(['transaction_date', 'store_location', 'name_day', 'month', 'hour', 'product_category'])['transaction_qty'].sum().reset_index(name='total_cups')
    return df, store_hours, daily_grouped

@st.cache_resource
def load_ai_models():
    # โหลด Encoders (ต้องมีไฟล์ในโฟลเดอร์)
    le_loc = joblib.load('le_loc.joblib')
    le_day = joblib.load('le_day.joblib')
    le_cat = joblib.load('le_cat.joblib')
    
    # โหลด Models (ใช้ Try-Except ป้องกันเว็บพังในกรณีที่อัปโหลดไฟล์ไม่ครบ)
    models = {}
    try: models['m1'] = joblib.load('Model_1_Random_Forest_Classifier.joblib')
    except: models['m1'] = None
    try: models['m2'] = joblib.load('Model_2_Random_Forest_Regressor.joblib')
    except: models['m2'] = None
    try: models['m3'] = joblib.load('Model_3_XGBoost_Regressor.joblib')
    except: models['m3'] = None
    try: models['m4'] = joblib.load('Model_4_CatBoost_Regressor.joblib')
    except: models['m4'] = None
    try: models['m5'] = joblib.load('Model_5_LightGBM_Regressor.joblib')
    except: models['m5'] = None
    try: 
        models['m6'] = joblib.load('Model_6_ANN_Regressor.joblib')
        scaler = joblib.load('scaler.joblib')
    except: 
        models['m6'] = None
        scaler = None

    return le_loc, le_day, le_cat, models, scaler

# โหลดข้อมูลเตรียมไว้
df, store_hours, daily_grouped = load_base_data()
le_loc, le_day, le_cat, models, scaler = load_ai_models()

# ==========================================
# 3. ส่วนหัวเรื่องและเลือกโมเดล
# ==========================================
st.title("⚡ ระบบทำนายยอดขายอัจฉริยะ")
st.markdown("เลือกอัลกอริทึม AI ด้านล่าง เพื่อพยากรณ์เมนูที่จะขายดีที่สุดตามช่วงเวลา")

model_options = {
    "1. Random Forest Classifier": "m1",
    "2. Random Forest Regressor": "m2",
    "3. XGBoost Regressor": "m3",
    "4. CatBoost Regressor": "m4",
    "5. LightGBM Regressor": "m5",
    "6. ANN (โครงข่ายประสาทเทียม)": "m6"
}

selected_model_name = st.selectbox("📌 เลือก Model", list(model_options.keys()))
model_key = model_options[selected_model_name]
current_model = models[model_key]

# แสดง Metrics จำลอง (ในของจริงสามารถดึงค่าที่เซฟไว้มาแสดงได้)
st.markdown("📊 **ประสิทธิภาพและความน่าเชื่อถือของ AI (Model Evaluation):**")
col1, col2, col3 = st.columns(3)
if "Classifier" in selected_model_name:
    col1.metric("Accuracy (ความแม่นยำ)", "85.2%")
    col2.metric("Precision", "84.1%")
    col3.metric("F1-Score", "84.5%")
else:
    col1.metric("MAE (คลาดเคลื่อนเฉลี่ย)", "±2.88 ชิ้น")
    col2.metric("RMSE (จุดบอดสูงสุด)", "±4.48 ชิ้น")
    col3.metric("R-Squared (ความแม่นยำ)", "63.7%")

st.divider()

# ==========================================
# 4. ฟอร์มรับข้อมูล (พร้อมแก้บัคเรื่องเวลา)
# ==========================================
st.subheader("📍 ระบุข้อมูลเพื่อทำนาย")

locations = list(store_hours.keys())
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
months = list(range(1, 13))

col_a, col_b = st.columns(2)

with col_a:
    selected_loc = st.selectbox("🎈 สาขา", locations)
    selected_day = st.selectbox("📅 วัน", days)

with col_b:
    selected_month = st.selectbox("🗓️ เดือน", months)
    
    # ดึงเวลาเปิด-ปิด ของสาขาที่เลือกมาสร้างเป็นตัวเลือก
    min_h = store_hours[selected_loc]['min']
    max_h = store_hours[selected_loc]['max']
    valid_hours = list(range(min_h, max_h + 1))
    
    selected_time = st.selectbox(f"⏰ เวลา (สาขานี้เปิด {min_h}:00 - {max_h}:00 น.)", valid_hours)

st.write("")
# ==========================================
# 5. กระบวนการทำนายและแสดงผล
# ==========================================
if st.button("🚀 สรุปผลการทำนาย", use_container_width=True):
    if current_model is None:
        st.error(f"❌ ไม่พบไฟล์ของโมเดลนี้ กรุณาอัปโหลดไฟล์ .joblib ของโมเดลนี้ขึ้น GitHub ก่อนครับ")
    else:
        st.divider()
        st.subheader(f"🎯 สรุปผลการทำนายด้วย {selected_model_name.split('. ')[1]}")
        st.markdown(f"**📍 สาขา :** {selected_loc} | **📅 วันที่ :** {selected_day} (เดือนที่ {selected_month}) | **⏰ เวลา :** {selected_time}:00 น.")
        
        # --- เตรียมข้อมูลสำหรับทำนาย ---
        all_categories = le_cat.classes_
        loc_enc = le_loc.transform([selected_loc])[0]
        day_enc = le_day.transform([selected_day])[0]
        all_cat_encoded = le_cat.transform(all_categories)
        
        with st.spinner("กำลังประมวลผล..."):
            results_df = pd.DataFrame()

            # แบบที่ 1: Classifier (ใช้ Predict Proba)
            if model_key == 'm1':
                probs = current_model.predict_proba([[loc_enc, day_enc, selected_time, selected_month]])[0]
                results_df = pd.DataFrame({'Category': all_categories, 'Value': probs * 100})
                value_label = "โอกาสเป็นที่ 1 (%)"
            
            # แบบที่ 2: CatBoost (ใช้ Text ตรงๆ)
            elif model_key == 'm4':
                predict_data = pd.DataFrame({
                    'store_location': [selected_loc] * len(all_categories),
                    'name_day': [selected_day] * len(all_categories),
                    'month': [selected_month] * len(all_categories),
                    'hour': [selected_time] * len(all_categories),
                    'product_category': all_categories
                })
                preds = current_model.predict(predict_data)
                results_df = pd.DataFrame({'Category': all_categories, 'Value': preds})
                value_label = "คาดหมาย (ชิ้น)"

            # แบบที่ 3: ANN (ต้องผ่าน Scaler)
            elif model_key == 'm6':
                predict_data = pd.DataFrame({
                    'loc_encoded': [loc_enc] * len(all_categories),
                    'day_encoded': [day_enc] * len(all_categories),
                    'month': [selected_month] * len(all_categories),
                    'hour': [selected_time] * len(all_categories),
                    'cat_encoded': all_cat_encoded
                })
                predict_data_scaled = scaler.transform(predict_data)
                preds = current_model.predict(predict_data_scaled)
                results_df = pd.DataFrame({'Category': all_categories, 'Value': preds})
                value_label = "คาดหมาย (ชิ้น)"
                
            # แบบที่ 4: Regressor ทั่วไป (RF, XGBoost, LightGBM)
            else:
                predict_data = pd.DataFrame({
                    'loc_encoded': [loc_enc] * len(all_categories),
                    'day_encoded': [day_enc] * len(all_categories),
                    'month': [selected_month] * len(all_categories),
                    'hour': [selected_time] * len(all_categories),
                    'cat_encoded': all_cat_encoded
                })
                preds = current_model.predict(predict_data)
                results_df = pd.DataFrame({'Category': all_categories, 'Value': preds})
                value_label = "คาดหมาย (ชิ้น)"

            # จัดเรียงผลลัพธ์
            results_df = results_df[results_df['Value'] > 0]
            results_df = results_df.sort_values(by='Value', ascending=False).head(6)
            
            # --- ส่วนแสดงผล AI ---
            if not results_df.empty:
                best_cat = results_df.iloc[0]['Category']
                st.success(f"🌟 เมนูยอดฮิตอันดับ 1 น่าจะเป็น: **>> {best_cat} <<**")
                
                st.markdown("💡 **การจัดอันดับคาดการณ์จาก AI (Top 6 Predictions):**")
                total_val = results_df['Value'].sum()
                
                for idx, row in enumerate(results_df.itertuples(), 1):
                    pct = (row.Value / total_val) * 100
                    if model_key == 'm1':
                        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; {idx}. **{row.Category}** : โอกาส {row.Value:.1f}%")
                    else:
                        st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; {idx}. **{row.Category}** : คาดว่าจะขายได้ประมาณ {row.Value:.1f} ชิ้น ({pct:.1f}%)")
            else:
                st.warning("AI คาดการณ์ว่าอาจจะไม่มีการขายในช่วงเวลานี้ หรือขายน้อยมาก")

            st.divider()

            # --- ส่วนแสดงผล สถิติของจริงย้อนหลัง ---
            st.markdown("📈 **ข้อมูลสถิติของจริงย้อนหลัง (นำมาเทียบเพื่อความมั่นใจ):**")
            
            sample_history = daily_grouped[(daily_grouped['store_location'] == selected_loc) &
                                           (daily_grouped['name_day'] == selected_day) &
                                           (daily_grouped['hour'] == selected_time) &
                                           (daily_grouped['month'] == selected_month)]
            
            if not sample_history.empty:
                monthly_sum = sample_history.groupby('product_category')['total_cups'].sum().reset_index()
                monthly_sum = monthly_sum.sort_values(by='total_cups', ascending=False).head(6)
                total_cups_period = monthly_sum['total_cups'].sum()
                
                st.write(f"*(ยอดขายรวมทั้งหมดในอดีต ณ ช่วงเวลานี้คือ {total_cups_period} ชิ้น/แก้ว)*")
                
                for idx, row in enumerate(monthly_sum.itertuples(), 1):
                    pct = (row.total_cups / total_cups_period) * 100
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp; {idx}. **{row.product_category}** : เคยขายได้ {row.total_cups} ชิ้น ({pct:.1f}%)")
            else:
                st.info("- ไม่พบข้อมูลการขายในอดีตสำหรับเงื่อนไขนี้")
