# AI-2025
A collection of academic projects and coursework.
<img width="741" height="334" alt="image" src="https://github.com/user-attachments/assets/44ff81de-3246-42ab-a870-4f0aaec1c5d5" />

# Intelligent Assistant and Advisory System for Online Stores
# ผู้ช่วยและให้คำปรึกษาอัจฉริยะสำหรับร้านค้าออนไลน์

   ระบบทำนายยอดขายอัจฉริยะ (Intelligent Sales Prediction System) ถูกพัฒนาขึ้นเพื่อช่วยวิเคราะห์และคาดการณ์ยอดขายสินค้าในร้านค้า โดยใช้เทคโนโลยี Machine Learning ในการเรียนรู้จากข้อมูลยอดขายย้อนหลัง เช่น สาขา วัน เดือน และเวลา เพื่อทำนายว่าสินค้าประเภทใดมีแนวโน้มขายดีที่สุดในช่วงเวลาที่กำหนด
ภายในระบบสามารถเลือกใช้โมเดลปัญญาประดิษฐ์หลายประเภท ได้แก่ Random Forest, XGBoost, CatBoost, LightGBM และ Artificial Neural Network (ANN) เพื่อเปรียบเทียบประสิทธิภาพของแต่ละโมเดล โดยประเมินความแม่นยำของการทำนายด้วยตัวชี้วัด เช่น Accuracy, Precision, F1-score, MAE, RMSE และ R² Score
ผลลัพธ์จากระบบจะช่วยให้ผู้ใช้งานสามารถวางแผนการขาย การเตรียมสินค้า และการบริหารจัดการร้านค้าได้อย่างมีประสิทธิภาพมากยิ่งขึ้น

## 🚀 ฟีเจอร์หลัก (Features)
1. ระบบทำนายยอดขายสินค้า
ใช้โมเดล Machine Learning วิเคราะห์ข้อมูลจาก สาขา วัน เดือน และเวลา เพื่อคาดการณ์ว่าสินค้าประเภทใดมีแนวโน้มขายดีที่สุดในช่วงเวลานั้น

2. ระบบเลือกและเปรียบเทียบโมเดล AI
ผู้ใช้สามารถเลือกโมเดล เช่น Random Forest, XGBoost, CatBoost, LightGBM และ ANN เพื่อเปรียบเทียบประสิทธิภาพของแต่ละโมเดลในการทำนายยอดขาย

3. ระบบประเมินประสิทธิภาพโมเดล (Model Evaluation)
แสดงค่าประเมินความแม่นยำของ AI เช่น Accuracy, Precision, F1-score, MAE, RMSE และ R² Score เพื่อช่วยวิเคราะห์ความน่าเชื่อถือของผลการทำนาย

## 👥 ภาระงาน (Responsibilities)

| สัปดาห์ที่ (Week) |นายชัยวัฒน์ แอบสุข รหัสประจำตัว 67026652|
| :--- | :--- |
| **Week 1** | วางแผนการทำงานโครงงาน |
| **Week 2** | ศึกษาขั้นตอนและวิธีการเพิ่มเติม เริ่มทดสอบทีละขั้นตอน |
| **Week 3** | เริ่ม Train และ Test Ai  |
| **Week 4** | นำ Model ที่ทำการทดสอบแล้วมาทำเว็บลง Streamlit |
---

## 🛠️ เทคโนโลยีที่ใช้ (Tech Stack)
- **Language:** Python
- **Framework:** Streamlit
- **Libraries:** pandas,numpy,scikit-learn,xgboost,catboost,lightgbm,streamlit,joblib

## 📦 การติดตั้งและการใช้งาน (Setup & Installation)
1.ทำการติดตั้ง Python (VScode) หรือใช้ colab ในการ Run Code

2.บันทึกไฟล์ Data([coffee_shop_sales.csv](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/coffee_shop_sales.csv)) ที่ใช้ในการ Train Model มาใช้ในการ Run Code

3.นำทั้ง 6 Model ([model_1_random_forest_classifier.py](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/model_1_random_forest_classifier.py),[model_2_random_forest_regressor.py](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/model_2_random_forest_regressor.py),[model_3_xgboost_regressor.py](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/model_3_xgboost_regressor.py),[model_4_catboost_regressor.py](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/model_4_catboost_regressor.py),[model_5_lightgbm_regressor.py](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/model_5_lightgbm_regressor.py),[model_6_ann_regressor.py](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/model_6_ann_regressor.py)) ไป Run Code ทดสอบ

4.เมื่อทดสอบแล้วทำการโหลดไฟล์ที่ได้จากการ Run ([Model_1_Random_Forest_Classifier.joblib](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/Model_1_Random_Forest_Classifier.joblib),[Model_2_Random_Forest_Regressor.joblib](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/Model_2_Random_Forest_Regressor.joblib),[Model_3_XGBoost_Regressor.joblib](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/Model_3_XGBoost_Regressor.joblib),[Model_4_CatBoost_Regressor.joblib](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/Model_4_CatBoost_Regressor.joblib),[Model_5_LightGBM_Regressor.joblib](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/Model_5_LightGBM_Regressor.joblib),[Model_6_ANN_Regressor.joblib](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/Model_6_ANN_Regressor.joblib)) เข้า Git hub

5.เพิ่มไฟล์สำหรับลง library ที่จำเป็นลงใน Git Hub ([requirements.txt](https://github.com/Chaiwat-Aepsuk/AI-2025/blob/e776a102b2f10177aaacb36a6c8c29ab398d04a6/requirements.txt))

6.นำไฟล์ app.py ที่สร้างจาก Python เป็นหน้าเว็บ ใส่ลงใน Git Hub และนำไป Run ใน Streamlit

- เว็บที่ทำจาก Python : https://ai-2025-iu4ammg93cnd9ymg5ydyu9.streamlit.app/#project-methodologies-of-artificial-intelligence-by-pxe-chaiwat

## 📗 ผลสรุป (Summarize)
   ระบบทำนายยอดขายอัจฉริยะถูกพัฒนาขึ้นเพื่อช่วยวิเคราะห์และคาดการณ์ยอดขายสินค้าในร้านค้า โดยใช้เทคโนโลยี Machine Learning ในการเรียนรู้จากข้อมูลยอดขายย้อนหลัง ระบบจะนำข้อมูลสำคัญ ได้แก่ สาขา วัน เดือน และเวลา มาวิเคราะห์เพื่อทำนายว่าสินค้าประเภทใดมีแนวโน้มขายดีที่สุดในช่วงเวลานั้น
   
   ภายในระบบสามารถเลือกใช้โมเดล AI หลายประเภท เช่น Random Forest, XGBoost, CatBoost, LightGBM และ Artificial Neural Network (ANN) เพื่อเปรียบเทียบประสิทธิภาพของแต่ละโมเดล พร้อมแสดงค่าประเมินความแม่นยำของโมเดล เช่น Accuracy, Precision, F1-score, MAE, RMSE และ R² Score
   
ผลลัพธ์ที่ได้จากระบบช่วยให้ผู้ใช้งานสามารถวางแผนการขาย เตรียมสินค้า และบริหารจัดการร้านค้าได้อย่างมีประสิทธิภาพมากขึ้น

## 🔰 เเหล่งที่มา (Source)
**Data จาก web kaggle :** https://www.kaggle.com/datasets/ahmedabbas757/coffee-sales

## 💡 สะท้อนคิด (Reflection)
   จากการทำโครงงานครั้งนี้ ทำให้ผมได้เรียนรู้เกี่ยวกับการนำเทคโนโลยี Machine Learning มาประยุกต์ใช้ในการวิเคราะห์และทำนายยอดขายสินค้า โดยได้ศึกษาและทดลองใช้โมเดลหลายประเภท เช่น Random Forest, XGBoost, CatBoost, LightGBM และ Artificial Neural Network (ANN)

   โดยแต่ละโมเดลมีบทบาทที่แตกต่างกัน เช่น 
   - Random Forest ใช้สำหรับการจำแนกหรือพยากรณ์ข้อมูลจากการรวมผลของต้นไม้หลายต้น
   - XGBoost เป็นโมเดลที่เน้นการเพิ่มประสิทธิภาพของการทำนายด้วยเทคนิค Boosting เรียนรู้จากความผิดพลาด
   - CatBoost เหมาะกับข้อมูลที่มีลักษณะเป็นหมวดหมู่
   - LightGBM เป็นโมเดลที่มีความเร็วในการประมวลผลสูง
   - Artificial Neural Network (ANN) เป็นโมเดลที่จำลองการทำงานของโครงข่ายประสาทมนุษย์เพื่อเรียนรู้รูปแบบข้อมูลที่ซับซ้อน

โครงงานนี้ทำให้ผมได้เข้าใจหลักการทำงานของโมเดล Machine Learning มากขึ้น รวมถึงสามารถนำความรู้ไปประยุกต์ใช้ในการวิเคราะห์ข้อมูลและพัฒนาระบบอัจฉริยะในอนาคตได้
