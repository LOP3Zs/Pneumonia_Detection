from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
from tensorflow import keras
import mysql.connector

app = Flask(__name__)

model = keras.models.load_model('densenet121_model_finetune.keras')

db_config = {
    'host': 'localhost',  # Change if using a remote server
    'user': 'root',  # Replace with your MySQL username
    'password': '123456',  # Replace with your MySQL password
    'database': 'pneunomia'
}

def get_prediction_text(pred):
    """
    Hàm chuyển kết quả dự đoán (0 hoặc 1) thành văn bản.
    0: Normal, 1: Pneumonia
    """
    if pred == 0:
        return "Normal"
    elif pred == 1:
        return "Pneumonia"
    else:
        return "Invalid Prediction"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ID = request.form.get("ID")
        name = request.form.get("name")
        surname = request.form.get("surname")
        gioi_tinh = request.form.get("gioi_tinh")
        do_tuoi = int(request.form.get("do_tuoi"))
        phone_number = request.form.get("phone_number")
        city = request.form.get("city")



        # Kiểm tra file ảnh được gửi lên qua request.files
        if 'image' not in request.files:
            return render_template('index.html', result="Không tìm thấy file ảnh trong yêu cầu.")
        file = request.files['image']
        if file.filename == "":
            return render_template('index.html', result="Không có file ảnh được chọn.")

        # Mở file ảnh bằng Pillow
        image = Image.open(file)
        # Nếu ảnh không ở chế độ RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize ảnh theo kích thước mà model yêu cầu (ví dụ: 224x224)
        image = image.resize((224, 224))
        
        # Chuyển ảnh sang mảng NumPy và chuẩn hóa về khoảng [0,1]
        image_array = np.array(image).astype('float32') / 255.0
        # Thêm dimension cho batch: shape (1, 224, 224, 3)
        image_array = np.expand_dims(image_array, axis=0)

        # Dự đoán với model
        # Giả sử model trả về xác suất cho lớp "Pneumonia" ở vị trí đầu tiên
        y_pred_probs = model.predict(image_array)
        # Áp dụng threshold 0.5 để chuyển đổi xác suất thành nhãn 0 hoặc 1
        pred_class = 1 if y_pred_probs[0][0] >= 0.5 else 0

        # Chuyển kết quả dự đoán thành văn bản
        result_text = get_prediction_text(pred_class)
        image.save(f"D:/cdio3/images/Predict_{result_text}_{ID}.png")
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="pneunomia"
)
        cursor = conn.cursor()
        sql = """INSERT INTO Users (ID,name, surname, gioi_tinh, do_tuoi, phone_number, city, prediction_label) 
                 VALUES (%s,%s, %s,%s, %s, %s, %s, %s)"""
        values = (ID,name, surname, gioi_tinh, do_tuoi, phone_number, city, result_text)
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
        conn.close()        

    except Exception as e:
        return jsonify({"result": f"Có lỗi xảy ra: {e}"})
    
    
    return render_template('index.html', result=result_text)
if __name__ == '__main__':
    app.run(debug=True)
