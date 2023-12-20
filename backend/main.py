import os
import string
import urllib
import uuid
import pickle
import datetime
import time
import shutil
import pdb;
import cv2
import numpy as np
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, UploadFile, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import starlette
import firebase_admin
from firebase_admin import credentials, storage
import util
from test import test as test_module
# Định nghĩa đường dẫn cho thư mục chứa bản ghi chấm công và cơ sở dữ liệu người dùng
ATTENDANCE_LOG_DIR = './logs'
DB_PATH = './db'

# Khởi tạo Firebase Admin SDK
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'bookingroom-7b732.appspot.com'})

# Tạo thư mục nếu chúng không tồn tại
for dir_ in [ATTENDANCE_LOG_DIR, DB_PATH]:
    if not os.path.exists(dir_):
        os.mkdir(dir_)

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Thiết lập các cài đặt cho CORS (Cross-Origin Resource Sharing)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa endpoint cho việc đăng nhập
@app.post("/login")
async def login(file: UploadFile = File(...)):
    # Đặt tên file ảnh là một UUID với định dạng png
    file.filename = f"{uuid.uuid4()}.png"
    contents = await file.read()
    # Lưu file ảnh vào thư mục tạm thời (hoặc có thể lưu vào Firebase Storage trực tiếp)
    temp_file_path = f"image/{file.filename}"
    remote_file_path = f"face/image/{file.filename}"
    # Lưu trữ nội dung ảnh vào file
    with open(temp_file_path, 'wb') as f:
        f.write(contents)
    upload_to_firebase_storage(temp_file_path,remote_file_path )

    # Nhận diện khuôn mặt và trạng thái kết quả
    user_name, match_status = recognize(cv2.imread(temp_file_path))
    img = cv2.imread(temp_file_path)
    new_height = int(img.shape[1] * 3 / 4)

    # Thay đổi kích thước ảnh để đáp ứng yêu cầu chiều cao/chiều rộng là 4/3
    resized_image = cv2.resize(img, (img.shape[1], new_height))
  # Thêm test cho mặt giả mạo
    label = test_module(img, model_dir='C:\\Users\\quang\\OneDrive\\Máy tính\\FACE\\face-attendance-web-app-react-python\\backend\\resources\\anti_spoof_models', device_id=0)

    if label == 1:
        # Gọi test anti-spoofing thành công
        print("Anti-spoofing test passed")
        # Nếu nhận diện thành công, ghi thông tin chấm công vào file
        if match_status:
            epoch_time = time.time()
            date = time.strftime('%Y%m%d', time.localtime(epoch_time))
            with open(os.path.join(ATTENDANCE_LOG_DIR, '{}.csv'.format(date)), 'a') as f:
                f.write('{},{},{}\n'.format(user_name, datetime.datetime.now(), 'IN'))
                f.close()
    else:
        # Gọi test anti-spoofing không thành công
        print("Anti-spoofing test failed")
        return {'user': user_name, 'match_status': 'false'}

    # xóa ảnh tạm đi 
    os.remove(temp_file_path)
    # Trả về kết quả
    return {'user': user_name, 'match_status': match_status}
  #  return JSONResponse(content={'user': user_name, 'match_status': match_status, 'image_data': image_data})

# Endpoint cho việc đăng xuất
@app.post("/logout")
async def logout(file: UploadFile = File(...)):

    # Đặt tên file ảnh là một UUID với định dạng png
    file.filename = f"{uuid.uuid4()}.png"
    contents = await file.read()

    # Lưu trữ nội dung ảnh vào file
    with open(file.filename, "wb") as f:
        f.write(contents)

    # Nhận diện khuôn mặt và trạng thái kết quả
    user_name, match_status = recognize(cv2.imread(file.filename))

    # Nếu nhận diện thành công, ghi thông tin chấm công vào file
    if match_status:
        epoch_time = time.time()
        date = time.strftime('%Y%m%d', time.localtime(epoch_time))
        with open(os.path.join(ATTENDANCE_LOG_DIR, '{}.csv'.format(date)), 'a') as f:
            f.write('{},{},{}\n'.format(user_name, datetime.datetime.now(), 'OUT'))
            f.close()

    # Trả về kết quả
    return {'user': user_name, 'match_status': match_status}

# Endpoint để đăng ký người dùng mới
@app.post("/register_new_user")
async def register_new_user(file: UploadFile = File(...), text=None):
    file.filename = f"{uuid.uuid4()}.png"
    contents = await file.read()

    # Lưu trữ nội dung ảnh vào file
    with open(file.filename, "wb") as f:
        f.write(contents)

    # Sao chép file ảnh vào thư mục cơ sở dữ liệu với tên là text
    shutil.copy(file.filename, os.path.join(DB_PATH, '{}.png'.format(text)))

    # Trích xuất mã nhúng khuôn mặt và lưu vào file pickle
    embeddings = face_recognition.face_encodings(cv2.imread(file.filename))
    file_ = open(os.path.join(DB_PATH, '{}.pickle'.format(text)), 'wb')
    pickle.dump(embeddings, file_)

    # Xóa file ảnh tạm thời
    os.remove(file.filename)

    # Trả về kết quả
    return {'registration_status': 200}

# Endpoint để lấy các bản ghi chấm công
@app.get("/get_attendance_logs")
async def get_attendance_logs():

    # Tạo một tệp ZIP chứa thư mục chứa bản ghi chấm công
    filename = 'out.zip'
    shutil.make_archive(filename[:-4], 'zip', ATTENDANCE_LOG_DIR)

    # Trả về tệp ZIP
    return starlette.responses.FileResponse(filename, media_type='application/zip', filename=filename)

# Hàm nhận diện khuôn mặt
def recognize(img):
    embeddings_unknown = face_recognition.face_encodings(img)
    
    # Nếu không tìm thấy khuôn mặt, trả về 'no_persons_found'
    if len(embeddings_unknown) == 0:
        return 'no_persons_found', False
    else:
        # Lấy mã nhúng của khuôn mặt đầu tiên
        embeddings_unknown = embeddings_unknown[0]

    match = False
    j = 0

    # Lấy danh sách các tệp pickle trong thư mục cơ sở dữ liệu
    db_dir = sorted([j for j in os.listdir(DB_PATH) if j.endswith('.pickle')])

    # Tìm kiếm so sánh với các mã nhúng trong cơ sở dữ liệu
    while ((not match) and (j < len(db_dir))):
        path_ = os.path.join(DB_PATH, db_dir[j])
        file = open(path_, 'rb')
        embeddings = pickle.load(file)[0]

        # So sánh mã nhúng
        match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]

        j += 1

    # Nếu tìm thấy sự khớp, trả về tên người dùng và True, ngược lại, trả về 'unknown_person' và False
    if match:
        return db_dir[j - 1][:-7], True
    else:
        return 'unknown_person', False
def upload_to_firebase_storage(local_file_path, remote_file_name):
    # Tạo đối tượng storage của Firebase
    bucket = storage.bucket()

    # Upload file lên Firebase Storage
    blob = bucket.blob(remote_file_name)
    blob.upload_from_filename(local_file_path)
    # Hàm lấy ảnh từ    
# def get_image_from_firebase(image_path_in_storage):
#     # Tạo đối tượng storage của Firebase
#     bucket = storage.bucket()

#     # Tạo đối tượng blob từ đường dẫn của ảnh trên Firebase Storage
#     blob = bucket.blob(image_path_in_storage)

#     # Lấy dữ liệu của ảnh từ Firebase Storage
#     image_data = blob.download_as_text()

#     # Chuyển dữ liệu về mảng numpy
#     image_array = np.frombuffer(image_data.encode(), dtype=np.uint8)
    
#     # Đọc ảnh từ mảng numpy
#     image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

#     return image