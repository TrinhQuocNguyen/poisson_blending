import requests
from requests.auth import HTTPDigestAuth

import io
from PIL import Image
import numpy as np
import cv2

import datetime

# カメラのIPアドレス　192.168.0.10
# 画像データの取得　　/cgi-bin/camera
# 解像度の指定　　　　resolution
url = r'http://taiyoyuden:Taiyoyuden1234@192.168.111.13/cgi-bin/mjpeg?framerate=30&resolution=1280x720'

# 認証情報
user = "*****"
pswd = "*****"

# 顔を識別するためのファイル
cascade_file = "haarcascade_frontalface_alt2.xml"
cascade = cv2.CascadeClassifier(cascade_file)

while True:

    # 画像の取得
    rs = requests.get(url, auth=HTTPDigestAuth(user, pswd))

    # 取得した画像データをOpenCVで扱う形式に変換
    img_bin = io.BytesIO(rs.content)
    img_pil = Image.open(img_bin)
    img_np  = np.asarray(img_pil)
    img  = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)

    # 顔検出のためにグレイスケール画像に変換
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 検出した顔の位置情報を取得
    face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))

    # 顔を検出したらループ処理を終了
    if len(face_list) != 0:
        break

cat = 0
now = "{:%Y%m%d%H%M%S}".format(datetime.datetime.now())

for (pos_x, pos_y, w, h) in face_list:

    # 顔の切出
    img_face = img[pos_y:pos_y+h, pos_x:pos_x+w]

    # 顔画像をリサイズし、サイズを統一
    img_face = cv2.resize(img_face, (200, 200))