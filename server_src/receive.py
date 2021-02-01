from flask import Flask, request, jsonify
import json
from PIL import Image
import os
# import cv2

app = Flask(__name__)


@app.route('/api/post', methods=['POST'])
def post():
    try:
        # リクエストの中身を取得
        data = request.form
        file_id = data["body"]  # 様式2-2の場合 "2-2" が格納される

        # 保存先ファイルを決定する
        file_num = _get_filenum(file_id)
        save_path = "images/{0}/{1}.jpg".format(file_id, file_num)

        # 画像を保存する
        file = request.files['img_file'] # POSTされた画像ファイルを取得
        with open(save_path, 'wb') as img:
            print("\n---------------------------------------")
            print("posted_img saved in --> ", save_path)
            img.write(file.stream.read())

        return "200"
    except:
        return "500"

# 保存先ファイルを決定する関数
def _get_filenum(file_id: str) -> str:
    path = "images/" + file_id

    if os.path.exists(path):
        # ファイルパスのリストを取得
        imagePaths = [f for f in os.listdir(path)]

        # 各ファイルの番号を格納するリスト
        file_nums = []
        for image_name in imagePaths:
            file_nums.append(int(image_name.split(".")[0]))

        # 最新ファイルの番号に1加えた値を格納
        file_num = max(file_nums) + 1
    else:
        os.mkdir(path)
        file_num = 1

    return str(file_num)

if __name__ == '__main__':
    # _get_filenum("2-2")
    app.run(host='0.0.0.0', debug=True)
