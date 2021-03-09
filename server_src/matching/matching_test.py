# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import sys
import os

class Matching:

    def __init__(self):
        # 画像を読み込む
        base_img_path = "images/14-7/1.jpg"

        # temp_img_path = "images/2-2/1.jpg"
        # temp_img_path = "images/5/1.jpg"
        temp_img_path = "images/14-7/2.jpg"
        
        

        self.gray_base_img = cv2.imread(base_img_path, 0)
        self.gray_temp_img = cv2.imread(temp_img_path, 0)

        # 画像をBGRカラーで読み込み
        self.color_base_img = cv2.imread(base_img_path, 1)
        self.color_temp_img = cv2.imread(temp_img_path, 1)

    def proc_match_bf(self):
        # 特徴点の検出
        type = cv2.AKAZE_create()
        kp_01, des_01 = type.detectAndCompute(self.gray_base_img, None)
        kp_02, des_02 = type.detectAndCompute(self.gray_temp_img, None)

        # マッチング処理
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # matches = bf.match(des_01, des_02)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_01, des_02, k=2)

        # データを間引く
        ratio = 0.5
        good = []
        for x, y in matches:
            if x.distance < ratio * y.distance:
                good.append(x)

        matches = sorted(matches, key=lambda x:x[0].distance)
        print(len(matches), len(good))

        mutch_image = cv2.drawMatches(self.color_base_img, kp_01, self.color_temp_img, kp_02, good, None, flags=2)

        # 結果の表示
        cv2.imshow("result", mutch_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def proc_match_bf_list(self):

        base_path = "images/base"
        base_paths = [os.path.join(base_path, f) for f in os.listdir(base_path)]

        temp_path = "images/5"
        temp_paths = [os.path.join(temp_path, f) for f in os.listdir(temp_path)]
        print(base_paths, temp_paths)

        # temp_img_path = "images/14-7/2.jpg"
        # temp_img_path = "images/5/2.jpg"
        # temp_img_path = "images/2-2/3.jpg"

        for temp_img_path in temp_paths:
            gray_temp_img = cv2.imread(temp_img_path, 0)
            color_temp_img = cv2.imread(temp_img_path, 1)

            base_paths = ["images/2-2/1.jpg", "images/5/1.jpg", "images/14-7/1.jpg"]

            print("\n********** template image --> {} *************".format(temp_img_path))

            # 各様式とどれくらいマッチするか保持
            match_dict = {}

            # 特徴点の検出
            type = cv2.AKAZE_create()

            for base_img_path in base_paths:
                
                gray_base_img = cv2.imread(base_img_path, 0)
                color_base_img = cv2.imread(base_img_path, 1)

                kp_01, des_01 = type.detectAndCompute(gray_base_img, None)
                kp_02, des_02 = type.detectAndCompute(gray_temp_img, None)

                # ブルートフォース & K近傍を使ってマッチング処理
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des_01, des_02, k=2)

                

                # データを間引く
                ratio = 0.6
                good = []
                for x, y in matches:
                    if x.distance < ratio * y.distance:
                        good.append(x)

                match_dict[base_img_path] = len(good)

                matches = sorted(matches, key=lambda x:x[0].distance)
                print("{}   =>   match_num={} , good_num={}".format(base_img_path,len(matches), len(good)))
                
                mutch_image = cv2.drawMatches(color_base_img, kp_01, color_temp_img, kp_02, good, None, flags=2)

                # 結果の表示
                cv2.imshow("result", mutch_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            print(match_dict)
            max_match = max(match_dict.values())
            key = [k for k, v in match_dict.items() if v == max_match][0]

            print(key)


    def proc_match(self):
        # 対象画像を指定
        base_image_path = "images/base/2-2.jpg"
        temp_image_path = "images/2-2/1.jpg"

        # 画像をグレースケールで読み込み
        gray_base_src = cv2.imread(base_image_path, 0)
        gray_temp_src= cv2.imread(temp_image_path, 0)

        print(gray_base_src.shape)
        print(gray_temp_src.shape)

        # ラベリング処理
        label = cv2.connectedComponentsWithStats(gray_base_src)
        n = label[0] - 1
        data = np.delete(label[2], 0, 0)

        # マッチング結果書き出し準備
        color_src = cv2.cvtColor(gray_base_src, cv2.COLOR_GRAY2BGR)
        height, width = gray_temp_src.shape[:2]

        # ラベリング情報を利用して各オブジェクトごとのマッチング結果を画面に表示
        for i in range(n):
    
            # 各オブジェクトの外接矩形を赤枠で表示
            x0 = data[i][0]
            y0 = data[i][1]
            x1 = data[i][0] + data[i][2]
            y1 = data[i][1] + data[i][3]
            cv2.rectangle(color_src, (x0, y0), (x1, y1), (0, 0, 255))

            # 各オブジェクトごとの類似度を求める
            x2 = x0 - 5
            y2 = y0 - 5
            x3 = x0 + width + 5
            y3 = y0 + height + 5

            crop_src = gray_base_src[y2:y3, x2:x3]
            c_height, c_width = crop_src.shape[:2]

            res = cv2.matchTemplate(crop_src, gray_temp_src, cv2.TM_CCOEFF_NORMED)
            res_num = cv2.minMaxLoc(res)[1]
            cv2.putText(color_src, str(i + 1) + ") " +str(round(res_num, 3)), (x0, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

        # 結果の表示
        cv2.imshow("color_src", color_src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

if __name__ == '__main__':
    match = Matching()
    match.proc_match_bf_list()
    # match.proc_match_bf()