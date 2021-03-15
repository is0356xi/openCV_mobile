# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random
import sys
import os
import pprint as pp
import sys

import ocr
from PIL import Image


class table_rec:
    def __init__(self, img_path, output_name, rotate=False):
        if rotate:
            self.angle = -2.5
            img = cv2.imread(img_path)
            self.img = self._rotate_img(img, self.angle)
        else:
            self.img = cv2.imread(img_path)

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.output_name = output_name

        # self.minArea = 30000
        self.minArea = 8000
        self.maxArea = 500000

        self.stop = 0

        self.ocr = ocr.ocr()

    def get_edge(self):
        # Cannyでエッジ抽出
        edges = cv2.Canny(self.gray, 1, 100, apertureSize=3)
        cv2.imwrite("rec_images/edge_{}.png".format(self.output_name), edges)

        # 膨張処理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        self.edges = cv2.dilate(edges, kernel)
        cv2.imwrite("rec_images/edge_morp_{}.png".format(self.output_name), self.edges)

    def get_contours(self):
        contours, hierarchy = cv2.findContours(self.edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for cnt, hrchy in zip(contours, hierarchy[0]):
            # if cv2.contourArea(cnt) < self.minArea:
            if cv2.contourArea(cnt) < self.minArea or self.maxArea < cv2.contourArea(cnt):
                continue # 面積が小さいものは除く
            if hrchy[3] == -1: # hrchy => [次の輪郭, 前の輪郭, 子要素, 親要素]となっていて、次の輪郭は同一階層が先に探索される
                continue # ルートノードは除く
            # if hrchy[0] != -1:
            #     continue


            # print(type(cnt))
            rect = cv2.minAreaRect(cnt)
            rect_points = cv2.boxPoints(rect).astype(int)
            # rect_points = np.array(sorted(rect_points, key=lambda x: (x[0], x[1])))
            # pp.pprint(rect_points.min(axis=0))
            rects.append(rect_points)

        np_rects = np.array(rects)
        # pp.pprint(np_rects.min(axis=0).min(axis=0))
        # print("------------------------------")
        # for rect in rects:
        #     pp.pprint(self._min_max(rect))
      
        self.rects = sorted(rects, key=lambda x: x.min(axis=0)[1])
        

        # print("------------------------------")
        # pp.pprint(self.rects)
        # print(self.rects)

    # def _zscore(self, x, axis = None):
    #     xmean = x.mean(axis=axis, keepdims=True)
    #     xstd  = np.std(x, axis=axis, keepdims=True)
    #     zscore = (x-xmean)/xstd
    #     return zscore

    # def _min_max(self, x, axis=None):
        # min = x.min(axis=axis, keepdims=True)
        # max = x.max(axis=axis, keepdims=True)
        # result = (x-min)/(max-min)
        # return result

    def draw_rect(self):
        draw_img = np.array(self.img, dtype=np.uint8)
        for i, rect in enumerate(self.rects):
            color = np.random.randint(0,255,3).tolist()
            # print(color)
            cv2.drawContours(draw_img, self.rects, i, color, 2)
            cv2.putText(draw_img, str(i), tuple(rect[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

        cv2.imwrite("rec_images/table_rec_{}.png".format(self.output_name), draw_img)

    def _rotate_img(self, img, angle):
        # 中心座標を計算
        h = img.shape[0]
        w = img.shape[1]
        center = (int(w/2), int(h/2))

        # 回転させる
        scale = 1.0
        trans = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_img = cv2.warpAffine(img, trans, (w,h))

        return rotated_img

    def txt_read(self, rects, img_corr=False):
        count = 0
        for i in range(4,10):
            for rect in rects:
                print(rect)
                img = self._transform(rect)

                

                if img_corr:
                    w = img.shape[1]
                    h = img.shape[0]
                    h2 = int(h*0.1)
                    w2 = int(w*0.1)

                    img = img[h2:h-h2, w2:w-w2]

                    # img = cv2.resize(img, (int(w*1.5), int(h*1.5)))
                
                cv2.imwrite("rec_images/ocr_{}.png".format(self.output_name), img)
                pil_img = self._cv2pil(img)
                # pil_img.save('rec_images/ocr_pil{}.jpg'.format(self.output_name))

            
                result = self.ocr.read_txt(pil_img, i)

                # ``` WordBox or LineBoxを使う時 ```
                for res in result:
                    print(res.content)
                    cv2.rectangle(img, res.position[0], res.position[1], (0, 0, 255), 1)
                cv2.imshow(str(img_corr), img)
                key = cv2.waitKey(0)
                if key & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

                print(result)

                if count == self.stop:
                    break

                count+=1

    def _transform(self, rect):
        epsilon = 0.01 * cv2.arcLength(rect, True)
        print(epsilon)
        approx = cv2.approxPolyDP(rect, epsilon, True)

        print(approx)

        x_min = rect.min(axis=0)[0]
        y_min = rect.min(axis=0)[1]
        x_max = rect.max(axis=0)[0]
        y_max = rect.max(axis=0)[1]


        w = x_max - x_min
        h = y_max - y_min

        src = np.float32(list(map(lambda x: x[0], approx)))
        # print(src)
        # dst = np.float32([[0,0],[0,card_img_width],[card_img_height,card_img_width],[card_img_height,0]])
        dst = np.float32([[w,h],[0,h],[0,0],[w,0]])

        pp.pprint(src)
        pp.pprint(dst)

        projectMatrix = cv2.getPerspectiveTransform(src, dst)

        # transformed_img = cv2.warpPerspective(self.img, projectMatrix, (card_img_height, card_img_width))
        transformed_img = cv2.warpPerspective(self.img, projectMatrix, (w, h))
        # transformed_img = transformed_img[20:,:]
        return transformed_img


    def _cv2pil(self,image):
        ''' OpenCV型 -> PIL型 '''
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = Image.fromarray(new_image)
        return new_image

    def main(self):
        self.get_edge()
        self.get_contours()
        self.draw_rect()

        self.txt_read(self.rects, img_corr=True)
        self.txt_read(self.rects)


if __name__ == "__main__":
    args = sys.argv
    file_name = args[1]
    img_path = "../images/base/{}.jpg".format(file_name)

    if len(args) >= 3:
        rotate = args[2]
        table_rec = table_rec(img_path, file_name, rotate)
        table_rec.main()
    else:
        table_rec = table_rec(img_path, file_name)
        table_rec.main()
