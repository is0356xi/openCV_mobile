import sys
from PIL import Image
import pyocr
import pyocr.builders

class ocr:
    def __init__(self):
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            print("No OCR tool found")
            sys.exit(1)

        self.tool = tools[0]
        print("Will use tool '%s'" % (self.tool.get_name()))

    def read_txt(self, img):
        txt = self.tool.image_to_string(
            img,
            lang="jpn",
            builder=pyocr.builders.TextBuilder(tesseract_layout=6)
        )
        
        # print(txt)

        return txt

if __name__ == "__main__":
    ocr = ocr()
    img = Image.open("../images/base/14-7.jpg")
    ocr.read_txt(img)