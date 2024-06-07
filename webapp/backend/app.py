# 运行此函数以搭建后端，提供api。
# ==================================
from flask import (
    Flask,
    jsonify,
    abort,
    request,
    make_response,
    url_for,
    redirect,
    render_template,
)
from flask_httpauth import HTTPBasicAuth
from werkzeug.utils import secure_filename
import os, cv2
from flask_cors import CORS
from predict import get_prediction

# ==================================

UPLOAD_FOLDER = r"./static/uploads"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg"])
app = Flask(__name__, template_folder="", static_folder="")
CORS(app, resources=r"/*")  # 注册CORS, "/*" 允许访问所有api
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
auth = HTTPBasicAuth()
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ================================================#
#                 以下是对API的定义               #
# ================================================#
"""
（待改进）
完成图像上传->预测->保存图片。
流程：
1. 规定api名称，前端调用后，上传图片，图片放到 ./uploads里
2. 然后调用predict函数，从/uploads里读取图片，预测，然后把结果图片保存到./static/label
3. 然后前端获取文件名，从label/文件名 拿图
"""


def pred(filename, rank, threshold):
    img = cv2.imread(r"./static/uploads/" + filename)
    from predict import predict

    full_path = os.path.join(r"./static/result_full/", filename)
    nohoi_path = os.path.join(r"./static/result_no_hoi/", filename)
    img, info = predict(img, rank, threshold, full_path, nohoi_path)
    return info


@app.route("/imgUpload", methods=["GET", "POST"])
def upload_img():
    if request.method == "POST" or request.method == "GET":
        if "file" not in request.files:  # 如果没有检测到上传了文件
            print("No file part")
            return redirect(request.url)  # 重定向

        print("image upload")
        file = request.files["file"]  # 如果检测到了文件，传给file
        rank = int(request.values.get("rank"))
        threshold = float(request.values.get("threshold"))

        if (
            file.filename == ""
        ):  # 如果用户没有选择文件，浏览器仍然会提交一个没有filename的part，所以要再判断
            print("No selected file")
            return redirect(request.url)

        if file:  # and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # 提取文件名
            # 把文件存到/upload 因为flask的file是封装好的，不能直接拿
            print(request.files, file)
            print(filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            print(
                os.path.join(app.config["UPLOAD_FOLDER"], filename), "已存放至uploads！"
            )

            # 1.预测，并把结果存到指定文件夹
            info = pred(filename, rank, threshold)  # 预测，并把结果存到指定文件夹

            # 2.获得图像外的结果 用于展示 比如置信度
            result = {}
            result.update({"filename": filename})
            print(result, '\n',info)
            json = jsonify(result, info)
            print("json:",json)
            return json  # 返回名称便于拿


# ==============================================================================================================================
#
#                                           Main function                                                        	            #
#
# ==============================================================================================================================
@app.route("/")
def main():
    return render_template("main.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
