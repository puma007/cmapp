# coding=utf-8
import os, datetime, random
import numpy as np

from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


# 定义一个flask web app
app = Flask(__name__)
app.config.update(
    SECRET_KEY = os.urandom(24),
    # 最大上传大小，当前5MB
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024
)
# 生成无重复随机数用于文件基本名
gen_rnd_filename = lambda :"%s%s" %(datetime.datetime.now().strftime('%Y%m%d%H%M%S'), str(random.randrange(1000, 10000)))
# 已经训练好的模型位置
MODEL_PATH = 'models/keras_medicine_tl.h5'

# 加载模型
model = load_model(MODEL_PATH)
# 编译并运行模型，可以让第一次预测更快，不然第一次加载会很慢
# 参考 https://github.com/keras-team/keras/issues/6124 解释
model._make_predict_function()
print('Model loaded. Start serving...')

# 也可以直接加载keras自带的模型 https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Start serving...')


def model_predict(img_path, model):
    '''
    预测
    :param img_path: 图片
    :param model: 模型
    :return:result
    '''
    img = image.load_img(img_path, target_size=(224, 224)) #224，224
    x = image.img_to_array(img)
    x /= 255.
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    proba = model.predict(x)[0]
    idx = np.argmax(proba)
    print(idx)
    result = ''
    label_medicine = {'枸杞子': 0, '白术 ': 1, '茯苓': 2}
    for k, v in label_medicine.items():
        if v == idx:
            result = k
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # 获取上传的文件
        f = request.files['file']
        # 保存上传文件到 ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(gen_rnd_filename() + "." + f.filename.split('.')[-1]))
        f.save(file_path)
        # 预测
        result = model_predict(file_path, model)
        return result
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8098, debug=True)
