import json
import shutil

import boxx
import numpy as np
from flask import Flask, request
from flask_cors import CORS

import detect

app = Flask(__name__)
CORS(app, resource=r'/*')

train_json = boxx.loadjson('instances_train2019.json')
price_list = np.loadtxt('price.txt', delimiter=',')


def create_json(pred_num, pred_class, path):
    path = str(path)
    json_text = {'data': []}
    for i in range(len(pred_num)):
        row = train_json['__raw_Chinese_name_df'][pred_class[i]]
        price = price_list[row['category_id']-1]
        json_text['data'].append({'id': row['sku_name'], 'name': row['name'], 'num': pred_num[i], 'price': price})

    json_data = json.dumps(json_text, indent=4, separators=(',', ': '), ensure_ascii=False)
    file = open(path + '/' + 'data.json', 'w', encoding='utf-8')

    file.write(json_data)

    file.close()
    return json_data


@app.route('/hello', methods=['GET'])
def index():
    return '123'


# 定义路由
@app.route("/photo", methods=['POST'])
def get_frame():
    # 接收图片
    upload_file = request.files['file']
    file_name = upload_file.filename
    upload_file.save('temp/' + file_name)
    pred_num, pred_class, save_dir = detect.run(source='temp/' + file_name, project='image', weights='weights/best.pt',
                                                conf_thres=0.6)
    result = create_json(pred_num, pred_class, save_dir)
    shutil.move('temp/' + file_name, str(save_dir) + '/origin_img_' + file_name)
    return result


if __name__ == "__main__":
    app.run(debug=True,port=2333)
