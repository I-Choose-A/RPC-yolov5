# RPC-yolov5
## 任务目标
本轮合作轮我的任务是训练出一个模型，用以配合小组完成对商品的快速清点任务。
## 数据集
选用[RPC训练集](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset)作为数据集。  
RPC（Retail Product Checkout）训练集拥有超过8w张图片，图片是许多零散的商品（200类），相比其他训练集来说，RPC训练集中包含的商品绝大多数的都是中国国内超市可以购买到的商品，相比其他包含见都没见过的国外商品数据集有着更大的优势，更适合中国市场。  
![image](https://user-images.githubusercontent.com/52622948/186954913-ba6b5eea-7fff-44fd-bcfb-8a84bd5a1dad.png)

## 模型
本次选用的是yolov5作为目标检测的模型。YOLOv5是一个在COCO数据集上预训练的物体检测架构和模型系列，它代表了Ultralytics对未来视觉AI方法的开源研究，其中包含了经过数千小时的研究和开发而形成的经验教训和最佳实践。  
使用yolov5s版本可以使我们的训练速度较快的同时还保持一个相对不错的准确率。下图是yolov5s的网络结构。
![image](https://user-images.githubusercontent.com/52622948/186965353-977aebb4-e23e-479f-8776-af9e014e463c.png)
## 模型训练
由于训练集中图片的形式和训练集和验证集差异过大，因此将原来的测试集切分为作为训练集和验证集，原来的验证集作为测试集。  
在训练了仅20个epoch后，在验证集上mAP0.5就达到的了0.99，mAP0.5:0.95达到了0.86。  
![image](https://user-images.githubusercontent.com/52622948/186959251-078945c0-58e7-4308-bbda-cc37b7631861.png)
## 接口编写
在训练好模型后使用flask搭建好接口。
```
app = Flask(__name__)
CORS(app, resource=r'/*')

train_json = boxx.loadjson('instances_train2019.json')
price_list = np.loadtxt('price.txt', delimiter=',')


def create_json(pred_num, pred_class, path):
    path = str(path)
    json_text = {'data': []}
    for i in range(len(pred_num)):
        row = train_json['__raw_Chinese_name_df'][pred_class[i]]
        price = price_list[row['category_id'] - 1]
        json_text['data'].append({'id': row['sku_name'], 'name': row['name'], 'num': pred_num[i], 'price': price})

    json_data = json.dumps(json_text, indent=4, separators=(',', ': '), ensure_ascii=False)
    file = open(path + '/' + 'data.json', 'w', encoding='utf-8')

    file.write(json_data)

    file.close()
    return json_data
    
    
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
    app.run(debug=True, host='0.0.0.0', port=2333)
```
通过访问端口post一个图片，可以返回图片中检测到的商品种类，对应的数量，以及单价。
## 算法流程图
### yolov5 part
![72AD63231AD3412D50C62F63407C574D](https://user-images.githubusercontent.com/52622948/186953738-1518ca86-5f50-4319-9f3e-74f29daa3c0e.png)

### 后端 part
![image](https://user-images.githubusercontent.com/52622948/186965150-84a66b3b-6d6a-4780-9955-3989170ae23c.png)

