#-*- coding: UTF-8 -*-
"""
基于Flask的服务端API.
process:
    响应对上传图像进行色素性皮肤病七分类预测请求.
"""
from __future__ import print_function, division
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import os
from model import resnet, inception, efficient
import numpy as np
import heapq
import torch.nn as nn

# 创建Flask类实例
app = Flask(__name__)

################################################################################
# 响应对上传图像进行色素性皮肤病七分类预测请求,请求方式为post.
################################################################################
@app.route('/process', methods=['post'])
def process():
    """
    响应对上传图像进行色素性皮肤病七分类预测请求,
    以json格式返回结果
    """

    try:
        # 存储上传图像的目录地址
        UPLOAD_FOLDER = 'Upload'
        # 创建目录
        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)

        ## 读取并保存上传图像 ##
        file = request.files['file']
        filename = secure_filename(file.filename)
        # 图像存储路径
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        # 选择设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## 图像处理 ##
        # 224 x 224
        normalize = transforms.Normalize(  # 224x224
            mean=[0.7634611, 0.54736304, 0.5729477],
            std=[0.1432169, 0.1519472, 0.16928367]
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize]
        )
        # 加载图片
        img = Image.open(path)
        print(np.asarray(img).shape)
        # 裁剪图片
        img = img.resize((224, 224))
        img = transform(img).unsqueeze(0)

        ## 加载模型 ##
        # 定义模型
        model_list = []
        assert os.path.isdir('Model'), 'Error: no Model directory found!'
        model = resnet(pretrained=False)  # Resnet152模型, 输入224x224
        model.load_state_dict(torch.load('Model/resnet_ckpt.pt', map_location=device))  # 加载模型权重
        model_list.append(model.double().to(device))
        # model = inception(pretrained=False) # InceptionV3, 输入299x299
        # model.load_state_dict(torch.load('Model/inception_ckpt.pt'))  # 加载模型权重
        # model_list.append(model.double().to(device))
        # model = efficient(pretrained=False) # EfficientNet, 输入224x224
        # model.load_state_dict(torch.load('Model/efficient_ckpt.pt'))  # 加载模型权重
        # model_list.append(model.double().to(device))

        ## 预测结果 ##
        img = img.double().to(device)
        # 所有模型预测结果之和
        sum = None
        # 集成学习平均策略
        for model in model_list:
            with torch.no_grad():
                # 预测结果
                output = model(img)
                output = nn.Softmax(dim=1)(output)
                output = output.squeeze()
                if sum is None:
                    sum = output.cpu().detach().numpy()
                else:
                    sum += output.cpu().detach().numpy()
        val = sum / len(model_list)
        # 获取top1、top2和top3概率
        top1, top2, top3 = heapq.nlargest(3, range(len(val)), val.take)

        ## 返回结果 ##
        # 七类色素性皮肤病名称
        classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        # 预测结果
        results_json = {}
        results_json['top1'] = classes[top1] + ' ' + str(format(val[top1], '.2f'))
        results_json['top2'] = classes[top2] + ' ' + str(format(val[top2], '.2f'))
        results_json['top3'] = classes[top3] + ' ' + str(format(val[top3], '.2f'))
        results_json['success'] = True
        for k, v in results_json.items():
            print(k, ' ', v)
        return jsonify(results_json)
    except Exception as e:
        results_json = {}
        results_json['success'] = False
        results_json['errMsg'] = repr(e)
        return jsonify(results_json)

################################################################################
# 函数入口.
################################################################################
if __name__ == '__main__':
    # 本地运行程序并设置外部可访问
    app.run(port=8086, host='0.0.0.0', debug=True)


    # # 服务端部署
    # # ssl配置文件地址,可以参考https://blog.csdn.net/robin912/article/details/80698896
    # pem = 'ssl/pem.pem'
    # key = 'ssl/key.key'
    #
    # # 运行程序并设置外部可访问
    # app.run(port = 8086, host='0.0.0.0', debug = True, ssl_context = (pem, key))