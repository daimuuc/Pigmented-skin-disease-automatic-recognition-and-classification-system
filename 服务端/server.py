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
from models import densenet, senet
import CONFIG

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
    # 单张图片数据增强后的图片个数
    NUM = 16
    # 模型权重的地址
    KWARGS = {'1': CONFIG.densenet_model,
              '2': CONFIG.senet_model }

    try:
        # 存储上传图像的目录地址
        UPLOAD_FOLDER = CONFIG.upload_dir

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
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize]
        )
        # 加载图片
        img = Image.open(path)
        # 裁剪图片
        img = img.resize((300, 300))
        # 数据增强后的所有图片
        imgs = None
        for i in range(NUM):
            if imgs is None:
                imgs = transform(img).view(1, 3, 224, 224)
            else:
                imgs = torch.cat((imgs, transform(img).view(1, 3, 224, 224)), 0)

        ## 加载模型 ##
        models = []
        for k, v in KWARGS.items():
            if k == '1':
                model = densenet(pretrained = False)
            elif k == '2':
                model = senet(pretrained = False)
            model.load_state_dict(torch.load(v, map_location = 'cpu'))
            model = model.to(device)
            model.eval()
            models.append(model)

        ## 预测结果 ##
        imgs = imgs.to(device)
        # 所有模型预测结果之和
        sum = None
        # 集成学习平均策略
        for model in models:
            # 预测结果
            output = model(imgs)

            output = output.detach()
            # 平均策略
            val = None
            for i in range(output.size(0)):
                if val is None:
                    val = output[i]
                else:
                    val = val + output[i]
            val = val / output.size(0)

            if sum is None:
                sum = val
            else:
                sum += val
        val = sum / len(models)
        _, a = torch.max(val, 0)
        # 预测结果
        a = a.item()

        ## 返回结果 ##
        # 七类色素性皮肤病名称
        classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        # 预测结果
        predict = classes[a]
        results_json = {}
        results_json['result'] = predict
        results_json['status'] = 'Success'
        return jsonify(results_json)
    except Exception:
        results_json = {}
        results_json['status'] = 'Failure'
        return jsonify(results_json)

################################################################################
# 函数入口.
################################################################################
if __name__ == '__main__':
    # ssl配置文件地址,可以参考https://blog.csdn.net/robin912/article/details/80698896
    pem = CONFIG.pem
    key = CONFIG.key

    # 运行程序并设置外部可访问
    app.run(port = 8086, host='0.0.0.0', debug = True, ssl_context = (pem, key))