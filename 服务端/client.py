#-*- coding: UTF-8 -*-
"""
模拟客户端请求.
client:
    模拟客户端上传图像进行色素性皮肤病七分类预测请求
"""
from __future__ import print_function, division
import requests

################################################################################
#模拟客户端上传图像进行色素性皮肤病七分类预测请求
################################################################################
def client(path):
    """
    模拟客户端请求，并展示预测结果.

    param:
        path -- 上传图像地址
    """
    # 上传文件信息
    files = {
        'file': open(path, 'rb')
    }

    # 模拟客户端请求
    results = requests.post('http://localhost:8086/process',
                             files = files)

    # 显示预测结果
    results = results.json()
    for k, v in results.items():
        print(k, ' ', v)

################################################################################
#函数入口
################################################################################
if __name__ == '__main__':
    ## 模拟客户端请求 ##
    # 图片地址
    path = 'Upload/ISIC_0024306.jpg'
    client(path)