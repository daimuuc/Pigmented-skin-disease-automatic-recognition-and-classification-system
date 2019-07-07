项目名称：
    色素性皮肤病七分类系统
项目功能：
    基于深度学习、集成学习、迁移学习等技术的色素性皮肤病自动识别七分类系统。
    本系统主要由服务端和客户端两个模块组成。服务端使用DenseNet161和SENet154
    两个模型构成集成模型，从而实现了对色素性皮肤病自动识别七分类。客户端使
    用微信小程序开发。用户通过微信小程序上传图像到服务端，服务端返回所属类别。
项目结构：
  服务端
    |———— datas(数据存储地址)
    |
    |———— intermediate_models(训练过程中，模型、损失图等存储地址)
    |
    |———— logs(训练过程中，训练loss日志存储地址)
    |
    |———— predict(预测过程中，图片存储地址)
    |
    |———— trained_models(训练好模型存储地址)
    |
    |———— client.py(模拟客户端请求)
    |
    |———— CONFIG.py(配置文件)
    |
    |———— data.py(数据预处理)
    |
    |———— ensemble.py(测试集成模型性能)
    |
    |———— model.py(自定义模型MFFNet)
    |
    |———— models.py(加载Resnet152、Densenet161等模型)
    |
    |———— predict.py(模型预测)
    |
    |———— server.py(基于Flask的服务端API)
    |
    |———— test.py(测试模型性能)
    |
    |———— train.py(训练模型)
    |
    |———— utils.py(常见工具函数)
项目部署：
    1、修改server.py文件并运行
    2、修改client.py文件并运行(可选)
    3、修改微信客户端服务器配置
项目数据集：
    https://challenge2018.isic-archive.com/task3/
