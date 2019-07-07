"""
配置文件
    ssl配置
    数据存储地址
    模型配置
    模型存储地址
"""
################################################################################
#ssl配置,可以参考https://blog.csdn.net/robin912/article/details/80698896.
################################################################################
pem = '你的文件地址'
key = '你的文件地址'

################################################################################
#数据存储地址.
################################################################################
# 图像、标签csv文件存储目录地址
data_path = 'datas'

# 训练集存储地址
traindataset_path = 'datas/train_data.txt'

# 测试集存储地址
testdataset_path = 'datas/test_data.txt'

# 训练集loss、验证集loss存储地址
loss_log = 'logs/loss_log.txt'

# 预测图片目录
predict_dir = 'predict'

# 服务端存储上传图像的目录地址
upload_dir = './uploads'



################################################################################
#模型配置.
################################################################################
# 单张图片复制次数
num = 16
# 挑选多少图片进行计算平均值(int)和标准差(std)
cnum = 2000

################################################################################
#模型存储地址.
################################################################################
# 最佳模型存储地址
best_model = 'intermediate_models/best_model.pt'

# 混淆矩阵图存储地址
confusion_matrix_image = 'intermediate_models/confusion_matrix.png'

# 损失图存储地址
loss_image = 'intermediate_models/loss_plot.png'

# 中间模型存储目录
intermediate_model = 'intermediate_models'

# 训练好模型地址
inception_model = 'trained_models/inception_model.pt'
resnet_model = 'trained_models/resnet_model.pt'
densenet_model = 'trained_models/densenet_model.pt'
senet_model = 'trained_models/senet_model.pt'

