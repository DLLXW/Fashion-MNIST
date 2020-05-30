import os
import numpy as np
from time import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
'''
该函数是用于处理fashion mnist数据的函数，因为原始是用ubyte格式保存，通过这个函数获得的训练数据形状为
(60000,784),60000行，每一行代表一个图片数据，784列，每一列代表一个像素，因为图片的大小是28x28=784
这也是机器学习方法的惯用套路:数据统一格式：(样本数，每个样本特征数)。这样，每次输入算法的就为一行数据，也就是一个样本。
而输出（也即标签）为(60000,)也就是60000维的列向量，每一个数代表该样本的类型

'''
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

#主函数
def main():
    X_train, y_train = load_mnist('fashionmnist_data/FashionMNIST/raw', kind='train')#处理训练数据
    X_test, y_test = load_mnist('fashionmnist_data/FashionMNIST/raw', kind='t10k')#处理测试数据
    #print(X_train.shape,y_train.shape)#可以打印训练数据的形状(60000, 784) (60000,)
    #使用机器学习算法来分类，首先选取随机森林算法
    #构建随机森林分类器,括号里面那些都是超参数，可以自己调节，俗称调参
    clf = RandomForestClassifier(bootstrap=True, oob_score=True, criterion='gini')
    clf.fit(X_train,y_train)#训练

    #打印分类信息

    print('.................打印分类结果的信息.............')
    print(classification_report(y_test, clf.predict(X_test)))
    ##利用scikit-learn自带的库计算多分类混淆矩阵
    mcm = multilabel_confusion_matrix(y_test, clf.predict(X_test))#mcm即为混淆矩阵
    #通过混淆矩阵可以得到tp,tn,fn,fp
    tp = mcm[:, 1, 1]
    tn = mcm[:, 0, 0]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    print('......................打印混淆矩阵................')
    print(mcm)

if __name__ == '__main__':
    main()




