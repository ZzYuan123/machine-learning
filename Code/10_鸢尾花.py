import jieba
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


# 鸢尾花示例
def datasets_demo():
    iris = load_iris()
    print(f"鸢尾花数据集:{iris.data}")
    print(f"查看数据集描述:{iris['DESCR']}")
    print(f"查看特征值名字:{iris.feature_names}")
    print(f"查看特征值:{iris.data, iris.data.shape}")

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    print(f"训练集的特征值:{x_train}")
    print(f"测试集的特征值:{x_test}")
    print(f"训练集的目标值:{y_train}")
    print(f"测试集的真实值:{y_test}")


# 字典特征抽取
def transformer_demo():
    datas = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60},
             {'city': '深圳', 'temperature': 30}]
    # 实例化一个转换器类
    # 字典特征提取 向量化 也就是二维数组 (3个特征值,4个样本)

    # sparse 默认是True 输出为稀疏矩阵 不含0
    # 可以将非0值按照位置表示出来 节省内存 提高加载效率
    transfer = DictVectorizer(sparse=False)
    # 调用fit_transform()
    datas_new = transfer.fit_transform(datas)
    print(f"datas_new: {datas_new}")
    print(f"特征名字: {transfer.get_feature_names_out()}")
    print(datas_new.shape)


# 提取英文文本
def count_vectorizer_demo():
    datas = ["life is short,i like like python", "life is too long,i dislike python"]
    # CountVectorizer 是计数的 不是出现才会变成1
    # 分词默认按照单词分 出现一次 就会+=1
    transfer = CountVectorizer()
    datas_new = transfer.fit_transform(datas)
    # 强制输出为数组类型 不让输出稀疏矩阵 .toarray()
    print(f"datas_new: {datas_new.toarray()}")
    print(f"特征名字:{transfer.get_feature_names_out()}")
    print(datas_new.shape)


# 使用jieba自动分词
def cut_word(text):
    # 将传入的文本按照jieba自动分词 然后按照空格连接起来
    return " ".join(jieba.cut(text))


# 提取中文文本
# 如果不使用jieba分词 系统对中文的分词是随机的 需要自己手动分词
def count_vectorizer_demo2():
    datas = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
             "我们看到的从很远的星系来的光实在几百万年之前发出来的，这样当我们看到宇宙时，我们是在看他的过去。",
             "如果只用一种方式了解某样事物，你就不会真正了解他，了解事物真正的含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    datas_new = []
    for data in datas:
        datas_new.append(cut_word(data))
    # 在这里加stop_words可以限制哪些词不做特征词
    transfer = CountVectorizer(stop_words=['如果'])
    datas_final = transfer.fit_transform(datas_new)
    print(f"datas_new: {datas_final.toarray()}")
    print(f"特征名字:{transfer.get_feature_names_out()}")


if __name__ == "__main__":
    # datasets_demo()
    # transformer_demo()
    # count_vectorizer_demo()
    count_vectorizer_demo2()
