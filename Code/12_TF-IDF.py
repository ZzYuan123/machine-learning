'''
TF-IDF 用来衡量一个词对一篇文章的重要程度。
它想回答的问题就是："哪些词最能代表这篇文章？"

TF Term Frequency（词频） 表示一个词在当前文章出现了多少次
    TF = 某词出现次数 / 总词数
IDF Inverse Document Frequency（逆文档频率） 表示一个词是不是很稀有
    IDF = log以10为底 然后取一个词在所有文章库中的文章/包含这个词的文章

eg:
    1000篇文章---语料库 100篇文章包含"非常" 10篇文章包含"经济"
    现在有两篇文章
                文章A(100词) 出现10次 经济
                TF = 10 / 100 = 0.1
                IDF = log 10 (1000 / 10) = 2
                TF-IDF = 0.1 * 2 = 0.2

                文章A(100词) 出现10次 非常
                TF = 10 / 100 = 0.1
                IDF = log 10 (1000 / 10) = 1
                TF-IDF = 0.1 * 2 = 0.1

    经济的 TF-IDF > 非常的 TF-IDF 所以也就是说 "经济" 这个次的重要性要大于 "非常"
'''

# 一个词在当前文章中出现的次数越多 TF越大
# 一个词在所有的我文章中出现的越少 IDF越大
import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# 使用jieba自动分词
def cut_word(text):
    # 将传入的文本按照jieba自动分词 然后按照空格连接起来
    return " ".join(jieba.cut(text))


def count_vectorizer_demo2():
    datas = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
             "我们看到的从很远的星系来的光实在几百万年之前发出来的，这样当我们看到宇宙时，我们是在看他的过去。",
             "如果只用一种方式了解某样事物，你就不会真正了解他，了解事物真正的含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    datas_new = []
    for data in datas:
        datas_new.append(cut_word(data))
    # 在这里加stop_words可以限制哪些词不做特征词
    transfer = TfidfVectorizer(stop_words=['如果'])
    datas_final = transfer.fit_transform(datas_new)
    print(f"datas_new: {datas_final.toarray()}")
    print(f"特征名字:{transfer.get_feature_names_out()}")


if __name__ == '__main__':
    count_vectorizer_demo2()
