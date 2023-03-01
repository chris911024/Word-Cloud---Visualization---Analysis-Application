from fbb import WordCloud
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import jieba
import jieba.analyse
from collections import Counter 

dictfile = "c:/xx/xx.txt"  # 字典檔
stopfile = "c:/xx/stopwords.txt"  # stopwords
fontpath = "c:/xx/xxx.ttc"  # 字型檔

mdfile = "c:/xx/xxxx.mdx"  # 文檔
pngfile = "c:/xx/cloud.jpg"  # 下載的底圖

alice_mask = np.array(Image.open(pngfile))

jieba.set_dictionary(dictfile)
jieba.analyse.set_stop_words(stopfile)

text = open(mdfile,"r",encoding="utf-8").read()

tags = jieba.analyse.extract_tags(text, topK=25)

seg_list = jieba.lcut(text, cut_all=False)
dictionary = Counter(seg_list)

#計算統計次數
freq = {}
for ele in dictionary:
    if ele in tags:
        freq[ele] = dictionary[ele]
print(freq) 

wordcloud = WordCloud(background_color="white", mask=alice_mask, contour_width=3, contour_color='steelblue', font_path= fontpath).generate_from_frequencies(freq)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()