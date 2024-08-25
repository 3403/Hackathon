import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.data.path.append('E:\\PycharmProjects\\Region\\venv\\nltk_data')

# 下载所需的nltk资源
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# 读取CSV文件
df = pd.read_csv('../mock_production_issues_varied.csv')

# 初始化词形还原器
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# 定义文本预处理函数
def preprocess_text(text):
    # 转换为小写
    text = text.lower()

    # 移除特殊字符和数字
    text = re.sub(r'\W+', ' ', text)

    # 分词
    words = word_tokenize(text)

    # 去除停用词和词形还原
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return processed_words


# 定义关键字提取函数
def extract_keywords(processed_words, top_n=5):
    # 计算词频
    word_freq = Counter(processed_words)

    # 提取出现频率最高的前n个词
    most_common_words = word_freq.most_common(top_n)

    # 返回关键词
    keywords = [word for word, freq in most_common_words]
    return ' '.join(keywords)


# 应用预处理函数到problem summary
df['processed_problem_summary'] = df['problem summary'].apply(preprocess_text)

# 提取并存储关键词
df['keyWords'] = df['processed_problem_summary'].apply(extract_keywords)

# 保存处理后的数据到新的CSV文件中
preprocessed_csv_path = 'preprocessed_production_issues_with_keywords.csv'
df.to_csv(preprocessed_csv_path, index=False)

print(f"Preprocessed data with keywords saved to {preprocessed_csv_path}")