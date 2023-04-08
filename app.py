# 导入必要的库
import pandas as pd
import streamlit as st
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 导入数据
data = pd.read_csv('ydl.csv')

# 数据清洗和处理
# ...

# 特征选择
features = ['Sex', 'Dist', 'Fre', 'Con']

# 将分类变量转为数字编码
data['Sex'] = data['Sex'].map({'女': 0, '男': 1})
data['Dist'] = data['Dist'].map({'跨城': 1, '跨省': 2, '跨国': 3})
data['Fre'] = data['Fre'].map({'0-1个月': 1, '1-6个月': 2, '6-12个月': 3})
data['Con'] = data['Con'].map({'累了看不到未来': 1, '性格不合适': 2, '家庭问题': 3, '激情退却': 4, '变心': 5, '没有安全感': 6, '其它': 7})
data['Prob'] = data['Prob']*1000
# 建立模型
X_train, X_test, y_train, y_test = train_test_split(data[features], data['Prob'], test_size=0.2)
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('预测准确率：', accuracy)


# streamlit显示
st.title('异地恋情侣小调查')

with st.expander(''):
     
     st.write('有人说，异地恋是爱情最快乐的想象。')
     st.write('也有人说，爱终究会因为距离消失。')
     st.write('异地恋有多大的概率能熬到最好的结局？')
     st.write('这份调查分析是对你末来或者正在进行的异地恋分手的预警分析。')
     st.write('你也可以将它看作对你过去某段异地恋的回顾。')
     st.write('如果触碰到伤疤,无意冒犯。')
my_bar = st.progress(0)

for percent_complete in range(100):
     time.sleep(0.0001)
     my_bar.progress(percent_complete + 1)

st.balloons()

st.write('调查小背景：此网页收集整理了一系列关于异地恋的数据，试图找出异地恋之间的共同点和如何避免在异地恋之中的一些小问题，并做了一个有关异地恋分手的概率测试。')

st.write('采用问卷调查所收集的数据，很好的采纳了不同年龄段的人群，各自对异地恋的看法，使数据呈现出一个的较为准确的结果。所用模型：随机森林。')

st.write('影响异地恋分手概率的因素很多 在这我们选择了其中影响较大的几个因素 下图是每个因素造成异地恋分手的概率。')

import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
st.title('经历异地恋的次数')

import streamlit as st
import matplotlib.pyplot as plt
fig = plt.figure()
label=['一次','两次','三次','四次及以上']
explode=[0.01,0.01,0.01,0.01]
value=[62.5,25,9.6,2.9]
plt.pie(value, explode=explode,labels=label, autopct='%1.1f%%')
plt.show()
st.pyplot(fig)

st.title('经历异地恋的时间')

import streamlit as st
import matplotlib.pyplot as plt
fig = plt.figure()
label=['0-6个月','6-12个月','1-3年','3年及以上']
explode=[0.01,0.01,0.01,0.01]
value=[25,21.9,32.9,20.2]
plt.pie(value, explode=explode,labels=label, autopct='%1.1f%%')
plt.show()
st.pyplot(fig)


st.title('异地恋成功的概率')

import streamlit as st
import matplotlib.pyplot as plt
fig = plt.figure()
label=['成功','失败']
explode=[0.01,0.01]
value=[70.6,29.4]
plt.pie(value, explode=explode,labels=label, autopct='%1.1f%%')
plt.show()
st.pyplot(fig)

st.write('现在大家想知道如果是多个因素 异地恋分手的概率是多少吗？')
st.write('IF YES! 请完成下面测试 它会告诉你想要的答案。')

#streamlit网页显示
st.header('异地恋分手概率测试')
sex = st.selectbox('1.性别', ['', '女', '男']) 
if not sex:
    st.write('请选择性别')
else:
    if sex == '男':
        df = pd.DataFrame({
            '性别': ['男'],
            '比例': ['32.1%']
        })
        st.write(df)
    else:
        df = pd.DataFrame({
            '性别': ['女'],
            '比例': ['67.9%']
        })
        st.write(df) 

dist = st.selectbox('2.您和您恋人的距离', ['', '跨城','跨省','跨国'])
if not dist:
    st.write('请选择距离')
else:
    if dist == '跨城':
        df = pd.DataFrame({
            '距离': ['跨城'],
            '比例': ['24.4%']
        })
        st.write(df)
    elif dist == '跨省':
        df = pd.DataFrame({
            '距离': ['跨省'],
            '比例': ['68.9%']
        })
        st.write(df)
    elif dist == '跨国':
        df = pd.DataFrame({
            '距离': ['跨国'],
            '比例': ['17.0%']
        })
        st.write(df)

fre = st.selectbox('3.您和您的恋人的见面频率', ['', '0-1个月', '1-6个月', '6-12个月']) 
if not fre:
    st.write('请选择频率')
else:
    if fre == '0-1个月':
        df = pd.DataFrame({
            '时间': ['0-1个月'],
            '总体比例': ['13.6%']
        })
        st.write(df)
    elif fre == '1-6个月':
        df = pd.DataFrame({
            '时间': ['1-6个月'],
            '总体比例': ['63.6%']
        })
        st.write(df)
    elif fre == '6-12个月':
        df = pd.DataFrame({
            '时间': ['6-12个月'],
            '总体比例': ['20.5%']
        })
        st.write(df)

con = st.selectbox('4.您和您恋人目前的矛盾', ['', '累了看不到未来', '性格不合适', '家庭问题', '激情退却', '变心', '没有安全感', '其它'])
if not con:
    st.write('请选择矛盾')
else:
    if con == '累了看不到未来':
        df = pd.DataFrame({
            '矛盾': ['累了看不到未来'],
            '总体比例': ['32.9%']
        })
        st.write(df)
    elif con == '性格不合适':
        df = pd.DataFrame({
            '时间': ['性格不合适'],
            '总体比例': ['16.5%']
        })
        st.write(df)
    elif con == '家庭问题':
        df = pd.DataFrame({
            '矛盾': ['家庭问题'],
            '总体比例': ['7%']
        })
        st.write(df)
    elif con == '激情退却':
        df = pd.DataFrame({
            '矛盾': ['激情退却'],
            '总体比例': ['5.9%']
        })
        st.write(df)
    elif con == '家庭问题':
        df = pd.DataFrame({
            '矛盾': ['变心'],
            '总体比例': ['18.2%']
        })
        st.write(df)
    elif con == '没有安全感':
        df = pd.DataFrame({
            '矛盾': ['没有安全感'],
            '总体比例': ['8.9%']
        })
        st.write(df)
    elif con == '其它':
        df = pd.DataFrame({
            '矛盾': ['其它'],
            '总体比例': ['10.6%']
        })
        st.write(df)

if st.button('提交'):
    # 将用户选择的特征转为模型所需的数字编码
    sex = 0 if sex == '女' else 1
    dist = {'跨城': 1, '跨省': 2, '跨国': 3}[dist]
    fre = {'0-1个月': 1, '1-6个月': 2, '6-12个月': 3}[fre]
    con = {'累了看不到未来': 1, '性格不合适': 2, '家庭问题': 3, '激情退却': 4, '变心': 5, '没有安全感': 6, '其它': 7}[con]
    
# 使用模型进行预测
    prob = model.predict([[sex, dist, fre, con]])[0]
    
    # 将预测结果输出到网页上
    st.write('分手概率预测：', prob/1000)

st.write('异地恋真的很辛苦，希望携手坚持到现在的你们，')
st.write ('不要因为一时的距离和一些细枝末节，而疏忽了这个本该相守一生的人。')
st.write ('等下次见面，请给他（她）一个大大的拥抱吧！')

