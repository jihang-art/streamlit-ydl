# 导入必要的库
import pandas as pd
import streamlit as st
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 添加登录界面
username = st.sidebar.text_input("用户名：异地恋")
password = st.sidebar.text_input("密码：123456", type="password")
if username == "异地恋" and password == "123456":
    st.sidebar.success("登录成功！")
    
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
    st.title('预测异地恋分手概率')

    with st.expander('项目简介'):
         st.text('数据管理与机器学习是一种人工智能技术，它可以通过学习大量数据和模式，自动发现数据中隐藏的规律')
         st.text('和趋势。本文利用数据管理与机器学习的相关知识来分析因素，收集数据，建立模型来预测异地恋分手的')
         st.text('概率，从而帮助人们更好的了解和应对异地恋的挑战。')
         st.text('关于因素我们在数据库(Eg:AI Studio飞桨、世界银行)查找了相关文章，选定了几个影响较大的因素，')
         st.text(' 根据选定的因素制作了调查问卷和表单，通过线上和线下的方式，收集了203份数据，作为本项目的数据')
         st.text('集。调查问卷和表单如下：')
         from PIL import Image
         image = Image.open('p1.jpg')
         st.image(image, caption='Sunset in the mountains', use_column_width=True)
         from PIL import Image
         image = Image.open('p2.jpg')
         st.image(image, caption='Sunset in the mountains', use_column_width=True)
         st.text('关于模型，我们分析了异地恋分手概率这个问题，它既可以看作回归问题也可以看作分类问题，因此我们')
         st.text('选择了既适用于回归问题和分类问题的预测概率的随机森林模型，同时利用决策树、LogisticRegress-')
         st.text('-ion模型进行了准确率分析!')

    st.write('有人说，异地恋是爱情最快乐的想象。')

    st.write('也有人说，爱终究会因为距离消失。')
 
    st.write('异地恋有多大的概率能熬到最好的结局？')

    st.write('这份调查分析是对你末来或者正在进行的异地恋分手的预警分析。')

    st.write('你也可以将它看作对你过去某段异地恋的回顾。')

    st.write('如果触碰到伤疤,无意冒犯。')

    st.header('影响异地恋分手概率的因素很多 在这我们选择了其中影响较大的几个因素 下图是每个因素造成异地恋分手的概率。')

    import matplotlib
    matplotlib.rcParams['font.family'] = 'SimHei'
    st.subheader('1.经历异地恋的次数（次）')

    fig = plt.figure()
    label=['1','2','3','>=4']
    explode=[0.01,0.01,0.01,0.01]
    value=[62.5,25,9.6,2.9]
    plt.pie(value, explode=explode,labels=label, radius = 0.7,autopct='%1.1f%%')
    plt.show()
    st.pyplot(fig)

    st.subheader('2.经历异地恋的时间(个月、年)')

    fig = plt.figure()
    label=['0-6 months','6-12 months','1-3 years','>=3 years']
    explode=[0.01,0.01,0.01,0.01]
    value=[25,21.9,32.9,20.2]
    plt.pie(value, explode=explode,labels=label, radius = 0.7,autopct='%1.1f%%')
    plt.show()
    st.pyplot(fig)


    st.subheader('3.异地恋成功的概率')

    fig = plt.figure()
    label=['success','fail']
    explode=[0.01,0.01]
    value=[70.6,29.4]
    plt.pie(value, explode=explode,labels=label, radius = 0.7,autopct='%1.1f%%')
    plt.show()
    st.pyplot(fig)

    st.header('现在大家想知道如果是多个因素 异地恋分手的概率是多少吗？')
    st.header('IF YES! 请完成下面测试 它会告诉你想要的答案。')

    #streamlit网页显示
    st.subheader('异地恋分手概率测试')
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
else:
    st.sidebar.error("按键盘Enter进行登录！")
