#ライブラリのインポート
import pandas as pd
import numpy as np
import pulp
model = pulp.LpProblem("MyModel", pulp.LpMinimize)

# 必要なライブラリをインポート
import streamlit as st
import pandas as pd

import requests
from io import StringIO

# タイトルとテキストを記入
st.title('Streamlit 時間割')

#基本情報のデータ
teacher_list = [f'教員{i}' for i in range(22)]
subject_list = ["英語","数学","国語","理科","社会","芸術","体育","情報","総合探究","自主自学"]
grade_list = [1,2,3]
class_dict = {3:[1,2,3,4],2:[1,2,3,4],1:[1,2,3,4]}
teacher_dict = {t:g for t,g in zip(teacher_list,[3,3,3,2,1,1,3,2,1,3,2,1,2,1,1,3,2,1,3,2,3,2])} #教員の所属学年
period = [1,2,3,4,5,6,7]
week = ["月","火","水","木","金"]
Classroom_mobility = ["芸術","体育","情報","家庭科"] #移動教室授業
six_period = ["総合探究","自主自学"] #6限のみの授業
subject_dict = {s:n for s,n in zip(subject_list,[4,5,5,4,4,2,2,2,1,3])} #必要授業数

url = "https://docs.google.com/spreadsheets/d/1nz31-E6E92Xzmw7JpUP6YQdc9UnYQcdb6OwWXQoDg7s/export?format=csv"

response = requests.get(url)
data = StringIO(response.text)

df = pd.read_csv(data)
lesson_df = pd.read_csv("https://docs.google.com/spreadsheets/d/1nz31-E6E92Xzmw7JpUP6YQdc9UnYQcdb6OwWXQoDg7s/export?format=csv")
def main():
    st.title('時間割作成アプリ')
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv", key="unique_file_uploader_key")


    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
        generate_timetable(data) #時間割作成関数を実行

x = {}
y = {}
z = {}
#x_曜日_時限_学年_クラス_授業
for d in week:
    for p in period:
        for g in grade_list:
            for c in class_dict[g]:
                for s in subject_list:
                    x[d,p,g,c,s] = pulp.LpVariable(cat="Binary",name=f"x_{d}_{p}_{g}_{c}_{s}")

#y_曜日_時限_教員
for d in week:
    for p in period:
        for t in teacher_list:
            y[d,p,t] = pulp.LpVariable(cat="Binary",name=f"y_{d}_{p}_{t}")

#z_曜日_時限_学年
for d in week:
    for p in period:
        for g in grade_list:
            z[d,p,g] = pulp.LpVariable(cat="Integer",name=f"z_{d}_{p}_{g}")

#(1)1 つの時限では必ず 1 つ授業を行う
for d in week:
    for p in period:
        for g in grade_list:
            for c in class_dict[g]:
                model += pulp.lpSum([x[d,p,g,c,s] for s in subject_list]) == 1

#(2)各教科sは1週間の必要授業数だけ行う
for g in grade_list:
    for c in class_dict[g]:
        for s in subject_list:
            model += pulp.lpSum([x[d,p,g,c,s] for d in week for p in period]) == subject_dict[s]

#(3)教科は 1 日の授業数の上下限を守る
for d in week:
    for g in grade_list:
        for c in class_dict[g]:
            for s in subject_list:
                model += pulp.lpSum([x[d,p,g,c,s] for p in period]) <= 1
#yをxの関数として定義 y=f(x)
for d in week:
    for p in period:
        for g in grade_list:
            for c in class_dict[g]:
                for s in subject_list:
                    df = lesson_df[(lesson_df["gr"] == g) & (lesson_df["cl"] == c)]
                    if df[s].empty:
                        print(f"No data for cl = {c}, s = {s}")
                    else:
                        t = df[s].values[0]
                        if (d, p, t) in y:
                            model += y[d,p,t] >= x[d,p,g,c,s]  # <- ここを修正

#(6)1教員が1日に行う授業数の上下限を守る
for d in week:
    for t in teacher_list:
        model += pulp.lpSum([y[d,p,t] for p in period]) <= 6
        model += pulp.lpSum([y[d,p,t] for p in period]) >= 4

for d in week:
    for p in period:
        for g in grade_list:
            z[d,p,g] = list(teacher_dict.values()).count(g) - pulp.lpSum([y[d,p,t] for t in [a for a in teacher_list if teacher_dict[a] == g]])

model += pulp.lpSum([np.max([z[d,p,g] for g in grade_list]) - np.min([z[d,p,g] for g in grade_list]) for d in week for p in period])

model.solve()

def export_table(g,c):
    timetable_df = pd.DataFrame(index=period, columns=week)

    for d in week:
        for p in period:
            assigned = False
            for s in subject_list:
                if x[d,p,g,c,s].value() == 1.0:
                    timetable_df.at[p, d] = s  # DataFrameに授業名を代入
                    assigned = True
            if not assigned:
                timetable_df.at[p, d] = "未割り当て"  # 未割り当ての場合の処理

    print(timetable_df)

export_table(3,1)

def generate_timetable(lesson_df):
    # この部分にモデルの定義や最適化のコードを入れる
    
    # 最適解の確認と結果の表示
    if model.solve() == pulp.LpStatusOptimal:
        st.write("最適解を見つけました！")
        export_table(3,1)
    else:
        st.write("最適解を見つけることができませんでした。")
if __name__ == "__main__":
    main()

# デバッグのためのコード
g = 1  # ここで適切な値を設定
c = 1  # ここで適切な値を設定
df = lesson_df[lesson_df["gr"] == g]
s = "数学"  # 例として数学を設定
if df[df["cl"] == c][s].empty:
    print(f"No data for cl = {c}, s = {s}")
else:
    t = df[df["cl"] == c][s].values[0]

model = pulp.LpProblem("model",pulp.LpMinimize)

# ・・・（略）

def define_model(lesson_df):
    # ここで model の定義や制約の追加を行います。
    model = pulp.LpProblem("model",pulp.LpMinimize)
    
    # 以前のコードの model の定義や制約の追加の部分をここに移動
    
    return model

def generate_timetable(lesson_df):
    model = define_model(lesson_df)  # モデルの定義
    result_status = model.solve()  # 最適化の実行
    
    # 最適解の確認と結果の表示
    if result_status == pulp.LpStatusOptimal:
        st.write("最適解を見つけました！")
        export_table(3,1)
    elif result_status == pulp.LpStatusInfeasible:
        st.write("モデルが不可能です。制約を再確認してください。")
    else:
        st.write("最適解を見つけることができませんでした。")
                
if __name__ == "__main__":
    main()

def main():
    st.title('時間割作成アプリ')
    # 初めてアプリを実行するかどうかをチェック
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None

    # セッション状態にuploaded_dataが存在しない場合、アップローダーを表示
if st.session_state.uploaded_data is None:
        uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv", key="unique_file_uploader_key")

if uploaded_file is not None:
    st.session_state.uploaded_data = pd.read_csv(uploaded_file)
    st.write(st.session_state.uploaded_data)
    generate_timetable(st.session_state.uploaded_data)  # 時間割作成関数を実行
else:
        # セッション状態にuploaded_dataが存在する場合、そのデータを表示
    st.write(st.session_state.uploaded_data)
    generate_timetable(st.session_state.uploaded_data)  # 時間割作成関数を実行