#ライブラリのインポート
import pandas as pd
import numpy as np
import pulp

# 必要なライブラリをインポート
import streamlit as st
import pandas as pd

# タイトルとテキストを記入
st.title('Streamlit 時間割')

#基本情報のデータ
teacher_list = [f'教員{i}' for i in range(22)]
subject_list = ["英語","数学","国語","理科","社会","芸術","体育","情報","総合探究","自主自学"]
grade_list = [1,2,3]
class_dict = {3:[1,2,3,4,5],2:[1,2,3,4],1:[1,2,3,4]}
teacher_dict = {t:g for t,g in zip(teacher_list,[3,3,3,2,1,1,3,2,1,3,2,1,2,1,1,3,2,1,3,2,3,2])} #教員の所属学年
period = [1,2,3,4,5,6,7]
week = ["月","火","水","木","金"]
Classroom_mobility = ["芸術","体育","情報","家庭科"] #移動教室授業
six_period = ["総合探究","自主自学"] #6限のみの授業
subject_dict = {s:n for s,n in zip(subject_list,[4,5,5,4,4,2,2,2,1,3])} #必要授業数


lesson_df = pd.read_csv("https://docs.google.com/spreadsheets/d/1nz31-E6E92Xzmw7JpUP6YQdc9UnYQcdb6OwWXQoDg7s/export?format=csv")


model = pulp.LpProblem("model",pulp.LpMinimize)
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




#(4)体育など移動教室は連続しない
for d in week:
    for p in period[:-1]:  # 最後の時限は除く
        for g in grade_list:
            for c in class_dict[g]:
                # 移動教室のみを対象にする
                for s in Classroom_mobility:
                    # 次の時限も存在する場合のみ制約を追加
                    if (d, p+1, g, c, s) in x:
                        model += x[d,p,g,c,s] + x[d,p+1,g,c,s] <= 1


#(5)総合探究と自主自学の制約
#➀総合探究と自主自学は6限
for d in week:
    for p in period[:5]:
        for g in grade_list:
            for c in class_dict[g]:
                model += pulp.lpSum([x[d,p,g,c,s] for s in six_period]) == 0

#➁総合探究と自主自学は学年で曜日を統一して行う
for d in week:
    for g in grade_list:
        for c in class_dict[g][:-1]:
            for s in six_period:
                model += x[d,6,g,c,s] == x[d,6,g,c+1,s]

#➂総合探究と自主自学は異なる学年で同じ時間には行わない
for d in week:
    for s in six_period:
        model += pulp.lpSum(x[d,6,g,1,s] for g in grade_list) <= 1

#yをxの関数として定義 y=f(x)
for d in week:
    for p in period:
        for g in grade_list:
            for c in class_dict[g]:
                for s in subject_list:
                    df = lesson_df[lesson_df["gr"] == g]
                    if df[df["cl"] == c][s].empty:
                      print(f"No data for cl = {c}, s = {s}")
                    else:
                      t = df[df["cl"] == c][s].values[0]
                    if (d, p, t) in y:   # <- Check if the key exists in y
                      y[d,p,t] += x[d,p,g,c,s] # <- ここを修正


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
            for s in subject_list:
                if x[d,p,g,c,s].value() == 1.0:
                    timetable_df.at[p, d] = s  # DataFrameに授業名を代入

    print(timetable_df)

export_table(3,1)


