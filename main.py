pip install streamlit

# 必要なライブラリをインポート
import streamlit as st
import pandas as pd

# 上記の時間割プログラムの内容を関数や変数として取り入れる（上記のプログラムコードをここにコピペ）

# Streamlitアプリのメイン関数
def main():
    st.title('時間割アプリ')

    # 学年とクラスを選択するためのウィジェットを追加
    selected_grade = st.selectbox("学年を選択", grade_list)
    selected_class = st.selectbox(f"{selected_grade}年生のクラスを選択", class_dict[selected_grade])

    # 「時間割を生成」ボタンをクリックしたときのアクションを定義
    if st.button('時間割を生成'):
        model.solve()
        timetable_df = export_table(selected_grade, selected_class)
        st.dataframe(timetable_df)  # Streamlitで時間割を表示

def export_table(g, c):
    timetable_df = pd.DataFrame(index=period, columns=week)
    for d in week:
        for p in period:
            for s in subject_list:
                if x[d, p, g, c, s].value() == 1.0:
                    timetable_df.at[p, d] = s
    return timetable_df  # DataFrameを返すように変更

if __name__ == '__main__':
    main()
