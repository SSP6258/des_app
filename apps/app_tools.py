import streamlit as st


def app():
    st.title('🧰 開發工具')
    st.write('')
    st.subheader('程式碼:')
    st.write("- GitHub Repo: [des_app](https://github.com/SSP6258/des_app)")

    st.write('')
    st.subheader('網頁製作:')
    st.write("- 純Python的極速網頁製作套件: [Streamlit](https://streamlit.io/)")
    st.write(
        "- Streamlit multi page framework: [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps)")
    st.write("- 畫文字 表情符號: [Emojipedia](https://emojipedia.org/)")
    st.write("- 影音嵌入: [Streamlit-player](https://github.com/okld/streamlit-player)")
    st.write("- 音樂庫: [SoundCloud](https://soundcloud.com/)")

    st.write('')
    st.subheader('函式庫:')
    st.write("- 離散事件模擬框架: [Simpy](https://simpy.readthedocs.io/en/latest/)")
