"""Frameworks for running multiple Streamlit applications as a single app.
"""
import random
import streamlit as st
try:
    from streamlit_player import st_player
except:
    pass


class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        st.set_page_config(page_title="離散事件模擬", page_icon="🎬")

        try:
            with st.sidebar:
                music = {
                    1: "https://soundcloud.com/audio-library-478708792/leaning-on-the-everlasting-arms-zachariah-hickman-audio-library-free-music?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing",
                    2: "https://soundcloud.com/user-443256645/esther-abrami-no-9-esthers?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing",
                    3: "https://soundcloud.com/audio_lava/hulu-ukulele?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing",
                    4: "https://soundcloud.com/xzammopcelmf/sbu4e1m2v1mt?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing",
                }
                i = random.randint(1, len(music))
                st_player(music[1], playing=False, loop=True, volume=0.3, height=250, light=True)
        except:
            pass

        st.sidebar.title("👨‍🏫 [Jack.Pan's](https://www.facebook.com/jack.pan.96/) 離散事件模擬")
        st.sidebar.write('- ✉️ssp6258@yahoo.com.tw')
        st.sidebar.write('- 🚧️ [故障報修、意見反饋](https://github.com/SSP6258/des_app/issues)')

        st.sidebar.header('🧭 功能導航')
        app = st.sidebar.selectbox(
            '應用選單',
            self.apps,
            format_func=lambda app: app['title'],
            index=1)



        app['function']()
