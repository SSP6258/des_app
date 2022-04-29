import streamlit as st
from streamlit_player import st_player
import simpy
import datetime
from statistics import mean
import random
import pprint
import pandas as pd
from st_aggrid import AgGrid
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

"""
Simulation Env Config
"""

CASHIER_TIME = 6
CASHIER_NUM = 2
CUSTOMER_NUM = 50
PEAK_ARRIVAL_CLOCK = 12
ARRIVAL_DURATION = 1
# ============================
dic_sim_cfg = {
    'CUSTOMER_NUM': CUSTOMER_NUM,
    'CASHIER_NUM': CASHIER_NUM,
    'CASHIER_TIME': CASHIER_TIME,
    'PEAK_ARRIVAL_CLOCK': PEAK_ARRIVAL_CLOCK,
    'ARRIVAL_DURATION': ARRIVAL_DURATION,
}
# =============================

# MU = PEAK_ARRIVAL_CLOCK * 60
# SIG = ARRIVAL_DURATION * 60
# ARRIVAL_TIMES = [int(random.gauss(MU, SIG)) for _ in range(CUSTOMER_NUM)]
# ARRIVAL_TIMES.sort()
# ARRIVAL_TIMES_CPY = ARRIVAL_TIMES.copy()
# SIM_TIME = ARRIVAL_TIMES[-1] + 60

ARRIVAL_TIMES = [0]
ARRIVAL_TIMES_CPY = [0]
SIM_TIME = 0

MUSIC = "https://soundcloud.com/xzammopcelmf/sbu4e1m2v1mt?utm_source=clipboard&utm_medium=text&utm_campaign=social_sharing"

"""
Simulation Records
"""
dic_record = {
    'arrival': [],
    'time': [],
    'queue': [],
    'custom_id': [],
    'wait_time': [],
    'done_time': [],
}


def fn_gen_plotly_hist(fig, data, title, row=1, col=1, margin=None, bins=100, line_color='white', showlegend=False,
                       hovertext=None, barmode='group', opacity=0.8, xaxis_range=None):
    fig.add_trace(
        go.Histogram(x=data, name=title, showlegend=showlegend, nbinsx=bins, hovertext=hovertext,
                     marker=dict(
                         opacity=opacity,
                         line=dict(
                             color=line_color, width=0.4
                         ),
                     )),
        row=row,
        col=col,
    )

    fig.update_layout(margin=margin,
                      barmode=barmode,
                      xaxis_range=xaxis_range)

    return fig


def fn_gen_plotly_scatter(fig, x_data, y_data, row=1, col=1, margin=None, color=None, text=None, opacity=0.8,
                          xlabel=None, ylabel=None, title=None, size=None, marker_sym=None,
                          legend=False, name=None, line_shape=None, mode=None, xaxis_range=None):

    fig.add_trace(go.Scatter(x=x_data, y=y_data, line_shape=line_shape, mode=mode, showlegend=legend,
                             marker_symbol=marker_sym, name=name,
                             marker=dict(size=size,
                                         opacity=opacity,
                                         line={'color': 'white', 'width': 1},
                                         color=color)
                             ), row=row, col=col)

    fig.update_layout(margin=margin, xaxis_range=xaxis_range)

    return fig


def fn_gen_gannt_chart(df, x_s, x_e, y, margin=None, color=None, op=None, title=None, hover=None):

    margin = {'l': 0, 'r': 100, 't': 30, 'b': 20} if margin is None else margin
    fig = px.timeline(df, x_start=x_s, x_end=x_e, y=y, color=color,
                      hover_data=hover,
                      color_continuous_scale='portland', opacity=op)
    fig.update_yaxes(autorange="reversed", title={'text': ''})
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_layout(margin=margin,
                      title={
                          'text': f'{title}',
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'
                      }, )

    return fig



def fn_2_timestamp(values):
    time_stamp = [datetime.datetime.utcfromtimestamp(t * 60) for t in values]
    return time_stamp


def fn_sim_init():
    global dic_sim_cfg, dic_record, ARRIVAL_TIMES, ARRIVAL_TIMES_CPY, SIM_TIME

    dic_record = {k: [] for k in dic_record.keys()}

    mu = dic_sim_cfg['PEAK_ARRIVAL_CLOCK'] * 60
    sig = dic_sim_cfg['ARRIVAL_DURATION'] * 60
    ARRIVAL_TIMES = [int(random.gauss(mu, sig)) for _ in range(dic_sim_cfg['CUSTOMER_NUM'])]
    ARRIVAL_TIMES.sort()
    ARRIVAL_TIMES_CPY = ARRIVAL_TIMES.copy()
    SIM_TIME = ARRIVAL_TIMES[-1] + 60
    dic_record['arrival'] = ARRIVAL_TIMES

    print(f'fn_sim_init {len(ARRIVAL_TIMES)} {SIM_TIME}')


def fn_sim_customer(env, res, log=True):
    custom = 0
    for t in ARRIVAL_TIMES:
        custom += 1
        yield env.timeout(t - env.now)
        yield res.put(1)
        print(f'CUSTOM {custom} time {env.now} queue {res.level}') if log else None
        dic_record['queue'].append(res.level)
        dic_record['time'].append(env.now)
        dic_record['custom_id'].append(custom)

        # dic_record['arrival'].append(t)
        # dic_record['wait_time'].append(-1)


def fn_sim_cashier(env, res, log=True):
    global ARRIVAL_TIMES_CPY
    while True:
        if len(ARRIVAL_TIMES_CPY):
            dic_record['wait_time'].append(max(0, env.now - ARRIVAL_TIMES_CPY[0]))
            ARRIVAL_TIMES_CPY.pop(0)
        yield res.get(1)
        yield env.timeout(dic_sim_cfg['CASHIER_TIME'])
        print(f'CASHIER N Time {env.now} queue {res.level}') if log else None
        dic_record['queue'].append(res.level)
        dic_record['time'].append(env.now)
        # dic_record['custom_id'].append(-1)
        # dic_record['arrival'].append(-1)


def fn_sim_main(log=True):
    env = simpy.Environment()
    res = simpy.Container(env)
    env.process(fn_sim_customer(env, res, log))
    for c in range(dic_sim_cfg['CASHIER_NUM']):
        env.process(fn_sim_cashier(env, res, log))

    env.run(until=SIM_TIME)

    # dic_record['done_time'] = [dic_record['arrival'][i] + dic_record['wait_time'][i] for i in
    #                            range(len(dic_record['wait_time']))]


def fn_sim_fr_st():
    st.title('離散事件模擬器')
    st.subheader('🛒 應用: 請支援收銀~ ')
    st.subheader('🔊 場景: 全聯福利中心 何時需要廣播 "請支援收銀~" ?')
    global dic_sim_cfg

    with st.form(key='sale1'):
        c1, c2, c3 = st.columns([2, 1, 1])
        dic_sim_cfg['CUSTOMER_NUM'] = c1.slider('幾位顧客?', min_value=5, max_value=100, value=CUSTOMER_NUM, step=1)
        dic_sim_cfg['CASHIER_NUM'] = c2.selectbox('幾個收銀員?', range(1, CASHIER_NUM + 5), CASHIER_NUM - 1)
        dic_sim_cfg['CASHIER_TIME'] = c3.selectbox('收銀需要幾分鐘?', range(1, CASHIER_TIME + 5), CASHIER_TIME - 1)
        submitted = st.form_submit_button('開始模擬')

        if submitted:
            fn_sim_init()
            t1 = datetime.datetime.now()
            fn_sim_main(log=False)
            t2 = datetime.datetime.now()
            st.write(f'模擬時間: {t2 - t1}')

            st.write('')
            st.write('歡迎光臨 全聯福利中心 🎵~ ')
            st_player(MUSIC, key=str(datetime.datetime.now()), playing=submitted, loop=True, volume=0.3, height=250)

            # pprint.pprint(dic_sim_cfg)
            fn_sim_result_render()


def fn_sim_result_render():
    # dic_record = {
    #     'arrival': [],
    #     'time': [],
    #     'queue': [],
    #     'custom_id': [],
    #     'wait_time': [],
    #     'done_time': [],
    # }

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dic_record.items()]))
    df.reset_index(inplace=True)
    df['done_time'] = df['arrival'] + df['wait_time']
    df_all = df.copy()
    df['arrival'] = df['arrival'].fillna(-1)
    df['arrival'] = df['arrival'].astype(int)
    df = df[df['arrival'] > 0]

    df['arrival_time'] = fn_2_timestamp(df['arrival'].tolist())
    df_all['tick_time'] = fn_2_timestamp(df_all['time'].tolist())

    fig = make_subplots(rows=2, cols=1, subplot_titles=('顧客人數分布', '排隊人數模擬'))
    margin = {'l': 0, 'r': 30, 't': 20, 'b': 0}

    x0 = df['arrival_time']
    xaxis_range = [df_all['tick_time'].min(), df_all['tick_time'].max()]
    fig = fn_gen_plotly_hist(fig, x0, '顧客分布', row=1, col=1, bins=df.shape[0], margin=margin, xaxis_range=xaxis_range, showlegend=True)

    x1 = df_all['tick_time']
    y1 = df_all['queue']
    fig = fn_gen_plotly_scatter(fig, x0, [1 for _ in x0], margin=margin, color='royalblue', size=14, marker_sym=6, row=2,
                                opacity=0.5, mode='markers', xaxis_range=xaxis_range, name='顧客抵達', legend=True)

    x2 = fn_2_timestamp([t+dic_sim_cfg['CASHIER_TIME'] for t in df['done_time'].values])
    fig = fn_gen_plotly_scatter(fig, x2, [0 for _ in x2], margin=margin, color='green', size=14, marker_sym=5, row=2,
                                opacity=0.5, mode='markers', xaxis_range=xaxis_range, name='顧客離開', legend=True)

    fig = fn_gen_plotly_scatter(fig, x1, y1, margin=margin, color='red', size=10, row=2, opacity=0.5, line_shape='hv',
                                mode='lines', xaxis_range=xaxis_range, name='排隊人數', legend=True)

    df_gannt = df.copy()
    df_gannt['done_time_tick'] = fn_2_timestamp(df_gannt['done_time'].values)
    df_gannt['duration'] = fn_2_timestamp(df_gannt['wait_time'].values)
    fig_gannt = fn_gen_gannt_chart(df_gannt, 'arrival_time', 'done_time_tick', 'custom_id', margin=None, color='wait_time', op=0.8, title='顧客等待時間', hover=['wait_time'])

    st.write('')
    st.plotly_chart(fig)
    st.write('')
    st.plotly_chart(fig_gannt)

    st.write('')
    with st.expander('檢視詳細資料'):
        st.write(dic_sim_cfg)
        st.write('')
        AgGrid(df_all, theme='blue')
        AgGrid(df[['custom_id', 'arrival_time', 'done_time', 'wait_time']], theme='blue')


def app():
    # fn_sim_init()
    fn_sim_fr_st()


if __name__ == '__main__':
    fn_sim_init()
    fn_sim_main()
    pprint.pprint(dic_record)
