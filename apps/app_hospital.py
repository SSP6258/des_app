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


dic_record = {
    'task_id': [],
    'status': [],
    'tick': [],
    'prio': [],
}

"""
Simulation Env Config
"""

RESOURCE_NUM = 2
TASK_NUM = 10
PROC_TIME = 30
PEAK_ARRIVAL_CLOCK = 24
ARRIVAL_DURATION = 1
SHOW_PREEMPT=True
# ============================
dic_sim_cfg = {
    'RESOURCE_NUM': RESOURCE_NUM,
    'TASK_NUM': TASK_NUM,
    'PROC_TIME': PROC_TIME,
    'PEAK_ARRIVAL_CLOCK': PEAK_ARRIVAL_CLOCK,
    'ARRIVAL_DURATION': ARRIVAL_DURATION,
    'SHOW_PREEMPT': SHOW_PREEMPT,
}
# =============================


def fn_record_it(tick, name, prio, status):
    dic_record['tick'].append(tick)
    dic_record['task_id'].append(name)
    dic_record['prio'].append(round(prio, 1))
    dic_record['status'].append(status)


def resource_user(name, env, resource, wait, prio, excu_time):
    yield env.timeout(wait)
    timeLeft = excu_time
    while timeLeft > 0:
        with resource.request(priority=prio) as req:
            fn_record_it(env.now, name, prio, 'req')
            yield req
            status = 'got' if timeLeft == excu_time else 'got_resumed'
            fn_record_it(env.now, name, prio, status)
            try:
                yield env.timeout(timeLeft)
                timeLeft = 0
                fn_record_it(env.now, name, prio, 'done')
            except simpy.Interrupt as interrupt:
                by = interrupt.cause.by
                usage = env.now - interrupt.cause.usage_since
                timeLeft -= usage
                fn_record_it(env.now, name, prio, 'preempted')
                if resource.capacity <= 1:
                    prio -= 0.1  # bump my prio enough so I'm next


def fn_add_date_line(fig, df, date, mode='lines', width=10, color='lightgreen', dash=None, op=None):
    # print(df.iloc[:, 0])
    # print(df.iloc[0, 0], df.iloc[-1, 0])
    fig.add_trace(
        go.Scatter(
            x=[date, date],
            y=[df.iloc[0, 0], df.iloc[-1, 0]],
            mode=mode,
            line=go.scatter.Line(color=color, width=width, dash=dash),
            showlegend=False,
            opacity=op,
        )
    )
    return fig


def fn_gen_plotly_hist(fig, data, name, row=1, col=1, margin=None, bins=100, line_color='white', showlegend=False,
                       legendgroup=None, hovertext=None, barmode='group', opacity=0.8, xaxis_range=None):
    fig.add_trace(
        go.Histogram(x=data, name=name, showlegend=showlegend, nbinsx=bins, hovertext=hovertext, legendgroup=legendgroup,
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


def fn_gen_plotly_gannt(df, x_s, x_e, y, margin=None, color=None, op=None, title=None, hover=None, text=None, x_typ=None):

    margin = {'l': 0, 'r': 100, 't': 60, 'b': 20} if margin is None else margin

    if x_typ == 'time':
        df[x_s] = fn_2_timestamp(df[x_s].values)
        df[x_e] = fn_2_timestamp(df[x_e].values)

    fig = px.timeline(df, x_start=x_s, x_end=x_e, y=y, color=color, text=text, color_continuous_scale='Spectral',
                      opacity=op, hover_data=hover)

    fig.update_traces(textposition='outside')

    fig.update_yaxes(autorange="reversed", title={'text': ''},
                     showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikedash='dot')

    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=False, spikedash='dot')

    if x_typ == 'time':
        fig.update_xaxes(tickformat="%H:%M")
    else:
        fig.layout.xaxis.type = 'linear'
        df['delta'] = df[x_e] - df[x_s]
        fig.data[0].x = df.delta.tolist()

    fig.update_layout(margin=margin,
                      title={
                          'text': f'{title}',
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'
                      },
                      xaxis=dict(
                          title='時間'
                      ),
                      yaxis=dict(
                          title='病患_級別'
                      ),
                      coloraxis_colorbar=dict(
                          title="檢傷分級",
                      ),
                      )

    return fig


def fn_2_timestamp(values):
    try:
        time_stamp = [datetime.datetime.utcfromtimestamp(t * 60) for t in values]
    except:
        print(values)
        time_stamp = [datetime.datetime.utcfromtimestamp(t * 60) for t in values]

    return time_stamp


def fn_sim_result_render(df, capacity, x_typ='linear', show_preempt=True):

    df['task_pri'] = df['task_id'].astype(str)+'_'+df['prio'].astype(str)
    df_s = df[df['status'] == 'req'].sort_values(by='task_pri')
    df_s = df_s.reset_index()

    df_e = df[df['status'].apply(lambda x: x == 'done' or x == 'preempted')].sort_values(by='task_pri')
    df_e.columns = [c+'_e' for c in df_e.columns]
    df_e = df_e.reset_index()

    df_se = pd.concat([df_s, df_e], axis=1)
    df_se = df_se.sort_values(by='prio', ascending=False)
    df_se = df_se[['task_pri', 'tick', 'tick_e', 'prio']]

    df_g = df[df['status'].apply(lambda x: 'got' in x)].sort_values(by='task_pri')
    df_g = df_g.reset_index()

    df_ge = pd.concat([df_g, df_e], axis=1)
    df_ge = df_ge[['task_pri', 'tick', 'tick_e', 'prio']]

    df_all = pd.concat([df_se, df_ge], axis=0)
    # df_all = df_ge
    df_all = df_all.sort_values(by='prio', ascending=False)
    df_all = df_all.reset_index()
    df_all = df_all[['task_pri', 'tick', 'tick_e', 'prio']]
    df_all.drop_duplicates(subset=['task_pri', 'tick', 'tick_e'], keep='first', inplace=True)

    print(df_se)
    print(df_ge)

    df1 = df_se.copy()
    df1['delta'] = df1['tick_e'] - df1['tick']
    # df1_g = pd.DataFrame(df1.groupby('task_pri', as_index=True)['delta'].sum())

    print(df1)
    # print(df1_g)

    wait_max = df1['delta'].max()
    df1 = df1[df1['delta']==wait_max]
    p = df1['task_pri'].values[0]
    who = p.split("_")[0]
    lev = int(round(float(p.split("_")[-1]), 0))

    icon = '😵' if wait_max < 60 else '🥴'
    wait_max = str(int(wait_max/60))+'小時'+str(wait_max%60)+'分鐘' if wait_max >= 60 else str(wait_max)+'分鐘'
    title = f'模擬: {capacity}位急診醫師 {dic_sim_cfg["TASK_NUM"]}位病患 5類檢傷分級<br>' \
            f'結果: {who} 等級{lev} 等待最久: {wait_max} {icon}'
    fig = fn_gen_plotly_gannt(df_all, 'tick', 'tick_e', 'task_pri', color='prio', op=0.5,
                              title=title, text='task_pri', x_typ=x_typ)

    p_ticks = df[df['status'] == 'preempted']['tick'].values
    r_ticks = df[df['status'] == 'got_resumed']['tick'].values

    if show_preempt:
        if x_typ == 'time':
            p_ticks = fn_2_timestamp(p_ticks.copy())
            r_ticks = fn_2_timestamp(r_ticks.copy())

        for t in p_ticks:
            fig = fn_add_date_line(fig, df_all, t, dash='dash', color='orangered', width=2, op=0.5)

        for t in r_ticks:
            fig = fn_add_date_line(fig, df_all, t, dash='dash', color='green', width=2, op=0.5)

    st.write('-  🚑  [急診 檢傷分級: 1.復甦急救、2.危急、3.緊急、4.次緊急、5.非緊急](https://www.mgems.org/index.php/zh/question-answer/hospital-ems-triage)')

    st.plotly_chart(fig)

    with st.expander('檢視詳細資料'):
        st.write('')
        AgGrid(df, theme='blue')
        st.write('')
        AgGrid(df_all, theme='blue')


def fn_sim_init():
    global dic_sim_cfg, dic_record

    dic_record = {k: [] for k in dic_record.keys()}

    mu = dic_sim_cfg['PEAK_ARRIVAL_CLOCK'] * 60
    sig = dic_sim_cfg['ARRIVAL_DURATION'] * 60
    ARRIVAL_TIMES = [int(random.gauss(mu, sig)) for _ in range(dic_sim_cfg['TASK_NUM'])]
    ARRIVAL_TIMES = [t+1 if t in ARRIVAL_TIMES else t for t in ARRIVAL_TIMES]
    ARRIVAL_TIMES.sort()
    dic_sim_cfg['ARRIVAL_TIMES'] = ARRIVAL_TIMES
    dic_sim_cfg['PRIORITY'] = [min(5, max(1, int(random.gauss(3.5, 0.8)))) for _ in range(dic_sim_cfg['TASK_NUM'])]

    print(dic_sim_cfg)


def fn_sim_fr_st():
    st.title('離散事件模擬器')
    st.subheader('🏥 應用: 急診很忙~ ')
    st.subheader('⏳ 場景: 急診需要等多久 ?')
    global dic_sim_cfg

    with st.form(key='task'):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        # dic_sim_cfg['RESOURCE_NUM'] = c1.slider('幾位急診醫師?', min_value=1, max_value=5, value=RESOURCE_NUM, step=1)
        dic_sim_cfg['RESOURCE_NUM'] = c1.selectbox('幾位急診醫師?', range(1, 5), RESOURCE_NUM-1)
        dic_sim_cfg['TASK_NUM'] = c2.selectbox('幾位病患?', range(1, TASK_NUM + 5), TASK_NUM - 1)
        dic_sim_cfg['PROC_TIME'] = c3.selectbox('看診需要幾分鐘?', range(10, 60, 10), list(range(10, 60, 10)).index(PROC_TIME))
        dic_sim_cfg['SHOW_PREEMPT'] = c4.selectbox('顯示中斷?', [True, False], 1)
        submitted = st.form_submit_button('開始模擬')

        if submitted:
            fn_sim_init()
            t1 = datetime.datetime.now()
            fn_sim_main()
            t2 = datetime.datetime.now()
            du = t2 - t1
            st.write(f'模擬時間: {du.microseconds} 微秒(us)')

            # st.write('🚑')
            # pprint.pprint(dic_sim_cfg)

            df = pd.DataFrame(dic_record)
            df = df[['tick', 'task_id', 'prio', 'status']]

            fn_sim_result_render(df, dic_sim_cfg['RESOURCE_NUM'], x_typ='time', show_preempt=dic_sim_cfg['SHOW_PREEMPT'])


def fn_sim_main():
    env = simpy.Environment()
    res = simpy.PreemptiveResource(env, capacity=dic_sim_cfg['RESOURCE_NUM'])

    for i in range(dic_sim_cfg['TASK_NUM']):
        env.process(resource_user('病患'+str(i), env, res,
                                  wait=dic_sim_cfg['ARRIVAL_TIMES'][i],
                                  prio=dic_sim_cfg['PRIORITY'][i],
                                  excu_time=dic_sim_cfg['PROC_TIME']))

    env.run()


def app():
    fn_sim_fr_st()


if __name__ == '__main__':
    fn_sim_fr_st()
