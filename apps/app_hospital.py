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
import plotly.colors

dic_record = {
    'task_id': [],
    'status': [],
    'tick': [],
    'prio': [],
    'queue': [],
}

"""
Simulation Env Config
"""

RESOURCE_NUM = 1
TASK_NUM = 10
PROC_TIME = 30
PEAK_ARRIVAL_CLOCK = 24
ARRIVAL_DURATION = 3
SHOW_PREEMPT = True
PRIO_MAX = 5
SHOW_TYP = "PROC_AND_WAIT"
# ============================
dic_sim_cfg = {
    'RESOURCE_NUM': RESOURCE_NUM,
    'TASK_NUM': TASK_NUM,
    'PROC_TIME': PROC_TIME,
    'PEAK_ARRIVAL_CLOCK': PEAK_ARRIVAL_CLOCK,
    'ARRIVAL_DURATION': ARRIVAL_DURATION,
    'SHOW_PREEMPT': SHOW_PREEMPT,
    'PRIO_MAX': PRIO_MAX,
    'IS_PROFILE_EN': True,
    'SHOW_TYP': SHOW_TYP,
    'RANDOM_SEED': None,
    'IS_PROC_TIME_FIX': True,
}


# =============================


def fn_record_it(tick, name, prio, status, queue):
    dic_record['tick'].append(tick)
    dic_record['task_id'].append(name)
    dic_record['prio'].append(round(prio, 1))
    dic_record['status'].append(status)
    dic_record['queue'].append(queue)


def fn_add_v_line(fig, x, width=10, color='lightgreen', dash=None, op=None):
    fig.add_vline(x=x, line_width=width, line_dash=dash, line_color=color, opacity=op)
    return fig


def fn_gen_plotly_scatter(fig, x_data, y_data, row=1, col=1, margin=None, color=None, text=None, opacity=0.8,
                          xlabel=None, ylabel=None, title=None, size=None, marker_sym=None,
                          legend=False, legendgroup=None, name=None, line_shape=None, mode=None, xaxis_range=None):
    fig.add_trace(go.Scatter(x=x_data, y=y_data, line_shape=line_shape, mode=mode, showlegend=legend,
                             marker_symbol=marker_sym, name=name, legendgroup=legendgroup,
                             marker=dict(size=size,
                                         opacity=opacity,
                                         line={'color': 'white', 'width': 1},
                                         color=color)
                             ), row=row, col=col)

    fig.update_layout(margin=margin, xaxis_range=xaxis_range, legend_tracegroupgap=225)

    return fig


def fn_gen_plotly_hist(fig, data, name, row=1, col=1, margin=None, bins=100, line_color='white', showlegend=False,
                       legendgroup=None, hovertext=None, bingroup=None, barmode='group', op=0.8, xaxis_range=None,
                       color=None):
    fig.add_trace(
        go.Histogram(x=data, name=name, bingroup=bingroup, showlegend=showlegend, nbinsx=bins, hovertext=hovertext,
                     legendgroup=legendgroup,
                     marker=dict(
                         opacity=op,
                         color=color,
                         line=dict(
                             color=line_color, width=0.5
                         ),
                     )),
        row=row,
        col=col,
    )

    fig.update_layout(margin=margin,
                      barmode=barmode,
                      xaxis_range=xaxis_range,
                      legend_tracegroupgap=225)

    return fig


def fn_gen_plotly_gannt(df, x_s, x_e, y, margin=None, color=None, op=None, title=None, hover=None, text=None,
                        x_typ=None, range_color=(1.0, dic_sim_cfg['PRIO_MAX'])):
    margin = {'l': 0, 'r': 100, 't': 60, 'b': 20} if margin is None else margin

    if x_typ == 'time':
        df[x_s] = fn_2_timestamp(df[x_s].values)
        df[x_e] = fn_2_timestamp(df[x_e].values)

    fig = px.timeline(df, x_start=x_s, x_end=x_e, y=y, color=color, text=text, color_continuous_scale='Spectral',
                      template='plotly', opacity=op, hover_data=hover, range_color=range_color)

    with st.expander("Show RAW data"):
        st.write(df)

    fig = fig.add_annotation(x=df[x_s][16], y=df[y][16],
                             text=f'🚑 {df["task_id"][16]} 到院 👉',
                             showarrow=False,
                             arrowhead=1,
                             arrowsize=2,
                             xshift=-50)

    fig.update_traces(textposition='outside')

    fig.update_yaxes(autorange="reversed", title={'text': ''},
                     showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1, showline=True,
                     spikedash='dot')

    fig.update_xaxes(showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1, showline=False,
                     spikedash='dot')

    if x_typ == 'time':
        fig.update_xaxes(tickformat="%H:%M")
    else:
        fig.layout.xaxis.type = 'linear'
        # df['delta'] = df[x_e] - df[x_s]
        fig.data[0].x = df.delta.tolist()

    fig.update_layout(margin=margin,
                      title={
                          'text': f'{title}',
                          'x': 0.5,
                          'font_size': 15,
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


def fn_check_df_nan_col(df, df_name='unknown_df', is_assert=True):
    for col in df.columns:
        is_nan = df[col].isnull().values.any()
        if is_nan:
            print(df_name, col, df.shape)
            print(df[col])
            assert is_assert, f'{df_name}[{col}] has null val'


def dc_profiler(func):
    def wrapper(*args, **kwargs):
        if dic_sim_cfg["IS_PROFILE_EN"]:
            ts = datetime.datetime.now()
            val = func(*args, **kwargs)
            te = datetime.datetime.now()
            dur = te - ts
            dur_us = dur.microseconds
            if dur_us > 1e6:
                time = f'{int(dur_us / 1e6)} 秒(sec)'
            elif dur_us >= 1e3:
                time = f'{int(dur_us / 1e3)} 毫秒(ms)'
            else:
                time = f'{dur_us} 微秒(us)'

            print(func.__name__, time)
            st.write(f'{func.__name__} 👉 {time}')
        else:
            val = func(*args, **kwargs)
        return val

    return wrapper


@dc_profiler
def fn_sim_result_render(df, capacity, x_typ='linear', show_preempt=True):
    df['task_pri'] = df['task_id'].astype(str) + '_' + df['prio'].astype(str)
    df['task_pri'] = df['task_pri'].apply(lambda x: x.split('.0')[0] + '級')
    df_s = df[df['status'] == 'req'].sort_values(by=['task_pri', 'tick'], ascending=[False, True])
    df_s = df_s.reset_index()

    fn_check_df_nan_col(df_s, 'df_s')

    df_e = df[df['status'].apply(lambda x: x == 'done' or x == 'preempted')].sort_values(by=['task_pri', 'tick'],
                                                                                         ascending=[False, True])
    df_e.columns = [c + '_e' for c in df_e.columns]
    df_e = df_e.reset_index()

    fn_check_df_nan_col(df_e, 'df_e')
    assert df_s.shape[0] == df_e.shape[0], f'df_s {df_s.shape} != df_e {df_e.shape} !'

    df_se = pd.concat([df_s, df_e], axis=1)
    df_se = df_se.sort_values(by='prio', ascending=False)
    df_se = df_se[['task_id', 'task_pri', 'tick', 'tick_e', 'prio']]
    df_se['fr'] = ['df_se' for _ in range(df_se.shape[0])]

    fn_check_df_nan_col(df_se, 'df_se')

    df_g = df[df['status'].apply(lambda x: 'got' in x)].sort_values(by=['task_pri', 'tick'], ascending=[False, True])
    df_g = df_g.reset_index()

    fn_check_df_nan_col(df_g, 'df_g')
    assert df_g.shape[0] == df_e.shape[0], f'df_g {df_g.shape} != df_e {df_e.shape} !'

    df_ge = pd.concat([df_g, df_e], axis=1)
    df_ge = df_ge[['task_id', 'task_pri', 'tick', 'tick_e', 'prio']]
    df_ge['fr'] = ['df_ge' for _ in range(df_ge.shape[0])]

    fn_check_df_nan_col(df_ge, 'df_ge')

    assert df_se.shape[1] == df_ge.shape[1], f'df_se {df_se.shape} != df_ge {df_ge.shape} !'

    df_all = pd.concat([df_se, *[df_ge for _ in range(5)]], axis=0)
    df_all = df_all.sort_values(by='prio', ascending=False)

    fn_check_df_nan_col(df_all, 'df_all')

    df_all = df_all.reset_index()
    df_all = df_all[['task_id', 'task_pri', 'tick', 'tick_e', 'prio', 'fr']]
    df_all['delta'] = df_all['tick_e'] - df_all['tick']

    df1 = df_se.copy()
    df1['delta'] = df1['tick_e'] - df1['tick']
    df1_g = pd.DataFrame(df1.groupby('task_id', as_index=True)['delta'].sum())
    wait_max = df1_g['delta'].max()
    who = df1_g[df1_g['delta'] == wait_max].index[0]
    lev = df1[df1['task_id'] == who]['prio'].values[0]
    lev = int(round(lev, 0))

    icon = '😵' if wait_max < 60 else '🥴'
    wait_max = str(int(wait_max / 60)) + '小時' + str(wait_max % 60) + '分鐘' if wait_max >= 60 else str(wait_max) + '分鐘'
    title = f'模擬: {capacity}位急診醫師 {dic_sim_cfg["TASK_NUM"]}位病患 {dic_sim_cfg["PRIO_MAX"]}類檢傷分級<br>' \
            f'結果: {who}_{lev}級 等待最久 👉 {wait_max} {icon}'

    if dic_sim_cfg['SHOW_TYP'] == 'PROC_ONLY':
        df_gannt = df_ge.copy()
        op = 0.9
    elif dic_sim_cfg['SHOW_TYP'] == 'WAIT_ONLY':
        df_gannt = df_se.copy()
        op = 0.3
    else:
        df_gannt = df_all.copy()
        op = 0.3

    df_gannt = df_gannt.sort_values(by='prio', ascending=False)

    margin = {'l': 0, 'r': 100, 't': 50, 'b': 20}
    fig = fn_gen_plotly_gannt(df_gannt, 'tick', 'tick_e', 'task_pri', margin=margin, color='prio', op=op,
                              title=title, text=None, x_typ=x_typ, range_color=(1.0, dic_sim_cfg['PRIO_MAX']))

    margin = {'l': 90, 'r': 100, 't': 40, 'b': 0}

    df_req = df[df['status'] == 'req']
    df_req.drop_duplicates(subset=['task_id'], keep='first', inplace=True)
    come_time = df_req['tick'].values
    x0 = fn_2_timestamp(come_time) if x_typ == 'time' else come_time
    x1 = fn_2_timestamp(df['tick'].values) if x_typ == 'time' else df['tick']
    y1 = df['queue'].values
    x_range = [min(x1), max(x1)]

    fig_q = make_subplots(rows=2, cols=1,
                          subplot_titles=(f'病患到院時間分布', f'急診候診人數分布 👉 最多 {max(y1)} 人'))
    fig_q = fn_gen_plotly_hist(fig_q, x0, '到院時間', row=1, margin=margin, showlegend=False,
                               legendgroup=1, xaxis_range=x_range)
    if x_typ == 'time':
        fig_q.update_xaxes(tickformat="%H:%M")

    fig_q = fn_gen_plotly_scatter(fig_q, x1, y1, margin=margin, row=2, color='red', size=10, opacity=0.5,
                                  line_shape='hv',
                                  mode='lines', name='候診人數', legend=False, legendgroup='2', xaxis_range=x_range)

    if show_preempt:
        p_ticks = df[df['status'] == 'preempted']['tick'].values
        r_ticks = df[df['status'] == 'got_resumed']['tick'].values
        if x_typ == 'time':
            p_ticks = fn_2_timestamp(p_ticks.copy())
            r_ticks = fn_2_timestamp(r_ticks.copy())

        for t in p_ticks:
            fig = fn_add_v_line(fig, t, dash='dash', color='orangered', width=1.5, op=0.9)

        for t in r_ticks:
            fig = fn_add_v_line(fig, t, dash='dash', color='green', width=1.5, op=0.9)

    df_h = df_all[df_all['fr'] == 'df_se']
    df_h = pd.DataFrame(df_h.groupby('task_id', as_index=True)['delta'].sum())
    df_h['prio'] = [df_all[df_all['task_id'] == idx]['prio'].values[0] for idx in df_h.index]
    # print(df_h)
    fig_h = make_subplots(rows=2, cols=1,
                          subplot_titles=(
                          f'等待時間分布 👉 平均{int(df_h["delta"].mean())}分鐘, 最久{int(df_h["delta"].max())}分鐘',
                          '各級別的等待時間分布 👉 箱型圖 📦 '))

    margin = {'l': 90, 'r': 60, 't': 40, 'b': 0}

    cols = plotly.colors.qualitative._cols
    # plotly.colors.DEFAULT_PLOTLY_COLORS
    c = 0
    for p in sorted(df_h['prio'].unique(), reverse=False):
        df_p = df_h[df_h['prio'] == p]
        fig_h = fn_gen_plotly_hist(fig_h, df_p['delta'], f'{int(p)}級', row=1, col=1, margin=margin,
                                   showlegend=False,
                                   legendgroup=1, bingroup=1, barmode='stack', color=cols[c])

        fig_h.add_trace(go.Box(x=df_p['delta'], name=f'{int(p)}級', legendgroup=2, marker=dict(color=cols[c]), showlegend=False), row=2, col=1)
        c = c + 1

    # fig_h = fn_gen_plotly_hist(fig_h, df_h['delta'], '等待時間(分)', row=1, col=1, margin=margin, showlegend=True,
    #                            legendgroup=1, bingroup=1)

    # fig_h.add_trace(go.Box(x=df_h['delta'], name='等待時間(分)', legendgroup=2), row=2, col=1)

    # =========== Rendering Here ===========

    st.write(
        '-  🚑  [$台灣醫院 急診 檢傷分級: 1.復甦急救 > 2.危急 > 3.緊急 > 4.次緊急 > 5.非緊急$](https://www.mgems.org/index.php/zh/question-answer/hospital-ems-triage)')

    st.plotly_chart(fig, use_container_width=True)
    cols = st.columns([1, 1])
    cols[0].plotly_chart(fig_q, use_container_width=True)
    cols[1].plotly_chart(fig_h, use_container_width=True)


    with st.expander('檢視詳細資料'):
        st.write('')
        AgGrid(df, theme='blue')
        st.write('')
        df_all.drop_duplicates(keep='first', inplace=True)
        AgGrid(df_all, theme='blue')
        st.write('')
        st.write(dic_sim_cfg)


def fn_sim_resource_user(name, env, resource, res_c, wait, prio, proc_time):
    yield env.timeout(wait)
    timeLeft = proc_time
    while timeLeft > 0:
        with resource.request(priority=prio) as req:
            yield res_c.put(1)
            fn_record_it(env.now, name, prio, 'req', res_c.level)

            yield req

            yield res_c.get(1)
            status = 'got' if timeLeft == proc_time else 'got_resumed'
            fn_record_it(env.now, name, prio, status, res_c.level)
            try:
                yield env.timeout(timeLeft)
                timeLeft = 0
                fn_record_it(env.now, name, prio, 'done', res_c.level)
            except simpy.Interrupt as interrupt:
                # by = interrupt.cause.by
                usage = env.now - interrupt.cause.usage_since
                timeLeft -= usage
                fn_record_it(env.now, name, prio, 'preempted', res_c.level)
                prio -= 0.1  # bump my prio enough so I'm next


def fn_sim_init():
    global dic_sim_cfg, dic_record

    dic_record = {k: [] for k in dic_record.keys()}

    seed = dic_sim_cfg['RANDOM_SEED']
    random.seed(seed)

    mu = dic_sim_cfg['PEAK_ARRIVAL_CLOCK'] * 60
    sig = dic_sim_cfg['ARRIVAL_DURATION'] * 60
    ARRIVAL_TIMES = [int(random.gauss(mu, sig)) for _ in range(dic_sim_cfg['TASK_NUM'])]
    ARRIVAL_TIMES = [t + 1 if t in ARRIVAL_TIMES else t for t in ARRIVAL_TIMES]
    ARRIVAL_TIMES.sort()
    dic_sim_cfg['ARRIVAL_TIMES'] = ARRIVAL_TIMES

    mu = dic_sim_cfg['PRIO_MAX'] / 1.4
    sig = 0.9 * dic_sim_cfg['PRIO_MAX'] / PRIO_MAX
    dic_sim_cfg['PRIORITY'] = [min(dic_sim_cfg['PRIO_MAX'], max(1, int(random.gauss(mu, sig)))) for _ in
                               range(dic_sim_cfg['TASK_NUM'])]

    # print(dic_sim_cfg)


@dc_profiler
def fn_sim_main():
    env = simpy.Environment()
    res = simpy.PreemptiveResource(env, capacity=dic_sim_cfg['RESOURCE_NUM'])
    res_c = simpy.Container(env)

    for i in range(dic_sim_cfg['TASK_NUM']):

        if dic_sim_cfg['IS_PROC_TIME_FIX']:
            proc_time = dic_sim_cfg['PROC_TIME']
        else:
            proc_time = int(random.gauss(dic_sim_cfg['PROC_TIME'], 10))

        env.process(fn_sim_resource_user('病患' + str(i + 1), env, res, res_c,
                                         wait=dic_sim_cfg['ARRIVAL_TIMES'][i],
                                         prio=dic_sim_cfg['PRIORITY'][i],
                                         proc_time=proc_time))

    env.run()


def fn_sim_fr_st():
    st.markdown('### $離散事件模擬器$')
    st.markdown('#### 🏥 $應用: 急診很忙~ $')
    st.markdown('#### ⏳ $場景: 看急診要等多久 ?$')
    global dic_sim_cfg

    with st.form(key='task'):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        dic_sim_cfg['TASK_NUM'] = c1.selectbox('幾位急診病患?', range(10, 60, 10), list(range(10, 60, 10)).index(TASK_NUM))
        dic_sim_cfg['RESOURCE_NUM'] = c2.selectbox('幾位急診醫師?', range(1, 6), RESOURCE_NUM - 1)
        dic_sim_cfg['PROC_TIME'] = c3.selectbox('看診需要幾分鐘?', range(10, 60, 10), list(range(10, 60, 10)).index(PROC_TIME))
        dic_sim_cfg['PRIO_MAX'] = c4.selectbox('要分成幾個級別?', range(1, 11), PRIO_MAX - 1)

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        show = c1.radio('顯示 中斷與回復', ['隱藏', '顯示'], 0)
        dic_sim_cfg['SHOW_PREEMPT'] = show == '顯示'

        # show = c2.radio('顯示狀態', ['看診與等待', '看診', '等待'], 0)
        show = c2.radio('顯示狀態', ['看診與等待', '看診'], 0)

        if show == '看診與等待':
            dic_sim_cfg['SHOW_TYP'] = 'PROC_AND_WAIT'
        elif show == '看診':
            dic_sim_cfg['SHOW_TYP'] = 'PROC_ONLY'
        else:
            dic_sim_cfg['SHOW_TYP'] = 'WAIT_ONLY'

        proc_typ = c3.radio('看診時間', [f'固定', f'常態分布'], 0)
        dic_sim_cfg['IS_PROC_TIME_FIX'] = '固定' in proc_typ

        seed = c4.radio('場景', ['固定', '隨機'], 0)
        dic_sim_cfg['RANDOM_SEED'] = 42 if seed == '固定' else None

        st.write('')
        submitted = st.form_submit_button('開始模擬', type="primary")

        if submitted:
            fn_sim_init()
            fn_sim_main()

            df = pd.DataFrame({k: dic_record[k] for k in ['tick', 'task_id', 'prio', 'status', 'queue']})
            df['prio'] = df['prio'].apply(lambda x: float(round(x, 0)))

            fn_sim_result_render(df.copy(), dic_sim_cfg['RESOURCE_NUM'], x_typ='time',
                                 show_preempt=dic_sim_cfg['SHOW_PREEMPT'])

            del df


def app():
    fn_sim_fr_st()


if __name__ == '__main__':
    fn_sim_fr_st()
