import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 W2W Funnel Report")

# === 1. Загрузка файлов ===
#st.markdown("### Загрузка файлов данных")
#col1, col2 = st.columns(2)
#with col1:
#    funnel_file = st.file_uploader("CSV событий (воронка)", type="csv", key="funnel_file")
#with col2:
#    costs_file = st.file_uploader("CSV затрат (costs)", type="csv", key="costs_file")

#funnel_path = funnel_file if funnel_file else "all_amplitude_events_with_quiz_id.csv"
#costs_path = costs_file if costs_file else "2025-5-20_21_11_adjust_report_export.csv"

# Чтение с автоопределением разделителя

# Пути к данным (положи их рядом с .py или в папке data/)
funnel_path = "all_amplitude_events_with_quiz_id.csv"      # или "data/all_amplitude_events_with_quiz_id.csv"
costs_path = "2025-5-20_21_11_adjust_report_export.csv"     # или "data/2025-5-20_21_11_adjust_report_export.csv"

def smart_read_csv(path):
    df = pd.read_csv(path)
    if len(df.columns) == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=';')
    return df

df = smart_read_csv(funnel_path)
df['event_time'] = pd.to_datetime(df['event_time'])
df['event_date'] = pd.to_datetime(df['event_date'])

costs_df = smart_read_csv(costs_path)
costs_df['day'] = pd.to_datetime(costs_df['day'])

# === DAILY REPORT: Вчера vs Позавчера ===

# Получаем уникальные даты (UTC или твой таймзон, смотри сам)
all_dates = sorted(df['event_date'].dt.date.unique())
if len(all_dates) >= 2:
    yesterday = all_dates[-1]
    day_before = all_dates[-2]

    df_yesterday = df[df['event_date'].dt.date == yesterday]
    df_day_before = df[df['event_date'].dt.date == day_before]
    costs_yesterday = costs_df[costs_df['day'].dt.date == yesterday]
    costs_day_before = costs_df[costs_df['day'].dt.date == day_before]

    def get_metrics(df_slice, costs_slice):
        quiz_users = df_slice[df_slice['event_type'].str.startswith('Step ')]['user_id'].nunique()
        users_paywall = df_slice[df_slice['event_type'] == 'CompleteRegistration']['user_id'].nunique()
        users_initiate = df_slice[df_slice['event_type'] == 'initiatecheckout']['user_id'].nunique()
        users_purchase = df_slice[df_slice['event_type'] == 'Purchase']['user_id'].nunique()
        spend = costs_slice['cost'].sum()
        cpl = spend / quiz_users if quiz_users > 0 else 0
        cpa = spend / users_purchase if users_purchase > 0 else 0
        return dict(
            quiz_users=quiz_users,
            users_paywall=users_paywall,
            users_initiate=users_initiate,
            users_purchase=users_purchase,
            spend=spend,
            cpl=cpl,
            cpa=cpa
        )

    metrics_y = get_metrics(df_yesterday, costs_yesterday)
    metrics_d = get_metrics(df_day_before, costs_day_before)

    def color_delta(val):
        if val > 0:
            return f"<span style='color:limegreen'>▲ {val:.1f}</span>"
        elif val < 0:
            return f"<span style='color:#e74c3c'>▼ {abs(val):.1f}</span>"
        else:
            return "<span style='color:#aaa'>—</span>"

    report_data = [
        ("Total Spend", metrics_y["spend"], metrics_d["spend"], metrics_y["spend"] - metrics_d["spend"]),
        ("Cost per Lead", metrics_y["cpl"], metrics_d["cpl"], metrics_y["cpl"] - metrics_d["cpl"]),
        ("CPA", metrics_y["cpa"], metrics_d["cpa"], metrics_y["cpa"] - metrics_d["cpa"]),
        ("Quiz Users", metrics_y["quiz_users"], metrics_d["quiz_users"], metrics_y["quiz_users"] - metrics_d["quiz_users"]),
        ("Paywall Users", metrics_y["users_paywall"], metrics_d["users_paywall"], metrics_y["users_paywall"] - metrics_d["users_paywall"]),
        ("Initiate", metrics_y["users_initiate"], metrics_d["users_initiate"], metrics_y["users_initiate"] - metrics_d["users_initiate"]),
        ("Purchase", metrics_y["users_purchase"], metrics_d["users_purchase"], metrics_y["users_purchase"] - metrics_d["users_purchase"]),
    ]

    st.markdown(f"""
    <div style='
        padding: 1em; border-radius: 14px; background: #232324; color: #fff; margin-bottom: 20px;
        border: 2.5px solid #ffe066; font-size: 16px;
    '>
    <h4 style="color:#ffe066; margin:0 0 7px 0;">📈 Динамика: <span style="color:#fff">{yesterday}</span> vs {day_before}</h4>
    <table style="width:100%; font-size:15px;">
    <tr><th align='left'>Метрика</th><th>Вчера</th><th>Позавчера</th><th>Δ</th></tr>
    """ + "\n".join([
        f"<tr><td>{name}</td><td><b>{y:.2f}</b></td><td>{d:.2f}</td><td>{color_delta(delta)}</td></tr>"
        for name, y, d, delta in report_data
    ]) + """
    </table>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Недостаточно данных для динамики за 2 дня.")


# === 2. Фильтры ===
import datetime

min_date = df['event_date'].min()
max_date = df['event_date'].max()
today = max_date.date()
yesterday = today - datetime.timedelta(days=1)

st.markdown("### 📅 Быстрый выбор периода")
date_option = st.radio(
    "Период данных", 
    options=["Сегодня", "Вчера", "Последние 3 дня", "Последние 7 дней", "Выбрать вручную"],
    index=1,  # по умолчанию "Сегодня"
    horizontal=True
)

if date_option == "Сегодня":
    date_from = date_to = today
elif date_option == "Вчера":
    date_from = date_to = yesterday
elif date_option == "Последние 3 дня":
    date_from = today - datetime.timedelta(days=3)
    date_to = today
elif date_option == "Последние 7 дней":
    date_from = today - datetime.timedelta(days=7)
    date_to = today
else:
    col1, col2 = st.columns(2)
    with col1:
        date_from = st.date_input("Start date", min_value=min_date, max_value=max_date, value=today)
    with col2:
        date_to = st.date_input("End date", min_value=min_date, max_value=max_date, value=today)

filtered_df = df[(df['event_date'] >= pd.to_datetime(date_from)) & (df['event_date'] <= pd.to_datetime(date_to))]
costs_period = costs_df[
    (costs_df['day'] >= pd.to_datetime(date_from)) & (costs_df['day'] <= pd.to_datetime(date_to))
]


quiz_ids = filtered_df['quiz_id'].unique()
quiz_id = st.selectbox("Quiz ID", quiz_ids)
quiz_df = filtered_df[filtered_df['quiz_id'] == quiz_id]


# === 3. Собираем шаги воронки ===
paywall_steps = [
    "CompleteRegistration",
    "InitiateCheckout",
    "Paddle checkout.payment.initiated",
    "Paddle checkout.completed"
]

# Step N шаги
step_events = quiz_df[quiz_df['event_type'].str.startswith("Step ")]
step_events = step_events.assign(
    step_num=step_events['event_type'].str.extract(r"Step (\d+)").astype(float)
)
steps_sorted = step_events[['event_type', 'step_num']].drop_duplicates().sort_values('step_num')
step_names = steps_sorted['event_type'].tolist()

# Добавляем paywall шаги если есть
for pw_step in paywall_steps:
    if pw_step in quiz_df['event_type'].unique():
        step_names.append(pw_step)

# === 4. Метрики по шагам ===
users_at_step = []
for step in step_names:
    users_count = quiz_df[quiz_df['event_type'] == step]['user_id'].nunique()
    users_at_step.append(users_count)

total_spend = costs_period['cost'].sum()
cpa_at_step = [total_spend / u if u > 0 else 0 for u in users_at_step]
percent_at_step = [u / users_at_step[0] * 100 if users_at_step[0] > 0 else 0 for u in users_at_step]
conversion_between_steps = [100 if i == 0 else users_at_step[i] / users_at_step[i - 1] * 100 if users_at_step[i - 1] > 0 else 0 for i in range(len(users_at_step))]
dropoff_between_steps = [0 if i == 0 else 100 - conversion_between_steps[i] for i in range(len(users_at_step))]
dropoff_threshold = 20

# === 5. Цвета и подписи ===
bar_colors = []
xtick_labels = []
for i, step in enumerate(step_names):
    is_paywall = step in paywall_steps
    if is_paywall:
        bar_colors.append('mediumvioletred')
    elif i > 0 and dropoff_between_steps[i] >= dropoff_threshold:
        bar_colors.append('crimson')
    else:
        bar_colors.append('cornflowerblue')
    if is_paywall:
        xtick_labels.append(f'<span style="color:mediumvioletred">{step}</span>')
    else:
        xtick_labels.append(step)

paywall_idxs = [i for i, step in enumerate(step_names) if step in paywall_steps]
if paywall_idxs:
    paywall_start = paywall_idxs[0] - 0.5
    paywall_end = paywall_idxs[-1] + 0.5
else:
    paywall_start, paywall_end = None, None

max_drop_idx = np.argmax(dropoff_between_steps[1:]) + 1 if len(dropoff_between_steps) > 1 else 0

hover_text = []
for i in range(len(users_at_step)):
    drop = f"🔻 <b>Drop-off:</b> <span style='color:#e74c3c'>{dropoff_between_steps[i]:.1f}%</span>" if i > 0 else ""
    conv = f"🔁 <b>Conversion:</b> <b>{conversion_between_steps[i]:.1f}%</b>" if i > 0 else ""
    text = (
        f"<b>🔹 {step_names[i]}</b><br>"
        f"<b>👤 Users:</b> {users_at_step[i]}<br>"
        f"<b>🧮 % of leads:</b> {percent_at_step[i]:.1f}%<br>"
        f"<b>💰 CPA:</b> ${cpa_at_step[i]:.2f}<br>"
        f"{conv}<br>"
        f"{drop}"
    )
    hover_text.append(text)

# === 6. SUMMARY BAR ===

# Ключевые расчёты
def unique_users_on_step(step):
    return quiz_df[quiz_df['event_type'] == step]['user_id'].nunique()

users_start = users_at_step[0] if users_at_step else 0
users_paywall = unique_users_on_step("CompleteRegistration")
users_initiate = unique_users_on_step("InitiateCheckout")
users_paddle_initiated = unique_users_on_step("Paddle checkout.payment.initiated")
users_paddle_completed = unique_users_on_step("Paddle checkout.completed")
users_purchase = unique_users_on_step("Purchase")

cr_paywall_to_initiate = (users_initiate / users_paywall * 100) if users_paywall > 0 else 0
cr_paywall_to_purchase = (users_purchase / users_paywall * 100) if users_paywall > 0 else 0
cpa_purchase = (total_spend / users_purchase) if users_purchase > 0 else 0
dropoff_paywall_to_purchase = 100 - cr_paywall_to_purchase if users_paywall > 0 else 0

paddle_success = users_paddle_completed
paddle_fail = unique_users_on_step("Paddle checkout.payment.failed")
paddle_total = paddle_success + paddle_fail
paddle_success_ratio = (paddle_success / paddle_total * 100) if paddle_total > 0 else 0
paddle_fail_ratio = (paddle_fail / paddle_total * 100) if paddle_total > 0 else 0

# ...все фильтры, расчёты пользователей и метрик...

# === Функции расчёта медианного времени (можно где угодно до вызова) ===

def median_time_to_paywall(df):
    times = []
    for user, group in df.groupby('user_id'):
        group = group.sort_values('event_time')
        first_step = group[group['event_type'].str.startswith('Step ')]['event_time']
        paywall = group[group['event_type'] == 'CompleteRegistration']['event_time']
        if not first_step.empty and not paywall.empty:
            delta = (paywall.iloc[0] - first_step.iloc[0]).total_seconds() / 60
            times.append(delta)
    return np.median(times) if times else None

def median_time_paywall_to_purchase(df):
    times = []
    for user, group in df.groupby('user_id'):
        group = group.sort_values('event_time')
        paywall = group[group['event_type'] == 'CompleteRegistration']['event_time']
        purchase = group[group['event_type'] == 'Purchase']['event_time']
        if not paywall.empty and not purchase.empty:
            delta = (purchase.iloc[0] - paywall.iloc[0]).total_seconds() / 60
            times.append(delta)
    return np.median(times) if times else None

# === Считаем значения медианного времени ПОСЛЕ quiz_df ===

median_minutes_to_paywall = median_time_to_paywall(quiz_df)
median_minutes_paywall_to_purchase = median_time_paywall_to_purchase(quiz_df)

# Основные метрики для summary bar
summary_cols_data = [
    ("💸 <span style='color:#ffe066'>Total Spend</span>", f"<b>${total_spend:,.2f}</b>"),
    ("🧮 Cost per Lead", f"<b>${total_spend / users_at_step[0]:.2f}</b>" if users_at_step[0] > 0 else "—"),
    ("⏳ Median time to paywall", f"<b>{median_minutes_to_paywall:.1f} мин</b>" if median_minutes_to_paywall is not None else "—"),
    ("⏳ Median paywall→purchase", f"<b>{median_minutes_paywall_to_purchase:.1f} мин</b>" if median_minutes_paywall_to_purchase is not None else "—"),
    ("📆 Dates", f"<b>{date_from} — {date_to}</b>"),
    ("🔻 Drop-off", f"<b>{step_names[max_drop_idx]}</b> <span style='color:#e74c3c'>({dropoff_between_steps[max_drop_idx]:.1f}%)</span>"),
    ("🟣 Paddle Initiate", f"<span style='color:#ad69fa'><b>{users_paddle_initiated}</b></span>"),
    ("✅ Paddle Success", f"<span style='color:limegreen'><b>{paddle_success} ({paddle_success_ratio:.1f}%)</b></span>"),
    ("❌ Paddle Fail", f"<span style='color:#e74c3c'><b>{paddle_fail} ({paddle_fail_ratio:.1f}%)</b></span>")
]



summary_bar = " &nbsp; | &nbsp; ".join(
    [f"{label}: {value}" for label, value in summary_cols_data]
)

st.markdown(
    f"""
    <div style='
        padding: 0.65em 1.2em;
        border-radius: 12px;
        background: #232324;
        border: 2.5px solid #ffe066;
        margin-bottom: 18px;
        text-align: left;
        font-size: 16px;
        color: #fff;
        line-height: 1.7;
        font-family: Inter, Arial, sans-serif;
        font-weight: 500;
    '>
    {summary_bar}
    </div>
    """,
    unsafe_allow_html=True
)

# ==== SUPER SUMMARY TABLE ====
st.markdown("## 📋 Сводная таблица по воронке, cost и пейволлу")

summary_data = [
    ["Total Spend", f"${total_spend:,.2f}", "Суммарные затраты"],
    ["Users на 1 шаге", users_start, "Вход в воронку"],
    ["Users на CompleteRegistration", users_paywall, "Дошли до paywall"],
    ["Users на initiatecheckout", users_initiate, "Кликнули 'оформить подписку'"],
    ["Users на Paddle checkout.payment.initiated", users_paddle_initiated, "Перешли к оплате в Paddle"],
    ["Users на Paddle checkout.completed", users_paddle_completed, "Завершили оплату (Paddle)"],
    ["Users на Purchase", users_purchase, "Успешная покупка (Purchase event)"],
    ["CR Paywall → Initiatecheckout", f"{cr_paywall_to_initiate:.1f}%", "CR с paywall до кнопки подписки"],
    ["CR Paywall → Покупка", f"{cr_paywall_to_purchase:.1f}%", "CR с paywall до покупки"],
    ["CPA (Cost per Purchase)", f"${cpa_purchase:,.2f}", "Стоимость одного покупателя"],
    ["Drop-off Paywall → Покупка", f"{dropoff_paywall_to_purchase:.1f}%", "Потери на пути с paywall до покупки"]
]

summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value", "Comment"])
st.dataframe(summary_df, hide_index=True, use_container_width=True)




# === 7. График ===
fig = go.Figure()

# PAYWALL RECTANGLE
if paywall_start is not None and paywall_end is not None:
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=paywall_start,
        y0=0,
        x1=paywall_end,
        y1=1,
        fillcolor="mediumpurple",
        opacity=0.12,
        layer="below",
        line_width=0,
    )
    mid = (paywall_start + paywall_end) / 2
    fig.add_annotation(
        x=mid,
        y=1.12,
        xref="x",
        yref="paper",
        showarrow=False,
        text="<b>PAYWALL</b>",
        font=dict(color="mediumvioletred", size=18),
        align="center",
        bgcolor="white",
        opacity=0.7,
    )

fig.add_trace(go.Bar(
    x=xtick_labels,
    y=users_at_step,
    name="Users",
    text=[f"{p:.0f}%" for p in percent_at_step],
    textposition='outside',
    marker_color=bar_colors,
    hovertext=hover_text,
    hoverinfo='text'
))
fig.add_trace(go.Scatter(
    x=xtick_labels,
    y=cpa_at_step,
    name="CPA",
    yaxis="y2",
    mode="lines+markers",
    line=dict(color='gold', width=2),
    marker=dict(size=8),
    hovertemplate="<b>CPA:</b> $%{y:.2f}<extra></extra>"
))
fig.update_layout(
    title="📊 Funnel: Users and CPA by Step",
    xaxis=dict(title="Step", tickangle=-30),
    yaxis=dict(title="Users"),
    yaxis2=dict(title="CPA", overlaying="y", side="right"),
    bargap=0.4,
    height=600,
    legend=dict(x=1, y=1.15, orientation="h"),
    margin=dict(t=80, b=80),
)
st.plotly_chart(fig, use_container_width=True)


# ===== PATH ANALYSIS ПО ПЕЙВОЛУ (обновлённый) =====
st.markdown("---")
st.markdown("## 🔀 Path Analysis по Paywall-событиям")

# Новый список ключевых событий:
key_events = [
    "CompleteRegistration",
    "Paywall scroll 1",
    "Paywall scroll 50",
    "Paywall Popup open",
    "Paywall click get_my_plan",
    "Paywall Popup close",
    "initiatecheckout",
    "Paddle checkout.payment.initiated",
    "Paddle checkout.payment.failed",
    "Paddle checkout.completed",
    "Purchase",
    #"Page closed"
    # если что-то появится — просто добавь
]

user_paths = []
for (user, quiz), group in quiz_df.groupby(['user_id', 'quiz_id']):
    group = group.sort_values('event_time')
    if 'CompleteRegistration' in group['event_type'].values:
        idx = group[group['event_type'] == 'CompleteRegistration'].index[0]
        after = group.loc[idx:]
        path = [x for x in after['event_type'] if x in key_events]
        if path and path[0] == 'CompleteRegistration':
            user_paths.append(path)

import pandas as pd
paths_df = pd.DataFrame({'path': [' → '.join(p) for p in user_paths]})
path_counts = paths_df['path'].value_counts().reset_index()
path_counts.columns = ['Path', 'Users']

top_n = st.slider("Сколько путей показать?", min_value=5, max_value=30, value=10)
st.markdown(f"### 🛣️ Топ-{top_n} самых популярных путей по Paywall (только с CompleteRegistration в начале)")
st.dataframe(path_counts.head(top_n), use_container_width=True)

# ===== SANKEY DIAGRAM =====
import plotly.graph_objects as go

def get_sankey_edges(paths):
    edges = {}
    for path in paths:
        steps = path.split(' → ')
        for i in range(len(steps) - 1):
            a, b = steps[i], steps[i+1]
            key = (a, b)
            edges[key] = edges.get(key, 0) + 1
    return edges

edges = get_sankey_edges(paths_df['path'])
unique_steps = sorted(set([s for edge in edges.keys() for s in edge]))
step_idx = {s: i for i, s in enumerate(unique_steps)}
sources = [step_idx[a] for (a, b) in edges.keys()]
targets = [step_idx[b] for (a, b) in edges.keys()]
values = list(edges.values())

if len(edges) > 0:
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=18,
            line=dict(color="black", width=0.5),
            label=unique_steps,
            color="mediumpurple"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(186,85,211,0.3)"
        )
    )])
    fig_sankey.update_layout(title_text="Paywall User Journeys (Sankey)", height=550, margin=dict(t=20, b=20))
    st.plotly_chart(fig_sankey, use_container_width=True)
else:
    st.info("Недостаточно данных для построения Sankey диаграммы.")



