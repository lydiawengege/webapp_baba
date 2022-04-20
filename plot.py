import numpy as np
import plotly.graph_objects as go
import streamlit as st
import math

dx = np.cumsum(3.5938*np.power(1.035012, range(200))) + 0.1
dx = np.insert(dx, 0, 0.1)


def grid_volume(dx):
    vol = np.zeros((96, 200))
    for i in range(200):
        vol[:, i] = math.pi*(dx[i+1]**2 - dx[i]**2) * 2.0833333333333335
    print('np.sum(vol)', np.sum(vol))
    return vol


def draw_figure(plot, title, unit):
    y = np.arange(0, 96)
    x = np.arange(0, 200)
    fig = go.Figure(data=go.Heatmap(x=x, y=y, z=np.flipud(plot),
                                    colorscale='jet',
                                    colorbar=dict(title=unit)
                                    ))
    fig.update_layout(title=title,
                      xaxis=dict(range=[0, 200]),
                      height=300)

    return fig

@st.cache
def draw_real_figure(plot, title, unit, thickness):
    y = np.linspace(0, thickness*2.0833, num=thickness)
    fig = go.Figure(data=go.Heatmap(x=dx, y=y, z=np.flipud(plot[:thickness,:]),
                                    colorscale='jet',
                                    colorbar=dict(title=unit)
                                    ))
    fig.update_layout(title=title,
                      xaxis=dict(range=[0, int(2000*thickness/96)], title='r (m)'),
                      yaxis=dict(title='z (m)'),
                      height=300)
    return fig


def draw_buildup_figure(plot, title, unit, thickness):
    y = np.linspace(0, thickness*2.0833, num=thickness)
    fig = go.Figure(data=go.Heatmap(x=dx, y=y, z=np.flipud(plot[:thickness,:]),
                                    zmin=0,
                                    zmax=max(10, int(np.max(plot))),
                                    colorscale='jet',
                                    colorbar=dict(title=unit)
                                    ))
    fig.update_layout(title=title,
                      xaxis=dict(range=[0, 2000], title='r (m)'),
                      yaxis=dict(title='z (m)'),
                      height=300)
    return fig


def draw_pressure_profile(x, time, monitoring_well):
    fig = go.Figure()
    idx = find_nearest_idx(dx, monitoring_well)
    print(idx)
    buildup = np.mean(x[:, idx, :], axis=0) * 300
    ymax = np.max(np.mean(x[:, :, :], axis=0) * 300)
    fig.add_trace(go.Scatter(x=time, y=buildup,
                             line=dict(color='royalblue', width=4, dash='dot')))
    fig.update_layout(xaxis_title='Injection duration (day)',
                      yaxis_title='Pressure buildup (bar)',
                      height=400,
                      yaxis=dict(range=[0, ymax]))
    return fig


def draw_sg_profile(x, time, monitoring_well, monitoring_well_depth):
    fig = go.Figure()
    print(int(monitoring_well_depth/2.09))
    idx = find_nearest_idx(dx, monitoring_well)
    buildup = x[95-int(monitoring_well_depth/2.09), idx, :]
    print(buildup)
    fig.add_trace(go.Scatter(x=time, y=buildup,
                             line=dict(color='royalblue', width=4, dash='dot')))
    fig.update_layout(xaxis_title='Injection duration (day)',
                      yaxis_title='Gas saturation',
                      yaxis=dict(range=[0, 1]),
                      height=400)
    return fig


@st.cache()
def draw_bl_profile(x, time_select):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dx, y=x[0, :, time_select],
                             line=dict(color='royalblue', width=4, dash='dot')))
    fig.update_layout(xaxis_title='Distance from injection well (m)',
                      yaxis_title='Gas saturation',
                      xaxis=dict(range=[0, 2000]), yaxis=dict(range=[0, 1]),
                      height=400)
    return fig


def draw_capacity_factor(x, thickness):
    return get_capacity_factor(x, thickness)


def find_nearest_idx(dx, value):
    dx_ref = []
    for i in range(len(dx)-1):
        dx_ref.append((dx[i+1] + dx[i])/2)
    dx_ref = np.array(dx_ref)
    idx = (np.abs(dx_ref - value)).argmin()
    return idx


def grid_volume(dx):
    vol = np.zeros((96, 200))
    for i in range(200):
        vol[:, i] = math.pi*(dx[i+1]**2 - dx[i]**2) * 2.0833333333333335
    return vol


def get_capacity_factor(x, thickness):
    vol = grid_volume(dx)
    plume_vol = np.array(plume_volume(vol, x))
    total_vol = np.array(total_volume(x)) * (thickness/96)
    eff = plume_vol/total_vol
    return eff


def total_volume(x):
    vertical_sg_sum = np.sum(x[:, :, -1], axis=0)
    for j in range(200):
        if vertical_sg_sum[j] < 0.05:
            break
    return 200 * math.pi * dx[j+1] ** 2


def plume_volume(vol, x):
    return np.sum(vol * x[:, :, -1])


@st.cache()
def draw_pressure_influence(x, time_select):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dx, y=np.mean(x[:, :, time_select],axis=0),
                             line=dict(color='royalblue', width=4, dash='dot')))
    fig.update_layout(xaxis_title='Distance from injection well (m)',
                      yaxis_title='Pressure buildup (bar)',
                      xaxis=dict(type='log'),
                      height=400)
    return fig
