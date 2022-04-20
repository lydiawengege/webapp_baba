import numpy as np
import streamlit as st
from ReflectionPadding3D import ReflectionPadding3D
import tensorflow as tf
from tensorflow.keras import backend as K


@st.cache(allow_output_mutation=True)
def load_sg_model():
    vae_model = tf.keras.models.load_model('models_CNN/SG_Feb2021.h5',
                                           custom_objects={'ReflectionPadding3D':
                                                            ReflectionPadding3D})
    vae_model._make_predict_function()
    session = K.get_session()
    return vae_model, session

def predict_sg(x):
    sg_model, sg_session = load_sg_model()
    with sg_session.as_default():
        with sg_session.graph.as_default():
            sg = sg_model.predict(x)[0, :, :, :, 0]
    sg[sg < 0.01] = 0
    sg[sg > 1] = 1
    return sg

@st.cache(allow_output_mutation=True)
def load_p_model():
    p_model = tf.keras.models.load_model('models_CNN/dP_Feb2021.h5',
                                           custom_objects={'ReflectionPadding3D':
                                                            ReflectionPadding3D})
    p_model._make_predict_function()
    session = K.get_session()
    return p_model, session


def predict_p(x):
    p_model, p_session = load_p_model()
    with p_session.as_default():
        with p_session.graph.as_default():
            p = p_model.predict(x)[0, :, :, :, 0]
    return p


@st.cache(allow_output_mutation=True)
def load_xco2_model():
    xco2_model = tf.keras.models.load_model('models_CNN/bxmf_Feb2021.h5',
                                           custom_objects={'ReflectionPadding3D':
                                                            ReflectionPadding3D})
    xco2_model._make_predict_function()
    session = K.get_session()
    return xco2_model, session


def predict_xco2(x, sg, p, p_init):
    data_sg = sg[np.newaxis, :, :, :, np.newaxis]
    data_p = p + p_init[:,:,np.newaxis]
    data_p = data_p[np.newaxis, :, :, :, np.newaxis] / 600
    model_input = np.concatenate([x, data_sg, data_p], axis=-1)

    xco2_model, xco2_session = load_xco2_model()
    with xco2_session.as_default():
        with xco2_session.graph.as_default():
            xco2 = xco2_model.predict(model_input)[0, :, :, :, 0]
    return xco2*0.038


def make_input(k, perfs, inj_rate, temp, P, Swi, lam, thickness):
    norm_inj = lambda a: (a - 3e5) / (3e6 - 3e5)
    norm_temp = lambda a: (a - 30) / (180 - 30)
    norm_P = lambda a: (a - 100) / (300 - 100)
    norm_lam = lambda a: (a - 0.3) / 0.4
    norm_Swi = lambda a: (a - 0.1) / 0.2

    k_map = k.reshape((1, 96, 200))
    k_map = np.repeat(k_map, 18, axis=0)
    perf_map = np.zeros((1, 96, 200))
    for perf in perfs:
        perf_map[0, perf[0]:perf[1], 0] = 1
    inj_map = np.ones((1, 96, 200)) * norm_inj(inj_rate)
    temp_map = np.ones((1, 96, 200)) * norm_temp(temp)
    P_map = np.ones((1, 96, 200)) * norm_P(P)
    Swi_map = np.ones((1, 96, 200)) * norm_Swi(Swi)
    lam_map = np.ones((1, 96, 200)) * norm_lam(lam)

    x = np.concatenate((k_map, perf_map, inj_map, temp_map, P_map, Swi_map, lam_map), axis=0)
    x = x.transpose((1,2,0))
    return x[np.newaxis, ..., np.newaxis]