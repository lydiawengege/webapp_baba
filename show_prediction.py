import streamlit as st
from plot import *
from download import *
from network import *
from density_model import *

# Sidebar section
time_print = ['1 days', '2 days', '4 days', '7 days', '11 days', '17 days',
              '25 days', '37 days', '53 days', '77 days', '111 days', '158 days',
              '226 days', '323 days', '1.3 years', '1.8 years', '2.6 years',
              '3.6 years', '5.2 years', '7.3 years', '10.4 years', '14.8 years',
              '21.1 years', '30.0 years']
time = np.cumsum(np.power(1.421245, range(24)))

def show_prediction(user_input, export_user_input, tabData, data, option, showPermeabilityMap):
    # unpack user variables
    injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, \
    capillary_lambda, well_top, well_btm, injection_duration, thickness = user_input
    if data is not None:
        k_map = np.log(data) / 15 # permeability map normalization
        k_map[thickness:] = -1.07453971 # use very low perm value to mask formation thickness
        model_input = make_input(k_map,
                       [[well_top, well_btm]],
                       injection_rate,
                       reservoir_temperature,
                       initial_pressure,
                       irr_water_saturation,
                       capillary_lambda,
                       thickness)
        # gas saturation prediction
        sg = predict_sg(model_input)
        sg[sg < 10**(-3)] = 0
        # pressure buildup prediction
        pressure = predict_p(model_input)
        pressure[pressure < 10**(-3)] = 0
        # initial pressure in the formation
        p_init = get_p_init(reservoir_temperature, initial_pressure)
        p_init[p_init < 10**(-3)] = 0
        # molar fraction of CO2 in the liquid phase
        xco2 = predict_xco2(model_input, sg, pressure, p_init)
        xco2[xco2 < 10**(-3)] = 0
        # used for the slider, initialize the time_index to be the last index
        time_index = time_print.index(injection_duration)

        # create the permeability map
        if showPermeabilityMap and option != 'Sweep efficiency factor':
            st.write(draw_real_figure(np.exp(k_map*15), 'permeability map', 'mD', thickness))

        if option == 'Gas saturation map':
            #st.write(draw_real_figure(np.exp(k_map*15), 'permeability map', 'mD', thickness))
            time_select = st.select_slider('Choose a time snapshot to view',
                                           options=time_print[:time_index+1],
                                           value=time_print[time_index])
            time_select = time_print.index(time_select)
            st.write(draw_real_figure(sg[:, :, time_select],
                                      f'CO2 gas saturation, ' f'{time_print[time_select]}',
                                      'SG',
                                      thickness))
            # st.markdown(download_file(sg[:, :, time_select]), unsafe_allow_html=True)

        if option == 'Pressure buildup map':
            #st.write(draw_real_figure(np.exp(k_map*15), 'permeability map', 'mD', thickness))
            time_select = st.select_slider('Choose a time snapshot to view',
                                           options=time_print[:time_index+1],
                                           value=time_print[time_index])
            time_select = time_print.index(time_select)
            st.write(draw_buildup_figure(pressure[:, :, time_select] * 300,
                                         f'Pressure buildup, '  f'{time_print[time_select]}',
                                         'bar',
                                         thickness))
            # download =pressure[:, :, time_select]* 300
            # st.markdown(download_file(download), unsafe_allow_html=True)

        if option == 'Reservoir pressure map':
            #st.write(draw_real_figure(np.exp(k_map*15),
            #                          'permeability map',
            #                          'mD',
            #                          thickness))
            time_select = st.select_slider('Choose a time snapshot to view',
                                           options=time_print[:time_index+1],
                                           value=time_print[time_index])
            time_select = time_print.index(time_select)
            st.write(draw_real_figure(pressure[:, :, time_select] * 300 + p_init,
                                      f'Reservoir pressure, ' f'{time_print[time_select]}',
                                      'bar',
                                      thickness))
            # download = pressure[:, :, time_select] * 300 + p_init
            # st.markdown(download_file(download), unsafe_allow_html=True)

        if option == 'Molar fraction of dissolved phase':
            #st.write(draw_real_figure(np.exp(k_map * 15),
            #                          'permeability map',
            #                          'mD',
            #                          thickness))
            time_select = st.select_slider('Choose a time snapshot to view',
                                           options=time_print[:time_index+1],
                                           value=time_print[time_index])
            time_select = time_print.index(time_select)
            st.write(draw_real_figure(xco2[:, :, time_select],
                                      f'Molar fraction of dissolved phase, '
                                      f'{time_print[time_select]}',
                                      '-',
                                      thickness))
            # st.markdown(download_file(xco2[:, :, time_select]), unsafe_allow_html=True)

        if option == 'Sweep efficiency factor':
            time_select = st.select_slider('Choose a time snapshot to view',
                                           options=time_print[:time_index+1],
                                           value=time_print[time_index])
            time_select = time_print.index(time_select)
            st.write(draw_real_figure(sg[:, :, time_select],
                                      f'CO2 gas saturation, ' f'{time_print[time_select]}',
                                      'SG',
                                      thickness))
            storage_eff = round(draw_capacity_factor(sg[:, :, :time_select], thickness), 3)
            st.write('')
            st.write('At', time_print[time_select], ' :')
            st.write(r'''
                    $$E_{sweep}=\frac{V_{gas}}{V_{r_{footprint}}}=\frac{\sum_nV_n\phi_nS_{n}}{\sum_{n\in footprint}V_{n}\phi_{n}}=$$
                    ''', storage_eff)

        # Save session

        injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, \
        capillary_lambda, well_top, well_btm, injection_duration, thickness = export_user_input

        export_user_input = injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, \
        capillary_lambda, well_top, well_btm, injection_duration, thickness, time_print[time_index]

        exportData = (export_user_input, tabData, data, sg, pressure*300, p_init, xco2, time_index)
        #st.markdown(save_session(exportData), unsafe_allow_html=True)
        st.markdown(get_table_download_link(exportData), unsafe_allow_html=True)

        # These options were used for Sally's class, just leave these commented

        # if option == 'Pressure buildup influence':
        #     time_select = st.select_slider('Choose a time snapshot to view',
        #                                    options=time_print[:time_index+1],
        #                                    value=time_print[time_index])
        #     time_select = time_print.index(time_select)
        #     st.write(draw_pressure_influence(pressure * 300, time_select))

        # if option == 'Gas saturation monitor':
        #     monitoring_well = float(
        #         st.slider('Distance from the injection well to the gas saturation monitor (m)',
        #                   value=500,
        #                   max_value=10000,
        #                   min_value=0,
        #                   step=10))
        #     monitoring_well_depth = float(
        #         st.slider('Distance from the reservoir bottom to the gas saturation monitor (m)',
        #                   min_value=0,
        #                   max_value=200,
        #                   value=200,
        #                   step=1))
        #     st.write(draw_sg_profile(sg, time, monitoring_well, monitoring_well_depth))

        # if option == 'Top layer gas saturation profile (plume front)':
        #     time_select = st.select_slider('Choose a time snapshot to view',
        #                                    options=time_print[:time_index+1],
        #                                    value=time_print[time_index])
        #     time_select = time_print.index(time_select)
        #     st.write(draw_bl_profile(sg, time_select))

        # if option == 'Monitoring well pressure buildup profile':
        #     monitoring_well = float(st.number_input('Location of the monitoring well (m)',
        #                                             value=300,
        #                                             max_value=100000,
        #                                             min_value=0))
        #     st.write(draw_pressure_profile(pressure, time, monitoring_well))

        # if option == 'Solubility trapping coefficient':
        #     time_select = st.select_slider('Choose a time snapshot to view',
        #                                    options=time_print[:time_index+1],
        #                                    value=time_print[time_index])
        #     time_select = time_print.index(time_select)
        #     st.write(draw_real_figure(sg[:, :, time_select],
        #                               f'CO2 gas saturation',
        #                               'SG',
        #                               thickness))
        #     storage_eff = 0.102
        #     st.write('')
        #     st.write('At', time_print[time_select], ' :')
        #     st.write(r'''
        #             $$C_{diss}=$$
        #             ''', storage_eff)


def get_p_init(reservoir_temperature, initial_pressure):
    rho_w = predict_liquid_density(reservoir_temperature, initial_pressure + 19.5, 0)
    p_same_rho = []
    p = initial_pressure
    for i in range(96):
        p_bottom = p + rho_w * 2.0833 * 9.8 / 100000
        p_same_rho.append((p_bottom + p) / 2)
        p = p_bottom
    p_init = np.array(p_same_rho)
    p_init = np.repeat(p_init[:, np.newaxis], 200, axis=1)
    return p_init
