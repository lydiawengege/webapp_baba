# difference between CCSNet and CCSNet for Cloud:
# - using slow code for synthetic field generation (must be run on CPU or GPU)
# - removed the upload file option

import streamlit as st
from show_prediction import *
from vonk2d import *
from generateSyntheticField import *
import numpy as np
import pandas as pd
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder

time_print = ['1 days', '2 days', '4 days', '7 days', '11 days', '17 days',
              '25 days', '37 days', '53 days', '77 days', '111 days', '158 days',
              '226 days', '323 days', '1.3 years', '1.8 years', '2.6 years',
              '3.6 years', '5.2 years', '7.3 years', '10.4 years', '14.8 years',
              '21.1 years', '30.0 years']

# Sidebar
injection_duration = st.sidebar.selectbox('Maximum injection duration', time_print[1:], index=22)
st.sidebar.subheader('Choose reservoir condition')
initial_pressure = st.sidebar.slider('Initial pressure at reservoir top (bar)', min_value=100, max_value=300, value=200)
depth = lambda a : (a - 1.01325)/9.8 * 100 # defining depth function in terms of pressure "a". 9.8 = gravity, 100 = density
# gradient btwn 15 and 50 °C/km
grad_min = 15
grad_max = 50
temp_min = 14.7 + ((depth(initial_pressure) + 100) * grad_min)/1000 # 14.7 = avg global surface temp, 100 = middle of 200m depth
temp_max = 14.7 + ((depth(initial_pressure) + 100) * grad_max)/1000
reservoir_temperature = st.sidebar.slider('Reservoir temperature (C)', min_value=int(temp_min), max_value=int(temp_max), value=int(np.mean([temp_min, temp_max])))
thickness_m = st.sidebar.slider('Reservoir thickness (m)', min_value=15, max_value=200, value=200, step=1)
# Number of grid on the vertical direction, grid thickness = 2.08333
thickness = int(thickness_m / 2.08333)
st.sidebar.subheader('Choose injection design')
injection_rate = st.sidebar.slider('Injection rate (MT/yr)',
                                   min_value=0.001*thickness_m,
                                   max_value=2.,
                                   value=(0.001*thickness_m + 2)/2, step=0.1)
injection_rate_MTyr = injection_rate
injection_rate = injection_rate * 1e6 / 365 * 1000/1.862 # back stage conversion
well_top = st.sidebar.slider('Perforation top depth (m)',
                             min_value=(200-thickness_m)+12, max_value=200, value=200)
perf_thick = st.sidebar.slider('Perforation thickness (m)',
                               min_value=10,
                               max_value=thickness_m-(200-well_top),
                               value=thickness_m-(200-well_top))
well_btm = well_top - perf_thick
well_top_meters = well_top; well_btm_meters = well_btm
well_top = int((200 - well_top) / 2.08) # conversion to grid location
well_btm = int((200 - well_btm) / 2.08) # conversion to grid location
st.sidebar.subheader('Choose rock properties')
irr_water_saturation = st.sidebar.slider('Irreducible water saturation', min_value=0.1, max_value=0.3, value=0.2)
capillary_lambda = st.sidebar.slider('van Genucheten scaling factor', min_value=0.3, max_value=0.70, value=0.5)
# pack user variables
user_input = injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, capillary_lambda, \
             well_top, well_btm, injection_duration, thickness

export_user_input = injection_rate_MTyr, initial_pressure, reservoir_temperature, irr_water_saturation, capillary_lambda, \
             well_top_meters, well_btm_meters, injection_duration, thickness_m


# Main section
def what_to_predict():
    return st.selectbox('Choose what to predict', ['Gas saturation map',
                                                   'Pressure buildup map',
                                                   'Reservoir pressure map',
                                                   'Molar fraction of dissolved phase',
                                                   'Sweep efficiency factor'])


st.title('CCSNet: a deep learning modeling suite for CO2 storage')
st.set_option('deprecation.showfileUploaderEncoding', False)
option_perm = st.selectbox('Choose the type of your permeability map', ['Homogeneous',
                                                                        'Synthetic Heterogeneous',
                                                                        'Purely Layered',
                                                                        'User Upload',])

if option_perm == 'Synthetic Heterogeneous':

    nMaterials = 5
    rseed = int(st.text_input("Random seed for generating the stochastic synthetic field (integer) ", 1234))
    newMean = float(st.text_input("Mean permeability (mD)", 15))
    newStd = float(st.text_input("Standard deviation of permeability (mD)", 25))

    az = float(st.text_input("Vertical correlation (m)", 10))
    az = int(az/(200/96)) # divide by minimum grid cell width in vertical dir.
    ax = float(st.text_input("Lateral correlation (m)", 1000))
    ax = int(ax/3.5938) # divide by minimum grid cell width in lateral dir.
    # make sure this is >= 2
    med = st.selectbox('Choose a permeability map medium', ['Gaussian', 'Von Karman'])
    med_str = med
    if med == 'Gaussian':
        med = 1
    elif med == 'Von Karman':
        med = 3

    pop = st.selectbox('Choose the continuity of layers', ['Continuous', 'Discontinuous'])
    pop_str = pop
    if pop == 'Continuous':
        pop = 1
    elif pop == 'Discontinuous':
        pop = 2
        nMaterials = int(st.text_input("Number of materials (> 1)", 5))

    rng = np.random.default_rng(seed=rseed)
    # np.random.seed(rseed)
    vel = np.random.random(size=(nMaterials,))

    if pop == 2:
        # scale the data with the new mean and standard deviation
        vel = newMean + (vel - np.mean(vel)) * (newStd/np.std(vel))

    frac = np.ones((nMaterials,))

    nx = 27826
    nz = 96
    dx = 1
    dz = 1
    ix = nx*dx
    iz = nz*dz
    nu = 0.6 # only applies to von karman

    rdata = generateSyntheticField(rseed, nx, nz, dx, dz, ix, iz, ax, az, pop, med, nu, vel, frac)

    if pop == 1:
        rdata = newMean + (rdata - np.mean(rdata)) * (newStd/np.std(rdata))

    # if data is not positive, set to 0.001 (otherwise log will not work)
    valuesToReset = rdata <= 0.1
    rdata[valuesToReset] = 0.1

    st.write(draw_real_figure(rdata, ' permeability map', 'mD', thickness))
    st.markdown("""---""")

    option = what_to_predict()
    showPermeabilityMap = False

    tabData = (option_perm, med_str, newMean, newStd, pop_str, az)

    show_prediction(user_input, export_user_input, tabData, rdata, option, showPermeabilityMap)


if option_perm == 'User Upload':
    r = st.empty() # empty streamlit placeholder which we will use for the radio widget
    # mapOption = r.radio("Choose an example permeability map to use", ('Gaussian','Von Karman', 'Discontinuous'), 0, key='1') #, 'Upload a File'
    mapOption = 'Upload a File'
    if mapOption == 'Upload a File':
        uploaded_file = st.file_uploader("Choose a permeability map file", type="txt")
        if uploaded_file is not None:
            data = np.loadtxt(uploaded_file)
            st.write(draw_real_figure(data, ' permeability map', 'mD', thickness))
            st.markdown("""---""")

            option = what_to_predict()
            showPermeabilityMap = False
            tabData = (option_perm, mapOption)
            show_prediction(user_input, export_user_input, tabData, data, option, showPermeabilityMap)
    else:

        if mapOption == "Von Karman":
            data = np.loadtxt('models_CNN/permeability1.txt')
        elif mapOption == "Discontinuous":
            data = np.loadtxt('models_CNN/permeability2.txt')
        elif mapOption == "Gaussian":
            data = np.loadtxt('models_CNN/permeability3.txt')

        oldMean = np.mean(data)
        oldStd = np.std(data)
        newMean = float(st.text_input("Mean permeability (mD)", np.round(oldMean, 1)))
        newStd = float(st.text_input("Standard deviation of permeability (mD)", np.round(oldStd, 1)))
        # scale the data with the new mean and standard deviation
        data = newMean + (data - oldMean) * (newStd/oldStd)

        # if data is not positive, set to 0.001 (otherwise log will not work)
        valuesToReset = data <= 0
        data[valuesToReset] = 0.001

        # unpack user variables
        injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, \
        capillary_lambda, well_top, well_btm, injection_duration, thickness = user_input
        k_map = np.log(data) / 15 # permeability map normalization
        k_map[thickness:] = -1.07453971 # use very low perm value to mask formation thickness

        showPermeabilityMap = False
        st.write(draw_real_figure(np.exp(k_map*15), (mapOption + ' permeability map'), 'mD', thickness))

        st.markdown("""---""")
        option = what_to_predict()
        tabData = (option_perm, mapOption, newMean, newStd)

        show_prediction(user_input, export_user_input, tabData, data, option, showPermeabilityMap)

if option_perm == 'Homogeneous':
    k_homo = float(st.number_input('Input permeability (mD): ', min_value=3, max_value=1000, value=100))
    data = np.ones((96, 200)) * k_homo
    st.write(draw_real_figure(data, ' permeability map', 'mD', thickness))
    st.markdown("""---""")
    option = what_to_predict()
    showPermeabilityMap = False

    tabData = (option_perm, k_homo)

    show_prediction(user_input, export_user_input, tabData, data, option, showPermeabilityMap)

if option_perm == 'Purely Layered':

    st.write(' ')
    st.write("**Guidelines for inputting a purely layered permeability map:**")
    st.write("- Thickness of each layer must be between _2_ and _200_ m \n"
             "- Total thickness of all layers must equal _200_ m \n"
             "- Permeability of each layer must be between _0.1_ and _2000_ mD \n"
             "- Average permeability of the field must be between _5_ and _1000_ mD \n")

    numLayers = int(st.slider("Number of Layers", min_value=2, max_value=20, value=6))

    input_dataframe = pd.DataFrame(
        '',
        index=range(numLayers),
        columns=[
            'Thickness (m): ', #[2 - 200]',
            'Permeability (mD): ', #[1e-1 - 2000]'
        ]
    )

    response = AgGrid(
        input_dataframe,
        editable=True,
        sortable=False,
        filter=False,
        resizable=True,
        fit_columns_on_grid_load=True,
        height='196px',
        # autoHeight=True,
        wrapText=True)#,
        #key='input_frame') # if there is a key, then the dataframe will not refresh on streamlit

    # st.header("Permeability map")

    #if 'data' in response:
    df = response['data']
    thick_inputs = df['Thickness (m): '].to_numpy(copy=True)
    perm_inputs = df['Permeability (mD): '].to_numpy(copy=True)
    # print(perm_inputs)
    perm_inputs[perm_inputs == ""] = np.nan # set the blank values to NaN so we can convert to float (blank str cannot be float)
    thick_inputs[thick_inputs == ""] = np.nan
    perm_inputs = np.array(perm_inputs, dtype=float)
    thick_inputs = np.array(thick_inputs, dtype=float)

    isErr = False
    isWarning = any(np.isnan(thick_inputs)) or any(np.isnan(perm_inputs))

    if any(perm_inputs < 0.1):
        st.write("ERROR: You entered a permeability value less than 0.1 mD.")
        isErr = True

    if any(perm_inputs > 2000):
        st.write("ERROR: You entered a permeability value greater than 2000 mD.")
        isErr = True

    if np.sum(perm_inputs*thick_inputs)/200 < 5:
        st.write("ERROR: Your average permeability is less than 2 mD.")
        isErr = True

    if np.sum(perm_inputs*thick_inputs)/200 > 1000:
        st.write("ERROR: Your average permeability is greater than 1000 mD.")
        isErr = True

    if np.sum(thick_inputs) != 200:
        st.write("ERROR: The total thickness of your layers is not 200 m.")
        isErr = True

    if any(thick_inputs > 200):
        st.write("ERROR: You entered a thickness value greater than 200 m.")
        isErr = True

    if any(thick_inputs < 2):
        st.write("ERROR: You entered a thickness value less than 2 m.")
        isErr = True

    if isWarning:
        st.write("WARNING: You have not provided a complete permeability map yet.")

    # don't need to show the output dataframe for user
    #if not isErr:
    #    st.dataframe(df)

    if not isErr and not isWarning:
        grid_layers = np.zeros((96,1))
        thick_inputs = np.rint(thick_inputs * 96/200)
        if np.sum(thick_inputs) > 96:
            thick_inputs[np.argmax(thick_inputs)] -= np.sum(thick_inputs) - 96
        elif np.sum(thick_inputs) < 96:
            thick_inputs[np.argmax(thick_inputs)] += 96 - np.sum(thick_inputs)
        counter = 0 #-1?
        for n in range(len(thick_inputs)):
            num_grid_layers = int(thick_inputs[n])
            for i in range(num_grid_layers):
                grid_layers[counter + i, 0] = perm_inputs[n]
            counter = counter + num_grid_layers

        data = np.repeat(grid_layers, 200, axis = 1)
        showPermeabilityMap = False
        mapType = "Purely Layered"
        mapData = "See \"Layer Data\" Tab"
        tabData = (mapType, mapData, df)

        st.write(draw_real_figure(data, ' permeability map', 'mD', thickness))
        st.markdown("""---""")

        option = what_to_predict()
        show_prediction(user_input, export_user_input, tabData, data, option, showPermeabilityMap)
