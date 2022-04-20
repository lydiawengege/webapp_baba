import base64
import pandas as pd
from io import BytesIO

def download_file(data):
    # When no file name is given, pandas returns the CSV as a string, nice.
    csv = pd.DataFrame(data).to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    link = f'<a href="data:file/csv;base64,{b64}">Download Output File</a> (right-click and save as .csv)'
    return link

def pd_to_excel(data):
    output = BytesIO()

    user_input, tabData, permeability, sg, pressure, p_init, xco2, time_index = data

    injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, \
    capillary_lambda, well_top, well_btm, injection_duration, thickness, time_index_input = user_input

    if tabData[0] == "Homogeneous":
        (option_perm, inputPerm) = tabData
        user_input = option_perm, inputPerm, injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, \
        capillary_lambda, well_top, well_btm, injection_duration, thickness, time_index_input
        df0 = pd.DataFrame(user_input)
        df0 = df0.T # transpose
        df0.columns = ["Type of Permeability Map", "Input Permeability", "Injection Rate (MT/yr)", "Initial Pressure (bar)", \
            "Reservoir Temperature (°C)", "Irreducible Water Saturation", "Van Genucheten Scaling Factor", "Perforation Top Depth (m)", \
            "Perforation Bottom Depth (m)", "Injection Duration", "Perforation Thickness (m)", "Time Index"]

    elif tabData[0] == "Heterogeneous":
        if tabData[1] == "Upload a File":
            (option_perm, mapOption) = tabData
            user_input = option_perm, mapOption, injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, \
            capillary_lambda, well_top, well_btm, injection_duration, thickness, time_index_input
            df0 = pd.DataFrame(user_input)
            df0 = df0.T # transpose
            df0.columns = ["Type of Permeability Map", "Permeability Map Option", "Injection Rate (MT/yr)", "Initial Pressure (bar)", \
                "Reservoir Temperature (°C)", "Irreducible Water Saturation", "Van Genucheten Scaling Factor", "Perforation Top Depth (m)", \
                "Perforation Bottom Depth (m)", "Injection Duration", "Perforation Thickness (m)", "Time Index"]

        else:
            (option_perm, mapOption, newMean, newStd) = tabData
            user_input = option_perm, mapOption, newMean, newStd, injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, \
            capillary_lambda, well_top, well_btm, injection_duration, thickness, time_index_input
            df0 = pd.DataFrame(user_input)
            df0 = df0.T # transpose
            df0.columns = ["Type of Permeability Map", "Permeability Map Option", "Permeability Mean", "Permeability Standard Deviation", \
                "Injection Rate (MT/yr)", "Initial Pressure (bar)", "Reservoir Temperature (°C)", "Irreducible Water Saturation", \
                "Van Genucheten Scaling Factor", "Perforation Top Depth (m)", "Perforation Bottom Depth (m)", "Injection Duration", \
                "Perforation Thickness (m)", "Time Index"]

    elif tabData[0] == "Synthetic Field":
        (option_perm, med, newMean, newStd, pop, az) = tabData
        user_input = option_perm, med, newMean, newStd, pop, az, injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, \
        capillary_lambda, well_top, well_btm, injection_duration, thickness, time_index_input
        df0 = pd.DataFrame(user_input)
        df0 = df0.T # transpose
        df0.columns = ["Type of Permeability Map", "Permeability Map Option", "Permeability Mean", "Permeability Standard Deviation", \
            "Layer Continuity", "Vertical Correlation", "Injection Rate (MT/yr)", "Initial Pressure (bar)", "Reservoir Temperature (°C)", "Irreducible Water Saturation", \
            "Van Genucheten Scaling Factor", "Perforation Top Depth (m)", "Perforation Bottom Depth (m)", "Injection Duration", \
            "Perforation Thickness (m)", "Time Index"]


    elif tabData[0] == "Purely Layered":
        (maptype, mapdata, _) = tabData
        user_input = maptype, mapdata, injection_rate, initial_pressure, reservoir_temperature, irr_water_saturation, \
        capillary_lambda, well_top, well_btm, injection_duration, thickness, time_index_input
        df0 = pd.DataFrame(user_input)
        df0 = df0.T # transpose
        df0.columns = ["Type of Permeability Map", "Permeability Map Data", "Injection Rate (MT/yr)", "Initial Pressure (bar)", "Reservoir Temperature (°C)", "Irreducible Water Saturation", \
            "Van Genucheten Scaling Factor", "Perforation Top Depth (m)", "Perforation Bottom Depth (m)", "Injection Duration", \
            "Perforation Thickness (m)", "Time Index"]


    df1 = pd.DataFrame(permeability)
    df2 = pd.DataFrame(sg[:, :, time_index])
    df3 = pd.DataFrame(pressure[:, :, time_index])
    df4 = pd.DataFrame(pressure[:, :, time_index] + p_init)
    df5 = pd.DataFrame(xco2[:, :, time_index])
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df0.to_excel(writer, sheet_name='Metadata')
    df1.to_excel(writer, sheet_name='Permeability Map')
    df2.to_excel(writer, sheet_name='Gas Saturation Map')
    df3.to_excel(writer, sheet_name='Pressure Buildup Map')
    df4.to_excel(writer, sheet_name='Reservoir Pressure Map')
    df5.to_excel(writer, sheet_name='Molar Frac. of Dissolved Phase')
    if tabData[0] == "Purely Layered":
        (_, _, df6) = tabData
        df6 = df6.apply(pd.to_numeric) # convert the df of strs to numerics
        df6.to_excel(writer, sheet_name='Layer Data')

    #writer = pd.ExcelWriter(output, engine='xlsxwriter')
    #df.to_excel(writer, sheet_name=sheetname)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(data):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = pd_to_excel(data)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="CCSNet_py.xlsx">Download all data as XLSX</a>' # decode b'abc' => abc

'''
def save_session(data):

    user_input, sg, pressure, p_init, xco2, time_index = data

    # When no file name is given, pandas returns the CSV as a string, nice.
    csv1 = pd.DataFrame(user_input).to_csv(r'csv1', index=False)
    #b64 = base64.b64encode(csv1.encode()).decode()  # some strings <-> bytes conversions necessary here
    #link = f'<a href="data:file/csv;base64,{b64}">Save Session</a> (right-click and save as .csv)'

    csv2 = pd.DataFrame(sg[:, :, time_index]).to_csv(r'csv2', index=False)
    #b64 = base64.b64encode(csv2.encode()).decode()  # some strings <-> bytes conversions necessary here

    with zipfile.ZipFile('myFolder.zip','w') as zip:
        # writing each file one by one
        zip.write("csv1")
        zip.write("csv2")

    link = f'<a href="myFolder.zip">Save Session</a> (right-click and save)'
    return link
'''
