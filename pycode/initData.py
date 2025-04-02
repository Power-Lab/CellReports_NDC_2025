import math
import os
import pickle
import sys
import shutil
import json
import shapely

import numpy as np
import pandas as pd
import scipy.io as scio
import geopandas as gpd

from obtainPrice import cfEleP
from callUtility import dirFlag, geo_distance, getResDir, getWorkDir, makeDir, str2int, getCRF, extractProvinceName

dir_flag = dirFlag()
work_dir = getWorkDir()


def winterHour():
    """
    Return the list of hours that are winter
    """
    winter_hour = []

    for i in range(24 * 74):
        winter_hour.append(i)

    for i in range(24 * (365 - 46), 24 * 365):
        winter_hour.append(i)

    f_wh = open(work_dir + 'data_csv' + dir_flag + 'simulation_meta/winter_hour.csv', 'w+')

    for i in winter_hour:
        f_wh.write('%s\n' % i)
    f_wh.close()


def seedHour(vre_year: str, years: int, step: int, days: int, res_tag: str, curr_year: int):
    """
    Parameters
    ----------
    vre_year : str
        VRE years
    years : int
        How many years to simulate
    step: int
        The distance between each day, in days
    days: int
        How many days to simulate within a year
    res_tag: str
        Result folder tag
    curr_year: int
        Current year
    """
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", res_tag + "_" + vre_year, str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")

    start_point = []
    hour_profile = []
    hour_pre = {}

    for i in range(days):
        start_point.append(i * step * 24)

    if years == 0:
        for i in start_point:
            for k in range(24):
                if i + k <= 8759:
                    hour_profile.append(i + k)
    else:
        for i in range(8760 * years):
            hour_profile.append(i)

    for i in range(1, len(hour_profile)):
        hour_pre[hour_profile[i]] = hour_profile[i - 1]

    # Write results into result folder
    hour_profile_df = pd.DataFrame()
    hour_profile_df["hours_index"] = hour_profile
    hour_profile_df.to_csv(os.path.join(out_input_path, 'hour_seed.csv'), index=False, header=False)

    hour_pre_df = pd.DataFrame()
    hour_pre_df["hours"] = hour_profile[1:]
    hour_pre_df["hours_index"] = hour_profile[:-1]
    hour_pre_df.to_csv(os.path.join(out_input_path, 'hour_pre.csv'), index=False, header=False)

    with open(os.path.join(out_input_path, 'hour_pre.pkl'), 'wb') as fout:
        pickle.dump(hour_pre, fout)
    fout.close()


def seedHbh(is8760, step, hours):
    dir_flag = dirFlag()
    work_dir = getWorkDir()

    hour_profile = []
    hour_pre = {}
    f_hour_seed = open(work_dir + 'data_csv' + dir_flag + 'simulation_meta/hour_seed.csv', 'w+')

    if is8760 == 0:
        for i in range(24):
            hour_profile.append(i)
        for i in range(7, hours):
            hour_profile.append(step * i - 1)
    else:
        for i in range(8760):
            hour_profile.append(i)

    for i in range(1, len(hour_profile)):
        hour_pre[hour_profile[i]] = hour_profile[i - 1]

    # print(hour_pre)
    for i in hour_profile:
        f_hour_seed.write('%s\n' % i)
    f_hour_seed.close()

    with open(work_dir + 'data_pkl' + dir_flag + 'hour_pre.pkl', 'wb') as fout:
        pickle.dump(hour_pre, fout)
    fout.close()


def SpurTrunkDis(vre, year, isSubLcInterProvince):
    if vre == 'wind':
        vre_dir = work_dir
    elif vre == 'solar':
        vre_dir = work_dir

    county_lc_dis = {}

    load_center = {}
    county = {}

    wind_cell_file = {'2016': ['China_windpower_offshore_provin_2016.csv', 'China_windpower_onshore_provin_2016.csv'],
                      '2015': ['China_windpower_offshore_provin_2015.csv', 'China_windpower_onshore_provin_2015.csv'],
                      '2024': ['China_windpower_offshore_provin_2024.csv', 'China_windpower_onshore_provin_2024.csv']}

    solar_cell_file = {'2015': ['China_solarpower_coordinate_2015.csv'],
                       '2016': ['China_solarpower_coordinate_2016.csv'],
                       '2024': ['China_solarpower_coordinate_2024.csv']}

    cell_file = {'wind': wind_cell_file, 'solar': solar_cell_file}

    f_lc = open(work_dir + 'data_csv' + dir_flag + 'geography/city_pos.csv', 'r+')
    next(f_lc)

    for line in f_lc:
        line = line.replace('\n', '')
        line = line.split(',')
        if line[2] not in load_center:
            load_center[line[2]] = []
        load_center[line[2]].append((eval(line[0]), eval(line[1])))

    f_lc.close()

    f_county = open(work_dir + 'data_csv' + dir_flag + 'geography/county_geo.csv', 'r+')
    next(f_county)

    for line in f_county:
        line = line.replace('\n', '')
        line = line.split(',')

        if line[2] not in county:
            county[line[2]] = []
        if line[2] == '':
            print(line)
        county[line[2]].append((eval(line[1]), eval(line[0])))

    f_county.close()

    county_lc_pair = {}

    for p in county:
        for c in county[p]:
            min_county_lc_dis = sys.maxsize
            if isSubLcInterProvince == 0:
                for lc in load_center[p]:
                    dis = geo_distance(c[1], c[0], lc[1], lc[0])
                    if dis < min_county_lc_dis:
                        min_county_lc_dis = dis
                        county_lc_dis[(c[1], c[0])] = dis
                        county_lc_pair[(c[1], c[0])] = (lc[1], lc[0])
            elif isSubLcInterProvince == 1:
                for p1 in load_center:
                    for lc in load_center[p1]:
                        dis = geo_distance(c[1], c[0], lc[1], lc[0])
                        if dis < min_county_lc_dis:
                            min_county_lc_dis = dis
                            county_lc_dis[(c[1]), c[0]] = dis
                            county_lc_pair[(c[1], c[0])] = (lc[1], lc[0])

    with open(work_dir + 'data_pkl' + dir_flag + 'county_lc_pair_' + vre + '_' + year + '.pkl', 'wb+') as fout:
        pickle.dump(county_lc_pair, fout)
    fout.close()

    for file in cell_file[vre][year]:
        fr_cell = open(vre_dir + 'data_csv' + dir_flag + "vre_potentials" + dir_flag + file, 'r+')
        next(fr_cell)
        fw_cell = open(work_dir + 'data_csv' + dir_flag + "vre_installations" + dir_flag + 'inter_connect_' + file, 'w+')
        for cell in fr_cell:
            cell = cell.replace('\n', '')
            tmp_cell = cell
            tmp_cell = tmp_cell.split(',')
            if vre == 'wind':
                cell_lat = eval(tmp_cell[3])
                cell_lon = eval(tmp_cell[4])
                province = tmp_cell[8]
            if vre == 'solar':
                cell_lat = eval(tmp_cell[1])
                cell_lon = eval(tmp_cell[2])
                province = tmp_cell[3]

            min_cell_sub_dis = sys.maxsize
            for i in range(len(county[province])):
                cell_sub_dis = geo_distance(cell_lat, cell_lon, county[province][i][1], county[province][i][0])
                if cell_sub_dis < min_cell_sub_dis:
                    min_cell_sub_dis = cell_sub_dis
                    s_lc_dis = county_lc_dis[(county[province][i][1], county[province][i][0])]
                    sub_station_lat = county[province][i][1]
                    sub_station_lon = county[province][i][0]
            fw_cell.write('%s,%s,%s,%s,%s\n' % (cell, sub_station_lat, sub_station_lon, min_cell_sub_dis, s_lc_dis))
        fr_cell.close()
        fw_cell.close()


def initCellData(vre: str, vre_year_single: str, hour_seed: list, res_tag: str, vre_year: str,
                 curr_year: int, last_year: int, scen_params: dict):
    """
    Arguments
    ---------
    vre: str
        VRE name (wind or solar)
    vre_year_single: str
        This should be either solar year or wind year, rather than the combined string "w2015_s2015"
    hour_seed: list
        Optimization hours
    vre_year : str
        VRE years
    res_tag: str
        Result folder tag
    curr_year: int
        The current modeled year
    last_year: int
        The last modeled year
    scen_params: dict
        Scenario parameters for the current modeled year
    """
    # Specify file paths for the current year
    out_path = os.path.join(work_dir, "data_res", res_tag + "_" + vre_year, str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    # Specify file paths for the last year
    out_path_last_year = os.path.join(work_dir, "data_res", res_tag + "_" + vre_year, str(last_year))
    out_input_path_last_year = os.path.join(out_path_last_year, "inputs")
    out_output_path_last_year = os.path.join(out_path_last_year, "outputs")
    out_output_processed_path_last_year = os.path.join(out_path_last_year, "outputs_processed")

    # Read parameters from scenario parameters json file
    weighted_average_cost_of_capital = scen_params["finance"]["weighted_average_cost_of_capital"]  # percentage
    lifespan_transmission_spur = scen_params["lifespan"]["transmission_spur"]  # yrs
    lifespan_transmission_trunk = scen_params["lifespan"]["transmission_trunk"]  # yrs
    loss_rate = scen_params["trans"]["trans_loss"]  # per km
    capex_spur_fixed = scen_params["trans"]["capex_spur_fixed"]  # RMB/kW
    capex_spur_var = scen_params["trans"]["capex_spur_var"]  # RMB/kW-km
    capex_trunk_fixed = scen_params["trans"]["capex_trunk_fixed"]  # RMB/kW
    capex_trunk_var = scen_params["trans"]["capex_trunk_var"]  # RMB/kW-km
    if vre == "solar":
        equip = [scen_params['vre']['capex_equip_pv'], scen_params['vre']['capex_equip_dpv']]
        other = [scen_params['vre']['capex_other_pv'], scen_params['vre']['capex_other_dpv']]
        om = [scen_params['vre']['capex_om_pv'], scen_params['vre']['capex_om_dpv']]
        cap_scale = scen_params["vre"]["cap_scale_pv"]
        cap_scale_east = scen_params["vre"]["cap_scale_pv_ep"]
    else:  # vre = "wind"
        equip = [scen_params['vre']['capex_equip_on_wind'], scen_params['vre']['capex_equip_off_wind']]
        other = [scen_params['vre']['capex_other_on_wind'], scen_params['vre']['capex_other_off_wind']]
        om = [scen_params['vre']['capex_om_on_wind'], scen_params['vre']['capex_om_off_wind']]
        cap_scale = scen_params["vre"]["cap_scale_wind"]
        cap_scale_east = scen_params["vre"]["cap_scale_wind_ep"]

    # Specify the params for calculating cell LCOEs
    r = 0.062
    N = 25
    iN = 15
    dN = 15

    # Pre-process the parameters
    Hour = len(hour_seed)
    spur_capex = capex_spur_var * getCRF(weighted_average_cost_of_capital, lifespan_transmission_spur)
    spur_capex_fixed = capex_spur_fixed * getCRF(weighted_average_cost_of_capital, lifespan_transmission_spur)
    trunk_capex = capex_trunk_var * getCRF(weighted_average_cost_of_capital, lifespan_transmission_trunk)
    trunk_capex_fixed = capex_trunk_fixed * getCRF(weighted_average_cost_of_capital, lifespan_transmission_trunk)

    # Read the province list of China
    provin_abbrev = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = []
    next(provin_abbrev)
    for pro in provin_abbrev:
        pro = pro.replace('\n', '')
        pro = pro.split(',')
        provins.append(pro[1])
    provin_abbrev.close()

    # Specify integrated wind/solar generation potential file names
    integrated_file = {'wind': 'integrated_wind.csv',
                       'solar': 'integrated_solar.csv'}

    # Copy the integrated files from the default folder OR last year's folder
    if not os.path.exists(os.path.join(out_input_path, integrated_file[vre])):
        if not os.path.exists(os.path.join(out_output_processed_path_last_year, f"integrated_{vre}_{last_year}.csv")):
            print("Copy integrated vre files from the default folder")
            shutil.copy2(work_dir + 'data_csv' + dir_flag + "vre_installations" + dir_flag + integrated_file[vre],
                         os.path.join(out_input_path, integrated_file[vre]))
        else:
            print("Copy integrated vre files from last year's output folder")
            shutil.copy2(os.path.join(out_output_processed_path_last_year, f"integrated_{vre}_{last_year}.csv"),
                         os.path.join(out_input_path, integrated_file[vre]))

    # Read the integrated files
    integrated = {}
    f_inted = open(os.path.join(out_input_path, integrated_file[vre]), 'r+', encoding='utf-8')
    next(f_inted)
    for line in f_inted:
        line = line.replace('\n', '')
        line = eval(line)
        integrated[(line[0], line[1])] = line[2]
    f_inted.close()

    # Initialize the dictionaries
    provin_cell = {}
    provin_cell_lon = {}
    provin_cell_lat = {}
    provin_cf_sort = {}
    cf_prof = {}
    provin_cell_genC = {}  # Generation cost
    c_s_dis = {}
    s_lc_dis = {}
    sub_lat = {}
    sub_lon = {}
    cf_elep = {'on': {}, 'off': {}}

    # Calculate the electricity price at varying capacity factor for solar and wind
    if vre == 'solar':
        for i in np.arange(0.0020, 0.8000, 0.0001):
            i = round(i, 4)
            cf_elep['on'][i] = np.round(cfEleP(cf=i, equipC=equip[0], otherC=other[0], OMC=om[0],
                                               r=r, N=N, iN=iN, dN=dN), 3)
        for i in np.arange(0.0020, 0.8000, 0.0001):
            i = round(i, 4)
            cf_elep['off'][i] = np.round(cfEleP(cf=i, equipC=equip[1], otherC=other[1], OMC=om[1],
                                                r=r, N=N, iN=iN, dN=dN), 3)
    elif vre == 'wind':
        for i in np.arange(0.0020, 1.0000, 0.0001):
            i = round(i, 4)
            cf_elep['on'][i] = np.round(cfEleP(cf=i, equipC=equip[0], otherC=other[0], OMC=om[0],
                                               r=r, N=N, iN=iN, dN=dN), 3)
        for i in np.arange(0.0020, 0.8000, 0.0001):
            i = round(i, 4)
            cf_elep['off'][i] = np.round(cfEleP(cf=i, equipC=equip[1], otherC=other[1], OMC=om[1],
                                                r=r, N=N, iN=iN, dN=dN), 3)

    for pro in provins:
        provin_cf_sort[pro] = []
        cf_prof[pro] = []
        provin_cell_genC[pro] = []

    # Merge all the cell costs, potential, CF data
    if vre == 'wind':
        for tech in ["offshore", "onshore"]:
            if (tech != "offshore") or (vre_year_single != "2024"):
                if tech == "onshore":
                    vre_year_single = 2015  # Use 2015 mat files for all the weather years
                for pro in provins:
                    provin_cell[pro] = {}
                    provin_cell_lon[pro] = {}
                    provin_cell_lat[pro] = {}
                    c_s_dis[pro] = {}
                    s_lc_dis[pro] = {}
                    sub_lat[pro] = {}
                    sub_lon[pro] = {}

                # Read VRE installation results
                cell_file_r = open(work_dir + 'data_csv' + dir_flag + "vre_installations" + dir_flag +
                                   f"inter_connect_China_windpower_{tech}_provin_{vre_year_single}.csv")
                for cell in cell_file_r:
                    cell = cell.replace('\n', '')
                    cell = cell.split(',')

                    cell[1] = str(int(eval(cell[1])))  # cell id X in mat files
                    cell[2] = str(int(eval(cell[2])))  # cell id Y in mat files

                    provin_cell_lat[cell[8]][(cell[1], cell[2])] = eval(cell[3])  # cell lat
                    provin_cell_lon[cell[8]][(cell[1], cell[2])] = eval(cell[4])  # cell lon
                    provin_cell[cell[8]][(cell[1], cell[2])] = eval(
                        cell[7])  # cell potential in MW or GW for 2024 offshore

                    sub_lat[cell[8]][(cell[1], cell[2])] = eval(cell[9])  # lat of the nearest substation
                    sub_lon[cell[8]][(cell[1], cell[2])] = eval(cell[10])  # lon of the nearest substation
                    c_s_dis[cell[8]][(cell[1], cell[2])] = eval(cell[11])  # province/cell-substation distance
                    s_lc_dis[cell[8]][(cell[1], cell[2])] = eval(cell[12])  # substation-load center distance
                cell_file_r.close()

                # Merge cell costs, CF, potentials
                for root, dirs, files in os.walk(work_dir + 'data_mat' + dir_flag + f"{tech}{vre_year_single}"):
                    for mat in files:
                        cell_flag = mat.replace('.mat', '').split('_')
                        for pro in provins:
                            if (cell_flag[1], cell_flag[2]) in provin_cell[pro]:
                                pro_flag = pro

                                # Read hourly CF at each cell for the whole year from mat files
                                cell_cf_prof = scio.loadmat(
                                    work_dir + 'data_mat' + dir_flag + f"{tech}{vre_year_single}" + dir_flag + mat)['X_cf']
                                cell_cf_prof = np.round(cell_cf_prof[:8760], 3)

                                # Compute average CF for the year and selected hours
                                year_cf = np.sum(cell_cf_prof[:8760]) / 8760
                                year_cf = round(year_cf, 4)
                                CF = sum(cell_cf_prof[i] for i in hour_seed) / Hour
                                CF = round(CF[0], 4)
                                cell_gen_poten = cap_scale * provin_cell[pro_flag][(cell_flag[1], cell_flag[2])]

                                if CF >= 0.0020 and year_cf >= 0.0020:
                                    file = "on" if tech == "onshore" else "off"
                                    gen_cost = cf_elep[file][CF] * year_cf / CF
                                    GEN_COST = cf_elep[file][year_cf]
                                if CF >= 0.0020 and gen_cost >= 0 and cell_gen_poten != 0:
                                    cell_sub_dis = c_s_dis[pro_flag][(cell_flag[1], cell_flag[2])]
                                    sub_lc_dis = s_lc_dis[pro_flag][(cell_flag[1], cell_flag[2])]
                                    cell_sub_lat = sub_lat[pro_flag][(cell_flag[1], cell_flag[2])]
                                    cell_sub_lon = sub_lon[pro_flag][(cell_flag[1], cell_flag[2])]

                                    spur_cost = ((spur_capex * cell_sub_dis + spur_capex_fixed) /
                                                 (Hour * CF * math.pow(1 - loss_rate, cell_sub_dis)))
                                    trunk_cost = ((trunk_capex * sub_lc_dis + trunk_capex_fixed) /
                                                  (Hour * CF * math.pow(1 - loss_rate, cell_sub_dis + sub_lc_dis)))

                                    cell_cf_prof = cell_cf_prof.T

                                    inted_count = 0

                                    lon = provin_cell_lon[pro_flag][(cell_flag[1], cell_flag[2])]
                                    lat = provin_cell_lat[pro_flag][(cell_flag[1], cell_flag[2])]
                                    cap_poten = cell_gen_poten / (8760 * year_cf)

                                    # Map existing wind resources to cells to calculate cell usage shares
                                    for pos in integrated.keys():
                                        if (lon - 0.15625) <= pos[0] and pos[0] <= (lon + 0.15625):
                                            if (lat - 0.125) <= pos[1] and pos[1] <= (lat + 0.125):
                                                inted_count += integrated[pos]

                                    if cap_poten != 0:
                                        inted_cof = (inted_count * 0.001) / cap_poten
                                    else:
                                        inted_cof = 0
                                    if inted_cof >= 1:
                                        inted_cof = 1

                                    inted_cof = np.round(inted_cof, 3)

                                    provin_cf_sort[pro_flag].append(
                                        [
                                            str2int(cell_flag[0] + cell_flag[1]),  # 0
                                            CF,  # 1
                                            1 if tech == "onshore" else 0,  # 2
                                            cap_poten,  # 3
                                            np.round(sum(cap_poten * cell_cf_prof[0][h] for h in hour_seed), 4),  # 4
                                            gen_cost,  # 5
                                            0,  # 6
                                            lon,  # 7
                                            lat,  # 8
                                            spur_cost,  # 9
                                            trunk_cost,  # 10
                                            cell_sub_dis,  # 11
                                            sub_lc_dis,  # 12
                                            inted_cof,  # 13
                                            GEN_COST,  # 14
                                            year_cf,  # 15
                                            cell_sub_lat,  # 16
                                            cell_sub_lon  # 17
                                        ]
                                    )
                                    cf_prof[pro_flag].append(cell_cf_prof[0])
                                    provin_cell_genC[pro_flag].append(GEN_COST)
                                break  # 很关键，不要动它 (important - don't remove this line)
            if (tech == "offshore") and (vre_year_single == "2024"):
                vre_cell = pd.read_csv(work_dir + 'data_csv' + dir_flag + "vre_installations" + dir_flag +
                                       f"inter_connect_China_windpower_{tech}_provin_{vre_year_single}.csv",
                                       header=None)
                vre_cell_cf = pd.read_pickle(work_dir + "data_pkl" + dir_flag + "offshore_cfprof_era5_cell_2019.pkl")
                for i in vre_cell.index:
                    cell_lat = vre_cell.iloc[i, 3]
                    cell_lon = vre_cell.iloc[i, 4]
                    cell_pro = vre_cell.iloc[i, 8]
                    cell_cap_gw = vre_cell.iloc[i, 7]
                    cell_cf_prof = np.round(vre_cell_cf[(cell_lon, cell_lat)], 3)

                    year_cf = round(np.sum(cell_cf_prof) / 8760, 4)
                    CF = round(min(sum(cell_cf_prof[i] for i in hour_seed) / Hour, 0.7999), 4)

                    if CF >= 0.0020 and year_cf >= 0.0020:
                        file = "on" if tech == "onshore" else "off"
                        gen_cost = cf_elep[file][CF] * year_cf / CF
                        GEN_COST = cf_elep[file][year_cf]
                    if CF >= 0.0020 and gen_cost >= 0 and cell_cap_gw != 0:
                        cell_sub_lat = vre_cell.iloc[i, 9]
                        cell_sub_lon = vre_cell.iloc[i, 10]
                        cell_sub_dis = vre_cell.iloc[i, 11]
                        sub_lc_dis = vre_cell.iloc[i, 12]

                        spur_cost = ((spur_capex * cell_sub_dis + spur_capex_fixed) /
                                     (Hour * CF * math.pow(1 - loss_rate, cell_sub_dis)))
                        trunk_cost = ((trunk_capex * sub_lc_dis + trunk_capex_fixed) /
                                      (Hour * CF * math.pow(1 - loss_rate, cell_sub_dis + sub_lc_dis)))

                        inted_count = 0
                        for pos in integrated.keys():
                            if (cell_lon - 0.15625) <= pos[0] and pos[0] <= (cell_lon + 0.15625):
                                if (cell_lat - 0.125) <= pos[1] and pos[1] <= (cell_lat + 0.125):
                                    inted_count += integrated[pos]
                        if cell_cap_gw != 0:
                            inted_cof = min((inted_count * 0.001) / cell_cap_gw, 1)
                        else:
                            inted_cof = 0
                        inted_cof = np.round(inted_cof, 3)

                        provin_cf_sort[cell_pro].append(
                            [
                                i,  # 0
                                CF,  # 1
                                1 if tech == "onshore" else 0,  # 2
                                cell_cap_gw,  # 3, cap potential in GW
                                np.round(sum(cell_cap_gw * cell_cf_prof[h] for h in hour_seed), 4),  # 4
                                gen_cost,  # 5
                                0,  # 6
                                cell_lon,  # 7
                                cell_lat,  # 8
                                spur_cost,  # 9
                                trunk_cost,  # 10
                                cell_sub_dis,  # 11
                                sub_lc_dis,  # 12
                                inted_cof,  # 13
                                GEN_COST,  # 14
                                year_cf,  # 15
                                cell_sub_lat,  # 16
                                cell_sub_lon  # 17
                            ]
                        )
                        cf_prof[cell_pro].append(cell_cf_prof)
                        provin_cell_genC[cell_pro].append(GEN_COST)
    elif vre == 'solar':
        for pro in provins:
            provin_cell[pro] = {}
            provin_cell_lon[pro] = {}
            provin_cell_lat[pro] = {}
            c_s_dis[pro] = {}
            s_lc_dis[pro] = {}
            sub_lat[pro] = {}
            sub_lon[pro] = {}

        # Read all the cells with CF and potentials
        with open(work_dir + 'data_pkl' + dir_flag + 'China_solarpower_province_' + vre_year_single + '.pkl',
                  'rb+') as fin:
            vre_cell = pickle.load(fin)
        fin.close()

        # Read all the cell coordinates and distances to substations/load centers
        vre_coordinate = {}
        f_coor = open(work_dir + 'data_csv' + dir_flag + "vre_installations" + dir_flag +
                      'inter_connect_China_solarpower_coordinate_' + vre_year_single + '.csv', 'r+')
        for line in f_coor:
            line = line.replace('\n', '')
            line = line.split(',')
            key = (eval(line[1]), eval(line[2]))
            vre_coordinate[key] = [
                eval(line[4]),  # lat of the nearest substation
                eval(line[5]),  # lon of the nearest substation
                eval(line[6]),  # cell-substation distance
                eval(line[7])  # substation-load center distance
            ]
        f_coor.close()

        for i in vre_cell['index']:  # 24454, including both UPV and DPV cells

            vre_cell['cf'][i] = np.round(vre_cell['cf'][i], 3)
            key = (vre_cell['lat'][i], vre_cell['lon'][i])
            year_cf = round(sum(vre_cell['cf'][i][:8760]) / 8760, 4)
            CF = round(sum(vre_cell['cf'][i][h] for h in hour_seed) / Hour, 4)
            cap_poten = cap_scale * vre_cell['cap'][i]

            ele_gen = cap_scale * round(sum(vre_cell['ele'][i][h] for h in hour_seed), 4)

            if CF >= 0.0020 and year_cf >= 0.0020:
                if vre_cell['isDPV'][i] == 0:
                    gen_cost = cf_elep['on'][CF] * year_cf / CF
                    GEN_COST = cf_elep['on'][year_cf]
                else:
                    gen_cost = cf_elep['off'][CF] * year_cf / CF
                    GEN_COST = cf_elep['off'][year_cf]

            if CF >= 0.0020 and gen_cost >= 0 and cap_poten > 0:
                cell_sub_lat = vre_coordinate[key][0]
                cell_sub_lon = vre_coordinate[key][1]

                cell_sub_dis = vre_coordinate[key][2]
                sub_lc_dis = vre_coordinate[key][3]

                spur_cost = (
                        (spur_capex * cell_sub_dis + spur_capex_fixed)
                        / (Hour * CF * math.pow(1 - loss_rate, cell_sub_dis))
                )

                trunk_cost = (
                        (trunk_capex * sub_lc_dis + trunk_capex_fixed)
                        / (Hour * CF * math.pow(1 - loss_rate, cell_sub_dis + sub_lc_dis))
                )

                inted_count = 0

                for pos in integrated:
                    if (key[1] - 0.15625) <= pos[0] and pos[0] <= (key[1] + 0.15625):
                        if (key[0] - 0.125) <= pos[1] and pos[1] <= (key[0] + 0.125):
                            inted_count += integrated[pos]

                if cap_poten > 0:
                    inted_cof = (0.001 * inted_count) / cap_poten
                else:
                    inted_cof = 0

                if inted_cof > 1:
                    inted_cof = 1

                provin_cf_sort[vre_cell['province'][i]].append([
                    i,  # 0
                    CF,  # 1
                    vre_cell['isDPV'][i],  # 2 是否为分布式光伏,0:不是分布式, 1:是分布式
                    cap_poten,  # 3
                    ele_gen,  # 4
                    gen_cost,  # 5
                    0,  # 6
                    vre_cell['lon'][i],  # 7
                    vre_cell['lat'][i],  # 8
                    spur_cost,  # 9
                    trunk_cost,  # 10
                    cell_sub_dis,  # 11
                    sub_lc_dis,  # 12
                    inted_cof,  # 13
                    GEN_COST,  # 14
                    year_cf,  # 15
                    cell_sub_lat,  # 16
                    cell_sub_lon  # 17
                ])

                cf_prof[vre_cell['province'][i]].append(vre_cell['cf'][i])

                provin_cell_genC[vre_cell['province'][i]].append(GEN_COST)

    for pro in provins:
        provin_cell_genC[pro] = np.array([provin_cell_genC[pro]]).T
        cf_prof[pro] = np.array(cf_prof[pro])
        cf_prof[pro] = np.hstack((provin_cell_genC[pro], cf_prof[pro]))

        provin_cf_sort[pro] = np.array(provin_cf_sort[pro])
        provin_cf_sort[pro] = provin_cf_sort[pro][np.argsort(provin_cf_sort[pro][:, 14])]

        cf_prof[pro] = cf_prof[pro][np.argsort(cf_prof[pro][:, 0])]
        cf_prof[pro] = np.delete(cf_prof[pro], 0, axis=1)

    sub_cell_info = {}

    for pro in provins:
        if pro not in sub_cell_info:
            sub_cell_info[pro] = {}

        for i in range(len(provin_cf_sort[pro])):
            sub_lat = provin_cf_sort[pro][i][16]
            sub_lon = provin_cf_sort[pro][i][17]

            if (sub_lat, sub_lon) not in sub_cell_info[pro]:
                sub_cell_info[pro][(sub_lat, sub_lon)] = {'cell': []}

            if (vre == 'wind') or (vre == 'solar' and provin_cf_sort[pro][i][2] == 0):
                sub_cell_info[pro][(sub_lat, sub_lon)]['cell'].append(i)

            if 'dis' not in sub_cell_info[pro][(sub_lat, sub_lon)]:
                sub_cell_info[pro][(sub_lat, sub_lon)]['dis'] = provin_cf_sort[pro][i][12]

    for pro in provins:
        for s in sub_cell_info[pro]:
            sub_cfs = np.zeros(len(cf_prof[pro][0]))
            sub_cap = 0

            for c in sub_cell_info[pro][s]['cell']:
                sub_cfs += cf_prof[pro][c] * provin_cf_sort[pro][c][3]
                sub_cap += provin_cf_sort[pro][c][3]

            sub_cell_info[pro][s]['cf'] = sub_cfs / sub_cap
            sub_cell_info[pro][s]['cap'] = sub_cap

    re_county = gpd.read_file(work_dir + 'data_shp' + dir_flag + 're_county_level.shp')

    cell_shp = {'pro': [], 'id': [], 'geometry': [], 'inted_cof': [], 'cap': [], 'cost': []}
    for pro in provins:
        for i in range(len(provin_cf_sort[pro])):
            if provin_cf_sort[pro][i][2] == 0:
                lon = provin_cf_sort[pro][i][7]
                lat = provin_cf_sort[pro][i][8]
                coordinate = [(lon - 0.15625, lat - 0.125), (lon + 0.15625, lat - 0.125),
                              (lon + 0.15625, lat + 0.125), (lon - 0.15625, lat + 0.125)]
                polygon = shapely.geometry.Polygon(coordinate)
                cell_shp['pro'].append(pro)
                cell_shp['id'].append(provin_cf_sort[pro][i][0])
                cell_shp['geometry'].append(polygon)
                cell_shp['inted_cof'].append(provin_cf_sort[pro][i][13])
                cell_shp['cap'].append(provin_cf_sort[pro][i][3])
                cell_shp['cost'].append(provin_cf_sort[pro][i][14])

    cell_gdf = gpd.GeoDataFrame(cell_shp, geometry=cell_shp['geometry'], crs='EPSG:4326')

    county_cell_gdf = gpd.sjoin(left_df=cell_gdf, right_df=re_county, op='intersects')

    county_cell_dict = county_cell_gdf.to_dict('list')

    county_cell = {}
    county_cap = {}
    cell_inted_cof = {}

    for i in range(len(county_cell_dict['NAME_3'])):
        county_name = county_cell_dict['NAME_3'][i]

        if county_cell_dict[vre][i] != 0 and county_name not in county_cap:
            county_cap[county_name] = county_cell_dict[vre][i]

        if county_cell_dict['id'][i] not in cell_inted_cof:
            cell_inted_cof[int(county_cell_dict['id'][i])] = county_cell_dict['inted_cof'][i]

        if county_name not in county_cell.keys():
            county_cell[county_name] = []
        else:
            county_cell[county_name].append([county_cell_dict['id'][i],
                                             county_cell_dict['cap'][i],
                                             county_cell_dict['cost'][i]])
    for county in county_cell:
        county_cell[county] = np.array(county_cell[county])
        if len(county_cell[county]) > 1:
            county_cell[county] = county_cell[county][np.argsort(county_cell[county][:, 2])]

    for county in county_cap:
        for i in range(len(county_cell[county])):
            deploy_cof = county_cap[county] / county_cell[county][i][1]

            if deploy_cof + cell_inted_cof[int(county_cell[county][i][0])] >= 1:
                county_cap[county] -= (1 - cell_inted_cof[int(county_cell[county][i][0])]) * county_cell[county][i][1]
                cell_inted_cof[int(county_cell[county][i][0])] = 1
            else:
                county_cap[county] = 0
                if deploy_cof > 0:
                    cell_inted_cof[int(county_cell[county][i][0])] += deploy_cof

    for pro in provins:
        for i in range(len(provin_cf_sort[pro])):
            if provin_cf_sort[pro][i][2] == 0:
                if int(provin_cf_sort[pro][i][0]) in cell_inted_cof:
                    if cell_inted_cof[int(provin_cf_sort[pro][i][0])] > 0:
                        provin_cf_sort[pro][i][13] = cell_inted_cof[int(provin_cf_sort[pro][i][0])]

    # Handle missing grid integration data 处理缺失并网数据
    ppc_file = {'wind': 'province_wind_data.csv', 'solar': 'province_solar_data.csv'}
    ppc = {}  # planned province cap

    ppc_dif = {}
    f_ppc = open(work_dir + 'data_csv' + dir_flag + "vre_installations" + dir_flag + ppc_file[vre], 'r+')
    for line in f_ppc:
        line = line.replace('\n', '')
        line = line.split(',')
        ppc[line[0]] = 0.01 * eval(line[1])  # Original data is in weird units of 10 MW, thus * 0.01 to obtain GW
    f_ppc.close()

    for pro in provins:
        cap_count = 0
        for cell in provin_cf_sort[pro]:
            cap_count += cell[3] * cell[13]

        ppc_dif[pro] = ppc[pro] - cap_count

    for pro in provins:
        dif_cap = ppc_dif[pro]
        for i in range(len(provin_cf_sort[pro])):
            dif_per_inted = dif_cap / provin_cf_sort[pro][i][3]
            if dif_per_inted + provin_cf_sort[pro][i][13] >= 1:
                dif_cap = dif_cap - (1 - provin_cf_sort[pro][i][13]) * provin_cf_sort[pro][i][3]
                provin_cf_sort[pro][i][13] = 1
            else:
                dif_cap = 0
                if dif_per_inted > 0:
                    provin_cf_sort[pro][i][13] += dif_per_inted

    # If not the first-time run, cell files should be based on last year's output only without no adjustments.
    # Make sure installed capacities are greater than last year's output capacity at each cell
    if not os.path.exists(os.path.join(out_output_path_last_year, f"x_{vre}")):
        print(f"No {vre} outputs for setting initial capacity conditions")
        pass
    else:
        print(f"Read {vre} outputs for setting initial capacity conditions")
        for pro in provins:
            try:
                # Read last year's outputs
                with open(os.path.join(out_output_path_last_year, f"x_{vre}", pro+".csv"), 'rb') as fin:
                    fx_vre = pd.read_csv(fin, header=None)
                    fx_vre.columns = ["index", "x_vre"]
                fin.close()

                # Read last year's input cell file
                with open(os.path.join(out_input_path_last_year, f"{vre}_cell.pkl"), 'rb') as fin:
                    vre_cell_last_year = pickle.load(fin)
                fin.close()

                # Iterate over cell ID and find out the index and vre share value based on last year's outputs
                for i in range(len(provin_cf_sort[pro])):
                    cell_id = int(provin_cf_sort[pro][i][0])
                    provin_cf_sort_df_last_year = pd.DataFrame(vre_cell_last_year["provin_cf_sort"][pro])
                    idx_last_year = np.array(provin_cf_sort_df_last_year[
                                                 provin_cf_sort_df_last_year[0] == cell_id].index)[0]

                    # Write last year's vre share values to this year's cell file
                    insted_coef_last_year = fx_vre["x_vre"][idx_last_year]
                    provin_cf_sort[pro][i][13] = insted_coef_last_year
            except ValueError:
                pass

    # Print total installed capacity
    total_cap_inted = 0
    for pro in provins:
        for cell in provin_cf_sort[pro]:
            total_cap_inted += cell[3] * cell[13]
    print(f"Initial conditions for {curr_year} -- {vre} installed GW: {total_cap_inted}")

    # Save wind/solar cell
    save_as_pkl = {'provin_cf_sort': provin_cf_sort, 'cf_prof': cf_prof, 'sub_cell_info': sub_cell_info}
    with open(os.path.join(out_input_path, f"{vre}_cell.pkl"), 'wb+') as fout:
        pickle.dump(save_as_pkl, fout)
    fout.close()


def initProvincialHydroBeta(total, sw, nw, e_c, other):
    """
    Takes in the future projection of major regions. e.g. sw - southwest
    Also read in current hydro capacity for each province 
    Linearly Scale up each province within each region to match the regional
    projection
    """
    pro_loc_in_china = {}

    f_province = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')
    next(f_province)

    for line in f_province:
        line = line.replace('\n', '')
        line = line.split(',')

        if line[6] == 'East' or line[6] == 'Central':
            region = 'e_c'
        elif line[6] == 'SW' or line[6] == 'NW':
            region = line[6]
        else:
            region = 'other'

        pro_loc_in_china[line[1]] = region

    f_province.close()

    region_hydro = {}

    current_hydro = {}

    f_hydro = open(work_dir + 'data_csv' + dir_flag + 'capacity_assumptions/hydro_2020.csv', 'r+')

    for line in f_hydro:
        line = line.replace('\n', '')
        line = line.split(',')

        if pro_loc_in_china[line[0]] not in region_hydro:
            region_hydro[pro_loc_in_china[line[0]]] = 0

        current_hydro[line[0]] = eval(line[1])
        region_hydro[pro_loc_in_china[line[0]]] += eval(line[1])

    f_hydro.close()

    region_hydro_2060 = {
        'SW': 1000 * total * sw,
        'NW': 1000 * total * nw,
        'e_c': 1000 * total * e_c,
        'other': 1000 * total * other
    }

    hydro_beta_province = {}

    for pro in current_hydro:
        region = pro_loc_in_china[pro]
        hydro_beta_province[pro] = region_hydro_2060[region] / region_hydro[region]

    with open(work_dir + 'data_pkl' + dir_flag + 'hydro_beta_province.pkl', 'wb+') as fout:
        pickle.dump(hydro_beta_province, fout)
    fout.close()


def initDemLayer(vre_year: str, res_tag: str, curr_year: int, scen_params: dict):
    """
    Initialize provincial demand layer for a given year.
    """
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", res_tag + "_" + vre_year, str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")

    # Read parameters from the scenario parameter json file
    alpha = scen_params["demand"]["scale"]  # demand scaling factor
    coalBeta = scen_params["coal"]["beta"]  # non-chp coal scaling factor
    nuclearBeta = scen_params["nuclear"]["beta"]  # nuclear capacity scaling factor
    hydroBeta = scen_params["hydro"]["beta"]  # hydro capacity scaling factor
    gasBeta = scen_params["gas"]["beta"]  # gas capacity scaling factor
    bioGama = scen_params["beccs"]["must_run"]  # BECCS must-run level
    ccsLoss = scen_params["ccs"]["bio_loss"]  # CCS electricity generation efficiency factor

    # Read province hourly demand folder for a given year
    # Note that a sample folder for 2060 is provided in the data_csv folder
    dem_folder = os.path.join(out_path, "provin_demand_hourly")

    # Read province demand for 2030
    grid4_dem = scio.loadmat(work_dir + "data_mat" + dir_flag + 'RegionDemand_Rev2.mat')
    grid4_dem = grid4_dem['Region_dem'][0]
    provins = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = provins.sort_values(by="provin_py").reset_index(drop=True)
    dem2030 = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'demand_assumptions/province_dem_2030.csv')

    # Initialize demand dictionaries
    provin_dem2030 = {}
    provin_hour_dem2030 = {}
    provin_hour_dem_year = {}

    for i in range(len(dem2030['provin'])):
        provin_dem2030[dem2030['provin'][i]] = dem2030['dem'][i]

    provin_py = list(provins['provin_py'])
    provin_reg = list(provins['region'])
    regions = {'NE': 0, 'NW': 1, 'S': 2, 'SH': 3}
    pro_reg = {}  # T: xizang
    pro_reg_cof = {}
    reg_tot_dem2030 = {'NE': 0, 'NW': 0, 'S': 0, 'SH': 0, 'T': 0}

    for i in range(len(provin_py)):
        pro_reg[provin_py[i]] = provin_reg[i]

    # # Anhui:SH
    # for pro in provin_dem2030:
    #     reg_tot_dem2030[pro_reg[pro]] += provin_dem2030[pro]
    #
    # for pro in pro_reg:
    #     pro_reg_cof[pro] = provin_dem2030[pro] / reg_tot_dem2030[pro_reg[pro]]
    #
    # for pro in pro_reg:
    #     if pro_reg[pro] != 'T':
    #         provin_hour_dem2030[pro] = scale * pro_reg_cof[pro] * grid4_dem[regions[pro_reg[pro]]][:]
    #     elif pro_reg[pro] == 'T':
    #         provin_hour_dem2030[pro] = scale * (provin_dem2030[pro] / reg_tot_dem2030['NE']) * grid4_dem[0][:]
    #
    # # Project demand based on 2030 demand data
    # for pro in pro_reg:
    #     if pro not in provin_hour_dem_year:
    #         provin_hour_dem_year[pro] = []
    #     for i in range(len(provin_hour_dem2030[pro])):
    #         provin_hour_dem2030[pro][i][0] = format(provin_hour_dem2030[pro][i][0], '.4f')
    #         provin_hour_dem_year[pro].append(alpha * provin_hour_dem2030[pro][i][0])
    #
    # # Write province hourly demand data for the given year
    # for pro in provin_hour_dem_year:
    #     f_pro_dem = open(os.path.join(dem_folder, pro + '.csv'), 'w+')
    #     for h in range(len(provin_hour_dem_year[pro])):
    #         f_pro_dem.write('%s,%s\n' % (h, provin_hour_dem_year[pro][h]))
    #     f_pro_dem.close()

    # Read the provincial hourly demand folder

    # Directly read from the interpolated demand folder
    for pro in pro_reg:
        f_pro_dem = pd.read_csv(os.path.join(dem_folder, pro + '.csv'))
        provin_hour_dem_year[pro] = f_pro_dem["dem"].to_list()

    # Calculate the sum of province demand data
    f_nation_dem_full = open(os.path.join(dem_folder, 'nation_dem_full.csv'), 'w+')
    for h in range(8760):
        dem_h = 0
        for pro in provin_hour_dem_year:
            dem_h += provin_hour_dem_year[pro][h]
        f_nation_dem_full.write(str(h) + ',' + str(dem_h) + '\n')
    f_nation_dem_full.close()

    # Save the provincial demand data as a pickle file
    with open(os.path.join(out_input_path, "province_demand_full_" + str(curr_year) + ".pkl"), 'wb') as fout:
        pickle.dump(provin_hour_dem_year, fout)
    fout.close()

    # Save the summary of the demand dictionary as a pickle file
    sum_dem = 0
    for pro in pro_reg:
        sum_dem += sum(provin_hour_dem_year[pro])
    print(f'Total demand in {curr_year}:', sum_dem)
    with open(os.path.join(out_input_path, "total_demand_" + str(curr_year) + ".pkl"), 'wb') as fout:
        pickle.dump({'total_demand': sum_dem}, fout)
    fout.close()

    # Calculate layer capacity by resource type and by province of a given year
    tech_set = ['coal', 'chp_ccs', 'beccs', 'hydro', 'nuclear', 'bio', 'gas']
    pro_layer_cap_gw = {}

    for tech in tech_set:
        df = pd.read_csv(os.path.join(out_path, tech + ".csv"))
        for idx in df.index:
            province_name = df.iloc[idx, 0]
            province_cap = df.iloc[idx, 1]
            if province_name not in pro_layer_cap_gw.keys():
                pro_layer_cap_gw[province_name] = {}
            pro_layer_cap_gw[province_name][tech] = np.round(province_cap / 1000, 2)  # MW -> capacity in GW by province
            if tech == "coal":
                pro_layer_cap_gw[province_name]["coal_unabated"] = np.round(df.iloc[idx, 2] / 1000,2)
                pro_layer_cap_gw[province_name]["coal_ccs"] = np.round(df.iloc[idx, 3] / 1000, 2)
            elif tech == "gas":
                pro_layer_cap_gw[province_name]["gas_unabated"] = np.round(df.iloc[idx, 2] / 1000, 2)
                pro_layer_cap_gw[province_name]["gas_ccs"] = np.round(df.iloc[idx, 3] / 1000, 2)
            else:
                pass

    # Calculate layer capacity profiles for the 4 main grid regions
    mat_name = 'levels_reg_Rev2_CHPfree_CHPmid_05_peakwk5margin.mat'
    grid4_layer = scio.loadmat(work_dir + 'data_mat' + dir_flag + mat_name)
    grid4_layer = grid4_layer['levels_reg'][0]

    layer_cap_profile = []
    for reg in regions:
        layer_cap_profile.append(np.zeros((len(grid4_layer[regions[reg]]), 4), float))
    for reg in regions:
        layer_cap_profile[regions[reg]][:, :-1] = grid4_layer[regions[reg]][:, 1:] - grid4_layer[regions[reg]][:, :-1]

    layer_cap = {'NW': [], 'NE': [], 'SH': [], 'S': []}
    for reg in layer_cap:
        layer_cap[reg].append(max(layer_cap_profile[regions[reg]][:, 0]))
        layer_cap[reg].append(max(layer_cap_profile[regions[reg]][:, 1]))
        layer_cap[reg].append(max(layer_cap_profile[regions[reg]][:, 2]))
        layer_cap[reg].append(max(layer_cap_profile[regions[reg]][:, 3]))

    # For each province, obtain the hourly capacity factor profile of each resource type
    pro_layer_cf = {}
    for pro in provin_py:
        if pro not in pro_layer_cf:
            pro_layer_cf[pro] = {}

        pro_layer_cf[pro]["nuclear_hourly_cf"] = np.ones(8760)
        pro_layer_cf[pro]["coal_hourly_cf"] = np.ones(8760)
        pro_layer_cf[pro]["chp_ccs_hourly_cf"] = np.ones(8760)
        pro_layer_cf[pro]["hydro_hourly_cf"] = np.ones(8760)
        pro_layer_cf[pro]["bio_hourly_cf"] = np.ones(8760)
        pro_layer_cf[pro]["beccs_hourly_cf"] = np.ones(8760)
        pro_layer_cf[pro]["gas_hourly_cf"] = np.ones(8760)
        for h in range(0, 8760):
            if pro != 'Xizang':
                # pro_layer_cf[pro]["coal_hourly_cf"][h] = layer_cap_profile[regions[pro_reg[pro]]][h][2] / \
                #                                          layer_cap[pro_reg[pro]][2]
                pro_layer_cf[pro]["hydro_hourly_cf"][h] = 0.6 * layer_cap_profile[regions[pro_reg[pro]]][h][1] / \
                                                          layer_cap[pro_reg[pro]][1]
                # pro_layer_cf[pro]["bio_hourly_cf"][h] = layer_cap_profile[regions[pro_reg[pro]]][h][2] / \
                #                                         layer_cap[pro_reg[pro]][2]
                # pro_layer_cf[pro]["gas_hourly_cf"][h] = layer_cap_profile[regions[pro_reg[pro]]][h][2] / \
                #                                         layer_cap[pro_reg[pro]][2]
            else:
                # pro_layer_cf[pro]["coal_hourly_cf"][h] = layer_cap_profile[1][h][2] / layer_cap['NW'][2]
                pro_layer_cf[pro]["hydro_hourly_cf"][h] = 0.6 * layer_cap_profile[1][h][1] / layer_cap['NW'][1]
                # pro_layer_cf[pro]["bio_hourly_cf"][h] = layer_cap_profile[1][h][2] / layer_cap['NW'][2]
                # pro_layer_cf[pro]["gas_hourly_cf"][h] = layer_cap_profile[1][h][2] / layer_cap['NW'][2]

        pro_layer_cf[pro]["coal_unabated_hourly_cf"] = pro_layer_cf[pro]["coal_hourly_cf"]
        pro_layer_cf[pro]["coal_ccs_hourly_cf"] = pro_layer_cf[pro]["coal_hourly_cf"]
        pro_layer_cf[pro]["gas_unabated_hourly_cf"] = pro_layer_cf[pro]["gas_hourly_cf"]
        pro_layer_cf[pro]["gas_ccs_hourly_cf"] = pro_layer_cf[pro]["gas_hourly_cf"]
        pro_layer_cf[pro]["beccs_hourly_cf"] = pro_layer_cf[pro]["beccs_hourly_cf"]
        pro_layer_cf[pro]["bio_hourly_cf"] = pro_layer_cf[pro]["bio_hourly_cf"]

    # Calculate hourly layer capacity in GW and net demand in GW
    pro_layer_cap = {}
    for pro in provin_py:
        if pro not in pro_layer_cap:
            pro_layer_cap[pro] = {}

        pro_layer_cap[pro]["coal"] = np.round(coalBeta * pro_layer_cf[pro]["coal_hourly_cf"] *
                                              pro_layer_cap_gw[pro]["coal"], 2)
        pro_layer_cap[pro]["coal_unabated"] = np.round(coalBeta * pro_layer_cf[pro]["coal_unabated_hourly_cf"] *
                                                       pro_layer_cap_gw[pro]["coal"], 2)
        pro_layer_cap[pro]["coal_ccs"] = np.round(coalBeta * pro_layer_cf[pro]["coal_ccs_hourly_cf"] *
                                                  pro_layer_cap_gw[pro]["coal"], 2)
        pro_layer_cap[pro]["chp_ccs"] = np.round(coalBeta * pro_layer_cf[pro]["chp_ccs_hourly_cf"] *
                                                 pro_layer_cap_gw[pro]["chp_ccs"], 2)
        pro_layer_cap[pro]["hydro"] = np.round(hydroBeta * pro_layer_cf[pro]["hydro_hourly_cf"] *
                                               pro_layer_cap_gw[pro]["hydro"], 2)
        pro_layer_cap[pro]["nuclear"] = np.round(nuclearBeta * pro_layer_cf[pro]["nuclear_hourly_cf"] *
                                                 pro_layer_cap_gw[pro]["nuclear"], 2)
        pro_layer_cap[pro]["gas"] = np.round(gasBeta * pro_layer_cf[pro]["gas_hourly_cf"] *
                                             pro_layer_cap_gw[pro]["gas"], 2)
        pro_layer_cap[pro]["gas_unabated"] = np.round(gasBeta * pro_layer_cf[pro]["gas_unabated_hourly_cf"] *
                                                      pro_layer_cap_gw[pro]["gas"], 2)
        pro_layer_cap[pro]["gas_ccs"] = np.round(gasBeta * pro_layer_cf[pro]["gas_ccs_hourly_cf"] *
                                                 pro_layer_cap_gw[pro]["gas"], 2)
        pro_layer_cap[pro]["beccs"] = np.round(pro_layer_cf[pro]["beccs_hourly_cf"] *
                                               pro_layer_cap_gw[pro]["beccs"], 2)
        pro_layer_cap[pro]["bio"] = np.round(pro_layer_cf[pro]["bio_hourly_cf"] *
                                               pro_layer_cap_gw[pro]["bio"], 2)

    # Aggregate all the calculation results
    layer_cap_load = {'layer_cf_norm': pro_layer_cf,
                      'layer_cap_total': pro_layer_cap_gw,
                      'layer_cap_hourly': pro_layer_cap  # Multiply normalized CFs with total layer capacities
                      }

    # Save the layer capacity dictionary of the given year as a pickle file
    with open(os.path.join(out_input_path, "layer_cap_load.pkl"), 'wb') as fout:
        pickle.dump(layer_cap_load, fout)
    fout.close()


def initModelExovar(vre_year: str, res_tag: str, curr_year: int, last_year: int, scen_params: dict):
    """
    Store exogenous and endogenous variable in place.
    """
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", res_tag + "_" + vre_year, str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")

    out_path_last_year = os.path.join(work_dir, "data_res", res_tag + "_" + vre_year, str(last_year))
    out_output_path_last_year = os.path.join(out_path_last_year, "outputs")
    out_output_processed_path_last_year = os.path.join(out_path_last_year, "outputs_processed")

    # Specify parameters
    scale = 0.001
    weighted_average_cost_of_capital = scen_params["finance"]["weighted_average_cost_of_capital"]
    lifespan_transmission_1 = 25  # yrs, substation?
    lifespan_transmission_2 = 50  # yrs, overhead line?

    # Read inter-provincial transmission line capability from the default folder or last year's output
    inter_pro_trans = {}
    if os.path.exists(os.path.join(out_output_processed_path_last_year, f"inter_pro_trans_{last_year}.csv")):
        print("Read transmission line matrix from last year's outputs")
        finter_pro_trans = pd.read_csv(os.path.join(out_output_processed_path_last_year,
                                                    f"inter_pro_trans_{last_year}.csv"))
    else:
        print("Read transmission line matrix from the default folder")
        finter_pro_trans = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'transmission_assumptions/inter_pro_trans.csv')
    finter_pro_trans.to_csv(os.path.join(out_input_path, "inter_pro_trans.csv"),
                            header=True, index=False)
    for pro in finter_pro_trans['province']:
        for i in range(len(finter_pro_trans[pro])):
            inter_pro_trans[(pro, finter_pro_trans['province'][i])] = scale * finter_pro_trans[pro][i]

    # Read the province list of China
    grid_pro = {}
    f_pi = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    next(f_pi)
    for line in f_pi:
        line = line.replace('\n', '')
        line = line.split(',')
        if line[5] not in grid_pro:
            grid_pro[line[5]] = []
        grid_pro[line[5]].append(line[1])
    f_pi.close()

    # Read the csv file that indicates whether there are existing transmission lines between provinces
    fnew_trans_cap = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'transmission_assumptions/new_trans.csv')
    new_trans_cap = {}
    for pro in fnew_trans_cap['province']:
        for i in range(len(fnew_trans_cap[pro])):
            if fnew_trans_cap[pro][i] == 1.0:
                new_trans_cap[(pro, fnew_trans_cap['province'][i])] = 1
            else:
                new_trans_cap[(pro, fnew_trans_cap['province'][i])] = 0

    # Obtain the distances between two provinces where there are existing transmission lines
    provin_coord = {}
    f_province = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')
    next(f_province)
    for line in f_province:
        line = line.replace('\n', '')
        line = line.split(',')
        provin_coord[line[1]] = []
        provin_coord[line[1]].append(eval(line[3]))
        provin_coord[line[1]].append(eval(line[4]))
    f_province.close()
    inter_pro_dis = {}
    for pro in new_trans_cap:
        inter_pro_dis[pro] = geo_distance(provin_coord[pro[0]][1], provin_coord[pro[0]][0],
                                          provin_coord[pro[1]][1], provin_coord[pro[1]][0])

    # Read voltage convert data
    voltage_convert = {}
    f_vc = open(work_dir + 'data_csv' + dir_flag + 'transmission_assumptions/voltage_convert.csv', 'r+')
    for line in f_vc:
        line = line.replace('\n', '')
        line = line.split(',')
        voltage_convert[line[0]] = line[1]
    f_vc.close()

    # Read the voltage levels of the transmission lines between two provinces
    trans_voltage = {}
    tv_keys = []
    voltage_kind = []
    f_tv = open(work_dir + 'data_csv' + dir_flag + 'transmission_assumptions/trans_voltage.csv', 'r+', encoding='utf-8')
    for line in f_tv:
        line = line.replace('\n', '')
        line = line.split(',')
        start = extractProvinceName(line[1])
        end = extractProvinceName(line[2])

        key = (start, end)
        tv_keys.append(key)

        if line[0] in voltage_convert:
            line[0] = voltage_convert[line[0]]
        if eval(line[0]) not in voltage_kind:
            voltage_kind.append(eval(line[0]))
        if key not in trans_voltage:
            trans_voltage[key] = eval(line[0])
        elif eval(line[0]) > trans_voltage[key]:
            trans_voltage[key] = eval(line[0])
    f_tv.close()
    for pro in tv_keys:
        if (pro[1], pro[0]) not in trans_voltage:
            trans_voltage[(pro[1], pro[0])] = trans_voltage[pro]
        else:
            if trans_voltage[(pro[1], pro[0])] < trans_voltage[pro]:
                trans_voltage[(pro[1], pro[0])] = trans_voltage[pro]

    # Read transmission line capex (RMB/kW)
    capex_trans_cap = {}
    f_ctc = open(work_dir + 'data_csv' + dir_flag + 'cost_assumptions/capex_trans_cap.csv', 'r+')
    for line in f_ctc:
        line = line.replace('\n', '')
        line = line.split(',')
        capex_trans_cap[eval(line[0])] = eval(line[1]) * \
                                         getCRF(weighted_average_cost_of_capital, lifespan_transmission_1)
    f_ctc.close()

    # Read transmission line capex
    capex_trans_dis = {}
    f_ctd = open(work_dir + 'data_csv' + dir_flag + 'cost_assumptions/capex_trans_dis.csv', 'r+')
    for line in f_ctd:
        line = line.replace('\n', '')
        line = line.split(',')
        capex_trans_dis[eval(line[0])] = eval(line[1]) * \
                                         getCRF(weighted_average_cost_of_capital, lifespan_transmission_2)
    f_ctd.close()

    # Read PHS capacity upper bounds based on 2030-2060 data
    phs_ub = {}
    f_phs_ub = open(work_dir + 'data_csv' + dir_flag + 'capacity_assumptions/phs_ub.csv', 'r+')
    for line in f_phs_ub:
        line = line.replace('\n', '')
        line = line.split(',')
        phs_ub[line[0]] = scale * eval(line[1])
    f_phs_ub.close()
    # if curr_year == 2030:
    #     f_phs_ub_2030 = open(work_dir + 'data_csv' + dir_flag + 'phs_ub_2030.csv', 'r+')
    #     for line in f_phs_ub_2030:
    #         line = line.replace('\n', '')
    #         line = line.split(',')
    #         phs_ub[line[0]] = scale * eval(line[1])
    #     f_phs_ub_2030.close()
    # elif curr_year == 2040:
    #     f_phs_ub_2040 = open(work_dir + 'data_csv' + dir_flag + 'phs_ub_2040.csv', 'r+')
    #     for line in f_phs_ub_2040:
    #         line = line.replace('\n', '')
    #         line = line.split(',')
    #         phs_ub[line[0]] = scale * eval(line[1])
    #     f_phs_ub_2040.close()

    # Read PHS/BAt/LDS capacity lower bounds based on 2020 data OR last year's outputs
    phs_lb, bat_lb, caes_lb, vrb_lb = {}, {}, {}, {}
    if os.path.exists(os.path.join(out_output_processed_path_last_year, f"integrated_storage_{last_year}.csv")):
        print("Read storage capacity from last year's outputs")
        storage_df = pd.read_csv(os.path.join(out_output_processed_path_last_year,
                                              f"integrated_storage_{last_year}.csv"))
        for i in range(len(storage_df["province"])):
            phs_lb[storage_df["province"].iloc[i]] = storage_df["phs_total_capacity_mw"].iloc[i] * scale
            bat_lb[storage_df["province"].iloc[i]] = storage_df["bat_total_capacity_mw"].iloc[i] * scale
            caes_lb[storage_df["province"].iloc[i]] = storage_df["caes_total_capacity_mw"].iloc[i] * scale
            vrb_lb[storage_df["province"].iloc[i]] = storage_df["vrb_total_capacity_mw"].iloc[i] * scale
    else:
        print("Read storage capacity from the default folder")
        f_phs_lb = open(work_dir + 'data_csv' + dir_flag + 'capacity_assumptions/phs_lb.csv', 'r+')
        for line in f_phs_lb:
            line = line.replace('\n', '')
            line = line.split(',')
            phs_lb[line[0]] = scale * eval(line[1])
            bat_lb[line[0]] = 0
            caes_lb[line[0]] = 0
            vrb_lb[line[0]] = 0
        f_phs_lb.close()

    # Read Hydro/Nuclear/BECCS capacity lower bounds based on 2020 data OR last year's outputs
    hydro_lb, nuclear_lb, beccs_lb, bio_lb = {}, {}, {}, {}
    if os.path.exists(os.path.join(out_output_path_last_year, "emissionBreakdowns.csv")):
        print("Read firm capacity from last year's outputs")
        firm_df = pd.read_csv(os.path.join(out_output_path_last_year, "emissionBreakdowns.csv"))
        for i in range(len(firm_df["province"])):
            hydro_lb[firm_df["province"].iloc[i]] = firm_df["cap_hydro_gw"].iloc[i]
            nuclear_lb[firm_df["province"].iloc[i]] = firm_df["cap_nuclear_gw"].iloc[i]
            beccs_lb[firm_df["province"].iloc[i]] = firm_df["cap_beccs_gw"].iloc[i]
            bio_lb[firm_df["province"].iloc[i]] = firm_df["cap_bio_gw"].iloc[i]
    else:
        print("Read firm capacity from the default folder")
        f_nuclear_lb = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "nuclear_2020.csv"),
                                   header=None)
        f_nuclear_lb.columns = ["province", "cap_mw"]
        f_hydro_lb = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "hydro_2020.csv"),
                                 header=None)
        f_hydro_lb.columns = ["province", "cap_mw"]
        for i in range(len(f_nuclear_lb["province"])):
            pro = f_hydro_lb["province"].iloc[i]
            hydro_lb[pro] = f_hydro_lb["cap_mw"].iloc[i] * scale
            nuclear_lb[pro] = f_nuclear_lb["cap_mw"].iloc[i] * scale
            beccs_lb[pro] = 0
            bio_lb[pro] = 0

    # Save the exogenous variables as a pickle file
    save_as_pkl = {
        'inter_pro_trans': inter_pro_trans,
        'new_trans': new_trans_cap,
        'trans_dis': inter_pro_dis,
        'trans_voltage': trans_voltage,
        'capex_trans_cap': capex_trans_cap,
        'capex_trans_dis': capex_trans_dis,
        'grid_pro': grid_pro,
        'phs_ub': phs_ub,
        'phs_lb': phs_lb,
        'bat_lb': bat_lb,
        'caes_lb': caes_lb,
        'vrb_lb': vrb_lb,
        'hydro_lb': hydro_lb,
        'nuclear_lb': nuclear_lb,
        'beccs_lb': beccs_lb,
        'bio_lb': bio_lb
    }
    with open(os.path.join(out_input_path, "model_exovar.pkl"), 'wb') as fout:
        pickle.dump(save_as_pkl, fout)
    fout.close()


def InitSenarioParams(vre_year, res_tag):
    """
    Initialize the parameters for the scenario. Mostly cost related parameters
    """
    res_dir = makeDir(getResDir(vre_year, res_tag))

    scen_params = {
        'trans': {},
        'storage': {},
        'vre': {},
        'nuclear': {},
        'demand': {},
        'resv': {},
        'coal': {},
        'ramp': {},
        'ccs': {},
        'shedding': {},
        'gas': {}
    }

    scen_params['trans'] = {
        'capex_spur_fixed': 1 * 262,  # yuan/kw
        'capex_spur_var': 1 * 3,  # yuan/kw-km
        'capex_trunk_fixed': 1 * 159,
        'capex_trunk_var': 1 * 1.76,
        'trans_loss': 0.000032,
        'interprovincial_scale': 1,
        'is_full_inter_province': 0,
        'trunk_inter_province': 0
    }

    scen_params['storage'] = {
        'duration_phs': 1 * 8,
        'duration_bat': 1 * 4,
        'duration_lds': {'caes': 1 * 20, 'vrb': 1 * 10},
        'sdiss_phs': 0,  # self discharge: %/h
        'sdiss_bat': 0,
        'sdiss_lds': {'caes': 1 * 0, 'vrb': 1 * 0},
        'capex_power_phs': 1 * 3840,  # yuan/kw
        'capex_power_bat': 1 * 480,
        'capex_power_lds': {'caes': 1 * 4800, 'vrb': 1 * 3600},
        'capex_energy_phs': 1 * 0,
        'capex_energy_bat': 1 * 480,
        'capex_energy_lds': {'caes': 1 * 0, 'vrb': 1 * 0},
        'rt_effi_phs': 0 + 0.78,
        'rt_effi_bat': 0 + 0.95,
        'rt_effi_lds': {'caes': 0 + 0.52, 'vrb': 0 + 0.7},
        'fixed_omc_phs': 1 * 39,
        'fixed_omc_bat': 1 * 18,
        'fixed_omc_lds': {'caes': 1 * 5, 'vrb': 1 * 18},
        'var_omc_phs': 1.5 * 0.001,
        'var_omc_bat': 20 * 0.001,
        'var_omc_lds': {'caes': 50 * 0.001, 'vrb': 20 * 0.001},
        'with_lds': 0,
        'with_caes': 0,
        'with_vrb': 0,
        'span_lds': {'caes': 30, 'vrb': 15}
    }

    scen_params['vre'] = {
        'cap_scale_pv': 1,
        'cap_scale_wind': 1,
        'cap_scale_wind_ep': 1,
        'cap_scale_pv_ep': 1,
        'capex_equip_pv': 1 * 1100,
        'capex_equip_dpv': 1 * 1400,
        'capex_equip_on_wind': 1 * 2200,
        'capex_equip_off_wind': 1 * 3800,
        'capex_other_pv': 1 * 400,
        'capex_other_dpv': 1 * 600,
        'capex_other_on_wind': 1 * 800,
        'capex_other_off_wind': 1 * 1600,
        'capex_om_on_wind': 1 * 45,
        'capex_om_off_wind': 1 * 81,
        'capex_om_pv': 1 * 7.5,
        'capex_om_dpv': 1 * 10,
        'aggregated': 0,
        'inter_annual': 0,
        'wind_with_xz': 0
    }

    scen_params['nuclear'] = {
        'must_run': 0 + 0.85,
        'ramp': 0 + 0.05,
        'var_cost': 0.09,
        'beta': 1
    }

    scen_params['gas'] = {
        'beta': 3.3
    }

    scen_params['demand'] = {
        'scale': 1
    }

    scen_params['resv'] = {
        'vre_resv': 0.0 + 0.05,
        'demand_resv': 0.0 + 0.05,
        'with_demand_resv': 1,
        'with_vre_resv': 1,
        'vre_resv_provincialy': 0
    }

    scen_params['coal'] = {
        'min_coal': 0 + 0.15,
        'beta': 0,
        'theta': 0
    }

    scen_params['ramp'] = {
        'l1': 0.25,
        'l2': 0.25,
        'l3': 0.05,
        'l4': 1
    }

    scen_params['ccs'] = {
        'coal_loss': 0.05,
        'gas_loss': 0.05,
        'bio_loss': 0.05,
        'coal_lcoe': 0.3095,
        'gas_lcoe': 0.62,
        'bio_lcoe': 0.59,
        'beccs_cf': 0.8
    }

    scen_params['shedding'] = {
        'with_shedding': 0,
        'shedding_vom': 2,  # yuan/kWh
        'shedding_cof': 0.1,
    }

    with open(res_dir + dir_flag + 'scen_params.pkl', 'wb+') as fout:
        pickle.dump(scen_params, fout)
    fout.close()

    scen_params_json = json.dumps(scen_params)
    fout = open(res_dir + dir_flag + 'scen_params.json', 'w+')
    fout.write(scen_params_json)
    fout.close()


if __name__ == '__main__':

    """
    Before running the test cases below, make sure that the corresponding result folder exists.
    If not, use multiYearAutomation.py to generate multi-year inputs.
    """

    # Specify scenario parameters
    with open(os.path.join(work_dir, "data_csv", "scen_params_template.json")) as f:
        scen_params = json.load(f)

    # Test winterHour()
    test_flag = True
    if test_flag:
        winterHour()

    # Test seedHour()
    test_flag = True
    if test_flag:
        seedHour(vre_year="w2015_s2015", years=0, step=10, days=10, res_tag="test", curr_year=2060)

    # Test initCellData()
    test_flag = True
    if test_flag:
        hour_seed = pd.read_csv(os.path.join(work_dir, "data_csv", "simulation_meta", "hour_seed.csv"),
                                header=None).iloc[:, 0].to_list()
        initCellData(vre="solar", vre_year_single="2015", hour_seed=hour_seed, res_tag="test",
                     vre_year="w2015_s2015", curr_year=2060, last_year=2050, scen_params=scen_params)
        initCellData(vre="wind", vre_year_single="2015", hour_seed=hour_seed, res_tag="test",
                     vre_year="w2015_s2015", curr_year=2060, last_year=2050, scen_params=scen_params)

    # Test initDemLayer()
    test_flag = True
    if test_flag:
        initDemLayer(vre_year="w2015_s2015", res_tag="test", curr_year=2060, scen_params=scen_params)

    # Test initModelExovar()
    test_flag = True
    if test_flag:
        initModelExovar(vre_year="w2015_s2015", res_tag="test", curr_year=2060, last_year=2050, scen_params=scen_params)
