import os
import pickle
import json

import numpy as np
import pandas as pd

from callUtility import dirFlag, getResDir, getWorkDir, makeDir, getProvinName, VreYearSplit, SplitMultipleVreYear, \
    getCRF

dir_flag = dirFlag()
work_dir = getWorkDir()


def averageStorageLength(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    # Specify magic numbers
    rt_rate = {'phs': 0.78, 'bat': 0.95}
    sd_rate = {'phs': 0.0, 'bat': 0.0}

    provins = []
    f_provins = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')
    next(f_provins)
    for line in f_provins:
        line = line.replace('\n', '')
        line = line.split(',')
        provins.append(line[1])
    f_provins.close()

    fhour_seed = open(os.path.join(out_input_path, 'hour_seed.csv'), 'r+')
    hour_seed = []
    for line in fhour_seed:
        line = line.replace('\n', '')
        line = eval(line)
        hour_seed.append(line)
    fhour_seed.close()

    ST = ['phs', 'bat']
    store_cap = {'phs': {}, 'bat': {}}
    for st in ST:
        fsc = open(os.path.join(out_output_path, 'es_cap', 'es_' + st + '_cap.csv'), 'r+')
        for line in fsc:
            line = line.replace('\n', '')
            line = line.split(',')
            store_cap[st][line[0]] = eval(line[1])
        fsc.close()

    char = {'bat': {}, 'phs': {}}
    dis_char = {'bat': {}, 'phs': {}}

    for pro in provins:
        for st in ST:
            if pro not in char[st]:
                char[st][pro] = np.zeros(8760)
                dis_char[st][pro] = np.zeros(8760)

            f_l2_char = open(os.path.join(out_output_path, 'es_char', st, 'l2', pro + '.csv'), 'r+')
            for line in f_l2_char:
                line = line.replace('\n', '')
                line = eval(line)
                char[st][pro][line[0]] += line[1]
            f_l2_char.close()

            f_l3_char = open(os.path.join(out_output_path, 'es_char', st, 'l3', pro + '.csv'), 'r+')
            for line in f_l3_char:
                line = line.replace('\n', '')
                line = eval(line)
                char[st][pro][line[0]] += line[1]
            f_l3_char.close()

            f_w_char = open(os.path.join(out_output_path, 'es_char', st, 'wind', pro + '.csv'), 'r+')
            for line in f_w_char:
                line = line.replace('\n', '')
                line = eval(line)
                char[st][pro][line[0]] += line[1]
            f_w_char.close()

            f_s_char = open(os.path.join(out_output_path, 'es_char', st, 'solar', pro + '.csv'), 'r+')
            for line in f_s_char:
                line = line.replace('\n', '')
                line = eval(line)
                char[st][pro][line[0]] += line[1]
            f_s_char.close()

            f_c_dischar = open(os.path.join(out_output_path, 'es_inte', st, pro + '.csv'), 'r+')
            for line in f_c_dischar:
                line = line.replace('\n', '')
                line = eval(line)
                dis_char[st][pro][line[0]] += line[1]
            f_c_dischar.close()

    char_in_hs = {'bat': {}, 'phs': {}}
    dis_char_in_hs = {'bat': {}, 'phs': {}}

    for st in ST:
        for pro in char[st]:
            if store_cap[st][pro] > 0:
                if pro not in char_in_hs[st]:
                    char_in_hs[st][pro] = []
                    dis_char_in_hs[st][pro] = []
                for i in range(8760):
                    if i in hour_seed:
                        char_in_hs[st][pro].append(char[st][pro][i])
                        dis_char_in_hs[st][pro].append(dis_char[st][pro][i])

    char_length = {'bat': {}, 'phs': {}}
    aveCL = {'bat': {}, 'phs': {}}

    for st in ST:
        for pro in char_in_hs[st]:
            if pro not in char_length:
                char_length[st][pro] = []
            for i in range(len(char_in_hs[st][pro])):
                cur_length = 0
                # print(pro,char_in_hs[st][pro][i])
                if char_in_hs[st][pro][i] == 0:
                    char_length[st][pro].append(0)
                    continue
                else:
                    char_cur = rt_rate[st] * char_in_hs[st][pro][i]
                    for j in range(i + 1, len(char_in_hs[st][pro])):
                        char_cur = (1 - sd_rate[st]) * char_cur
                        if char_cur == 0:
                            break
                        else:
                            if dis_char_in_hs[st][pro][j] > 0:
                                if dis_char_in_hs[st][pro][j] >= char_cur:
                                    dis_char_in_hs[st][pro][j] -= char_cur
                                    cur_length += char_cur * (j - i)
                                    char_cur = 0
                                else:
                                    char_cur -= dis_char_in_hs[st][pro][j]
                                    cur_length += dis_char_in_hs[st][pro][j] * (j - i)
                                    dis_char_in_hs[st][pro][j] = 0
                    char_length[st][pro].append(cur_length)

    for st in ST:
        for pro in char_in_hs[st]:
            if pro not in aveCL[st]:
                aveCL[st][pro] = []
            for i in range(len(char_in_hs[st][pro])):
                if char_in_hs[st][pro][i] != 0:
                    aveCL[st][pro].append(char_length[st][pro][i] / char_in_hs[st][pro][i])

    # Save all the processed outputs
    folder = makeDir(os.path.join(out_output_processed_path, 'store_length'))
    with open(os.path.join(folder, 'aveCL.pkl'), 'wb+') as fout:
        pickle.dump(aveCL, fout)
    fout.close()

    for st in ST:
        folder = makeDir(os.path.join(out_output_processed_path, 'store_length', st))
        for pro in aveCL[st]:
            f_avecl = open(os.path.join(folder, pro + '_aveCL.csv'), 'w+')
            for i in range(len(aveCL[st][pro])):
                f_avecl.write('%s\n' % aveCL[st][pro][i])
            f_avecl.close()


def cellResInfo(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    path_wind_cell = os.path.join(out_input_path, "wind_cell.pkl")
    path_solar_cell = os.path.join(out_input_path, "solar_cell.pkl")

    # Read province name list
    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins = sorted(provins_name['provin_py'])

    with open(os.path.join(out_input_path, 'scen_params.json')) as fin:
        scen_params = json.load(fin)
    fin.close()

    wind_year = VreYearSplit(vre_year)[0]
    solar_year = VreYearSplit(vre_year)[1]

    if scen_params['vre']['aggregated'] == 0:
        with open(path_wind_cell, 'rb') as fin:
            wind_cell = pickle.load(fin)
        fin.close()

        with open(path_solar_cell, 'rb') as fin:
            solar_cell = pickle.load(fin)
        fin.close()

        # Initialize wind/solar info csv files
        fres_wind_info = open(os.path.join(out_output_processed_path, 'wind_info.csv'), 'w+')
        fres_solar_info = open(os.path.join(out_output_processed_path, 'solar_info.csv'), 'w+')

    elif scen_params['vre']['aggregated'] == 1:
        with open(work_dir + 'data_pkl' + dir_flag + 'aggregated_cell.pkl', 'rb') as fin:
            aggregated_cell = pickle.load(fin)
        fin.close()

        wind_cell = aggregated_cell['wind']
        solar_cell = aggregated_cell['solar']

        # Initialize wind/solar info csv files
        fres_wind_info = open(os.path.join(out_output_processed_path, 'aggregated_wind_info.csv'), 'w+')
        fres_solar_info = open(os.path.join(out_output_processed_path, 'aggregated_solar_info.csv'), 'w+')

    # Read wind/solar cell outputs from the run
    wind_count = 0
    solar_count = 0
    offshore_count = 0
    dpv_count = 0
    pro_count = {}
    for pro in provins:
        fx_wind = open(os.path.join(out_output_path, 'x_wind', pro + '.csv'), 'r+')
        pro_count[pro] = {'wind': 0, 'solar': 0, 'offshore': 0, 'dpv': 0}
        for x in fx_wind:
            x = x.replace('\n', '')
            x = x.split(',')
            x[0] = eval(x[0])
            x[1] = round(eval(x[1]), 4)
            inted = round(wind_cell['provin_cf_sort'][pro][x[0]][13], 4)

            if wind_cell['provin_cf_sort'][pro][x[0]][2] == 0:
                offshore_count += x[1] * wind_cell['provin_cf_sort'][pro][x[0]][3]
                pro_count[pro]['offshore'] += x[1] * wind_cell['provin_cf_sort'][pro][x[0]][3]
            else:
                wind_count += x[1] * wind_cell['provin_cf_sort'][pro][x[0]][3]
                pro_count[pro]['wind'] += x[1] * wind_cell['provin_cf_sort'][pro][x[0]][3]

            fres_wind_info.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' %
                                 (x[0],
                                  x[1],
                                  wind_cell['provin_cf_sort'][pro][x[0]][7],
                                  wind_cell['provin_cf_sort'][pro][x[0]][8],
                                  wind_cell['provin_cf_sort'][pro][x[0]][3],
                                  wind_cell['provin_cf_sort'][pro][x[0]][1],
                                  wind_cell['provin_cf_sort'][pro][x[0]][14],
                                  wind_cell['provin_cf_sort'][pro][x[0]][9],
                                  pro,
                                  wind_cell['provin_cf_sort'][pro][x[0]][2],
                                  inted))
        fx_wind.close()

        fx_solar = open(os.path.join(out_output_path, 'x_solar', pro + '.csv'), 'r+')
        for x in fx_solar:
            x = x.replace('\n', '')
            x = x.split(',')
            x[0] = eval(x[0])
            x[1] = round(eval(x[1]), 4)
            inted = round(solar_cell['provin_cf_sort'][pro][x[0]][13], 4)

            if solar_cell['provin_cf_sort'][pro][x[0]][2] == 0:
                solar_count += x[1] * solar_cell['provin_cf_sort'][pro][x[0]][3]
                pro_count[pro]['solar'] += x[1] * solar_cell['provin_cf_sort'][pro][x[0]][3]
            else:
                dpv_count += x[1] * solar_cell['provin_cf_sort'][pro][x[0]][3]
                pro_count[pro]['dpv'] += x[1] * solar_cell['provin_cf_sort'][pro][x[0]][3]

            fres_solar_info.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' %
                                  (x[0],
                                   x[1],  # installed + new cpacity (% of cell)
                                   solar_cell['provin_cf_sort'][pro][x[0]][7],  # lon
                                   solar_cell['provin_cf_sort'][pro][x[0]][8],  # lat
                                   solar_cell['provin_cf_sort'][pro][x[0]][3],  # CAP max potential in GW
                                   solar_cell['provin_cf_sort'][pro][x[0]][1],  # cf
                                   solar_cell['provin_cf_sort'][pro][x[0]][5],  # gen cost
                                   solar_cell['provin_cf_sort'][pro][x[0]][9],  # spur cost
                                   pro,
                                   solar_cell['provin_cf_sort'][pro][x[0]][2],
                                   inted))  # installed capacity (% of cell)
        fx_solar.close()
    fres_wind_info.close()
    fres_solar_info.close()

    # Save wind/solar processed outputs
    f_tot_cap = open(os.path.join(out_output_processed_path, 'ws_capacity.csv'), 'w+')
    f_tot_cap.write('technology,total_gw\n')
    f_tot_cap.write('%s,%s\n' % ('onshore_wind_gw', wind_count))
    f_tot_cap.write('%s,%s\n' % ('offshore_wind_gw', offshore_count))
    f_tot_cap.write('%s,%s\n' % ('utility_solar_gw', solar_count))
    f_tot_cap.write('%s,%s' % ('distributed_solar_gw', dpv_count))
    f_tot_cap.close()

    f_pro_cap = open(os.path.join(out_output_processed_path, 'ws_capacity_pro.csv'), 'w+')
    f_pro_cap.write('province,onshore_wind_gw,offshore_wind_gw,utility_solar_gw,distributed_solar_gw\n')
    for pro in sorted(provins):
        f_pro_cap.write('%s,%s,%s,%s,%s\n' % (
            pro, pro_count[pro]['wind'], pro_count[pro]['offshore'], pro_count[pro]['solar'], pro_count[pro]['dpv']))
    f_pro_cap.close()

    # Create integrated_solar/wind csv files for the next decade
    gw_to_mw_scale = 1000

    solar_info_df = pd.read_csv(os.path.join(out_output_processed_path, 'solar_info.csv'), header=None)
    integrated_solar = pd.DataFrame()
    integrated_solar["longitude"] = solar_info_df.iloc[:, 2]
    integrated_solar["latitude"] = solar_info_df.iloc[:, 3]
    integrated_solar["existing_and_new_capacity_mw"] = \
        solar_info_df.iloc[:, 1] * solar_info_df.iloc[:, 4] * gw_to_mw_scale
    integrated_solar.to_csv(os.path.join(out_output_processed_path, f"integrated_solar_{curr_year}.csv"),
                            header=False, index=False)

    wind_info_df = pd.read_csv(os.path.join(out_output_processed_path, 'wind_info.csv'), header=None)
    integrated_wind = pd.DataFrame()
    integrated_wind["longitude"] = wind_info_df.iloc[:, 2]
    integrated_wind["latitude"] = wind_info_df.iloc[:, 3]
    integrated_wind["existing_and_new_capacity_mw"] = \
        wind_info_df.iloc[:, 1] * wind_info_df.iloc[:, 4] * gw_to_mw_scale
    integrated_wind.to_csv(os.path.join(out_output_processed_path, f"integrated_wind_{curr_year}.csv"),
                           header=False, index=False)


def TransInfo(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    # Specify magic numbers
    unit_scale = 0.001

    # Obtain secondary parameters
    wind_year = VreYearSplit(vre_year)[0]
    solar_year = VreYearSplit(vre_year)[1]
    wind_years = SplitMultipleVreYear(wind_year)
    solar_years = SplitMultipleVreYear(solar_year)
    year_count = len(wind_years)

    # Read province name list
    provins_name = pd.read_csv(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    provins_name = provins_name.sort_values(by="provin_py").reset_index(drop=True)
    provins = sorted(provins_name['provin_py'])
    pro_hz = {}
    for i in range(len(provins)):
        pro_hz[provins[i]] = provins_name['provin'][i]

    # Obtain the energy transferred across pairs of provinces
    trans_tot = {}
    trans_tot_hourly = {}
    folder = makeDir(os.path.join(out_output_processed_path, 'trans_matrix'))
    ftrans_tot = open(os.path.join(folder, 'trans_matrix_tot.csv'), 'w+')
    ftrans_tot_bi = open(os.path.join(folder, 'trans_matrix_tot_bi.csv'), 'w+')

    for root, dirs, files in os.walk(os.path.join(out_output_path, 'load_trans')):
        for file in files:
            ftrans = open(root + dir_flag + file, 'r+')
            trans_pair = file.replace('.csv', '')
            trans_pair = trans_pair.split('_')
            trans_tot_hourly[(trans_pair[0], trans_pair[1])] = [0] * 8760 * year_count

            for line in ftrans:
                line = line.replace('\n', '')
                line = eval(line)
                trans_tot_hourly[(trans_pair[0], trans_pair[1])][line[0]] += line[1]
            ftrans.close()

    for pro_pair in trans_tot_hourly:
        trans_tot[pro_pair] = unit_scale * sum(trans_tot_hourly[pro_pair][:8760])

    # Save processed outputs
    with open(os.path.join(out_output_processed_path, 'trans_tot_hourly.pkl'), 'wb+') as fout:
        pickle.dump(trans_tot_hourly, fout)
    fout.close()

    # Process transmission energy data again
    ftrans_tot.write('TWh,')
    ftrans_tot_bi.write('TWh,')

    for pro in provins:
        ftrans_tot.write('%s,' % pro_hz[pro])
        ftrans_tot_bi.write('%s,' % pro_hz[pro])

    ftrans_tot.write('\n')
    ftrans_tot_bi.write('\n')

    net_trans_tot = 0
    bi_trans_tot = 0

    for pro1 in provins:
        ftrans_tot.write('%s,' % pro_hz[pro1])
        ftrans_tot_bi.write('%s,' % pro_hz[pro1])
        for pro2 in provins:
            if (pro1, pro2) in trans_tot.keys():
                if trans_tot[(pro1, pro2)] > trans_tot[(pro2, pro1)]:
                    tot = trans_tot[(pro1, pro2)] - trans_tot[(pro2, pro1)]
                else:
                    tot = 0

                ftrans_tot.write('%s,' % tot)
                net_trans_tot += tot
                ftrans_tot_bi.write('%s,' % trans_tot[(pro1, pro2)])
                bi_trans_tot += trans_tot[(pro1, pro2)]
            else:
                ftrans_tot.write('%s,' % 0)
                ftrans_tot_bi.write('%s,' % 0)

        ftrans_tot.write('\n')
        ftrans_tot_bi.write('\n')

    ftrans_tot.close()
    ftrans_tot_bi.close()

    # Save processed outputs
    ftrans_count = open(os.path.join(out_output_processed_path, 'trans_count.csv'), 'w+')
    ftrans_count.write('net_tot,' + str(net_trans_tot) + '\n')
    ftrans_count.write('bi_tot,' + str(bi_trans_tot) + '\n')


def TransCap(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    path_inter_pro_trans = os.path.join(out_input_path, "inter_pro_trans.csv")

    # Obtain new transmission line capacity
    newTransCap = {}
    f_ntc = open(os.path.join(out_output_path, 'new_trans_cap', 'cap_trans_new.csv'), 'r+')
    for line in f_ntc:
        line = line.replace('\n', '')
        line = line.split(',')
        line[2] = eval(line[2])
        newTransCap[(line[0], line[1])] = round(line[2], 2)
    f_ntc.close()

    # Obtain the existing transmission line capacity matrix
    existingTransCap = {}
    f_etc = pd.read_csv(path_inter_pro_trans)
    for pro in f_etc['province']:
        for i in range(len(f_etc[pro])):
            existingTransCap[(pro, f_etc['province'][i])] = 0.5 * 0.001 * f_etc[pro][i]  # MW -> GW

    # Obtain total transmission line capacity
    totalTransCap = {}
    for pro_pair in newTransCap:
        totalTransCap[pro_pair] = newTransCap[pro_pair] + existingTransCap[pro_pair]

    # Save processed outputs
    ftrans_cap = open(os.path.join(out_output_processed_path, 'trans_cap.csv'), 'w+')
    ftrans_cap.write('existing cap,' + str(sum(existingTransCap.values())) + '\n')
    ftrans_cap.write('new cap,' + str(sum(newTransCap.values())) + '\n')
    ftrans_cap.write('tot cap,' + str(sum(totalTransCap.values())) + '\n')

    # Create inter_pro_trans file for the next run
    gw_to_mw_scale = 1000
    inter_pro_trans_df = f_etc.copy(deep=True)
    f_ntc = open(os.path.join(out_output_path, 'new_trans_cap', 'cap_trans_new.csv'), 'r+')
    for line in f_ntc:
        # Read new capacity
        line = line.replace('\n', '')
        line = line.split(',')
        province_1 = str(line[0])
        province_2 = str(line[1])
        new_capacity_mw = round(float(line[2]), 2) * gw_to_mw_scale

        # Add new capacity to the existing capacity matrix for both directions
        inter_pro_trans_df.loc[inter_pro_trans_df["province"] == province_1, province_2] += new_capacity_mw
        inter_pro_trans_df.loc[inter_pro_trans_df["province"] == province_2, province_1] += new_capacity_mw
    f_ntc.close()
    inter_pro_trans_df.to_csv(os.path.join(out_output_processed_path, f"inter_pro_trans_{curr_year}.csv"),
                              header=True, index=False)


def update_storage_capacity(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    # Read phs capacity in 2060 as the reference point for formats
    phs_potential_2060 = pd.read_csv(os.path.join(work_dir, 'data_csv', 'capacity_assumptions', 'phs_ub.csv'),
                                     header=None)
    phs_potential_2060.columns = ["province", "phs_capacity_mw"]

    # Initialize the result dataframe
    integrated_storage_df = phs_potential_2060[["province"]].copy(deep=True)
    gw_to_mw_scale = 1000

    # Read storage capacity from the optimization output
    phs_df = pd.read_csv(os.path.join(out_output_path, "es_cap", "es_phs_cap.csv"), header=None)
    phs_df.columns = ["province", "phs_total_capacity_gw"]
    phs_df["phs_total_capacity_mw"] = phs_df["phs_total_capacity_gw"].round(2) * gw_to_mw_scale
    phs_df = phs_df[["province", "phs_total_capacity_mw"]]
    integrated_storage_df = pd.merge(integrated_storage_df, phs_df, left_on="province", right_on="province", how="left")

    bat_df = pd.read_csv(os.path.join(out_output_path, "es_cap", "es_bat_cap.csv"), header=None)
    bat_df.columns = ["province", "bat_total_capacity_gw"]
    bat_df["bat_total_capacity_mw"] = bat_df["bat_total_capacity_gw"].round(2) * gw_to_mw_scale
    bat_df = bat_df[["province", "bat_total_capacity_mw"]]
    integrated_storage_df = pd.merge(integrated_storage_df, bat_df, left_on="province", right_on="province", how="left")

    if os.path.exists(os.path.join(out_output_path, "es_cap", "es_caes_cap.csv")):
        caes_df = pd.read_csv(os.path.join(out_output_path, "es_cap", "es_caes_cap.csv"), header=None)
        caes_df.columns = ["province", "caes_total_capacity_gw"]
        caes_df["caes_total_capacity_mw"] = caes_df["caes_total_capacity_gw"].round(2) * gw_to_mw_scale
        caes_df = caes_df[["province", "caes_total_capacity_mw"]]
        integrated_storage_df = pd.merge(integrated_storage_df, caes_df, left_on="province", right_on="province",
                                         how="left")
    else:
        integrated_storage_df["caes_total_capacity_mw"] = 0

    if os.path.exists(os.path.join(out_output_path, "es_cap", "es_vrb_cap.csv")):
        vrb_df = pd.read_csv(os.path.join(out_output_path, "es_cap", "es_vrb_cap.csv"), header=None)
        vrb_df.columns = ["province", "vrb_total_capacity_gw"]
        vrb_df["vrb_total_capacity_mw"] = vrb_df["vrb_total_capacity_gw"].round(2) * gw_to_mw_scale
        vrb_df = vrb_df[["province", "vrb_total_capacity_mw"]]
        integrated_storage_df = pd.merge(integrated_storage_df, vrb_df, left_on="province", right_on="province",
                                         how="left")
    else:
        integrated_storage_df["vrb_total_capacity_mw"] = 0

    # Write output
    integrated_storage_df = integrated_storage_df.sort_values(by="province").reset_index(drop=True)
    integrated_storage_df.to_csv(os.path.join(out_output_processed_path, f"integrated_storage_{curr_year}.csv"),
                                 header=True, index=False)


def obtain_output_summary(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    # Read output files
    pro_demand = pd.read_pickle(os.path.join(out_input_path, "province_demand_full_" + str(curr_year) + ".pkl"))
    firm_df = pd.read_csv(os.path.join(out_output_path, "emissionBreakdowns.csv"))
    vre_df = pd.read_csv(os.path.join(out_output_processed_path, "ws_capacity_pro.csv"))
    str_df = pd.read_csv(os.path.join(out_output_processed_path, f"integrated_storage_{curr_year}.csv"))
    for str_tech in ["phs", "bat", "caes", "vrb"]:
        str_df[f"cap_{str_tech}_gw"] = str_df[f"{str_tech}_total_capacity_mw"] / 1e3
    tra_df = pd.read_csv(os.path.join(out_output_processed_path, "trans_cap.csv"), header=None)

    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    if not os.path.exists(os.path.join(out_output_processed_path, "graphs")):
        os.makedirs(os.path.join(out_output_processed_path, "graphs"))

    # Obtain land use data
    cell_if = {'solar': {}, 'wind': {}}
    for re in ['solar', 'wind']:
        f_re_info = open(os.path.join(out_output_processed_path, re + '_info.csv'), 'r+')
        for line in f_re_info:
            line = line.replace('\n', '')
            line = line.split(',')
            if line[8] not in cell_if[re]:
                cell_if[re][line[8]] = []
            cell_if[re][line[8]].append(eval(line[1]))
        f_re_info.close()
    pro_if = {'solar': {}, 'wind': {}}
    for re in ['solar', 'wind']:
        for pro in cell_if[re]:
            pro_if[re][pro] = sum(cell_if[re][pro]) / len(cell_if[re][pro])
    pro_if['wind']['Xizang'] = 0
    pd.DataFrame.from_dict(pro_if["solar"], orient='index').to_csv(
        os.path.join(out_output_processed_path, f"graphs/solar_IF.csv"), header=False)
    pd.DataFrame.from_dict(pro_if["wind"], orient='index').to_csv(
        os.path.join(out_output_processed_path, f"graphs/wind_IF.csv"), header=False)
    solar_share_df = pd.read_csv(os.path.join(out_output_processed_path, "graphs", "solar_IF.csv"), header=None)
    solar_share_df.columns = ["province", "solar_land_use_share"]
    wind_share_df = pd.read_csv(os.path.join(out_output_processed_path, "graphs", "wind_IF.csv"), header=None)
    wind_share_df.columns = ["province", "wind_land_use_share"]

    # Provincial statistics
    out_df = pd.DataFrame()
    out_df["province"] = vre_df["province"]
    out_df["demand_twh"] = 0
    for i in range(len(out_df["province"])):
        pro = out_df["province"].iloc[i]
        out_df.loc[i, "demand_twh"] = np.sum(pro_demand[pro]) / 1e3
    out_df = pd.merge(out_df, vre_df, left_on="province", right_on="province", how="left")
    out_df = pd.merge(out_df, str_df[["province", "cap_phs_gw", "cap_bat_gw", "cap_caes_gw", "cap_vrb_gw"]],
                      left_on="province", right_on="province", how="left")
    out_df = pd.merge(out_df, solar_share_df, left_on="province", right_on="province", how="left")
    out_df = pd.merge(out_df, wind_share_df, left_on="province", right_on="province", how="left")

    # National statistics
    out_df_national = pd.DataFrame(columns=["item", "value"])
    out_df_national.loc[len(out_df_national.index)] = ["demand_pwh", np.sum(out_df["demand_twh"]) / 1e3]
    out_df_national.loc[len(out_df_national.index)] = ["--------", " "]
    out_df_national.loc[len(out_df_national.index)] = ["coal_unabated_pwh", np.sum(firm_df["electricity_coal_unabated"]
                                                                                   ) / 1e6]
    out_df_national.loc[len(out_df_national.index)] = ["coal_ccs_pwh", np.sum(firm_df["electricity_coal_ccs"]) / 1e6]
    out_df_national.loc[len(out_df_national.index)] = ["chp_ccs_pwh", np.sum(firm_df["electricity_chp_ccs"]) / 1e6]
    out_df_national.loc[len(out_df_national.index)] = ["gas_unabated_pwh", np.sum(firm_df["electricity_gas_unabated"]
                                                                                  ) / 1e6]
    out_df_national.loc[len(out_df_national.index)] = ["gas_ccs_pwh", np.sum(firm_df["electricity_gas_ccs"]) / 1e6]
    out_df_national.loc[len(out_df_national.index)] = ["--------", " "]
    out_df_national.loc[len(out_df_national.index)] = ["coal_unabated_gw", np.sum(firm_df["cap_coal_unabated_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["coal_ccs_gw", np.sum(firm_df["cap_coal_ccs_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["chp_ccs_gw", np.sum(firm_df["cap_chp_ccs_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["gas_unabated_gw", np.sum(firm_df["cap_gas_unabated_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["gas_ccs_gw", np.sum(firm_df["cap_gas_ccs_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["beccs_gw", np.sum(firm_df["cap_beccs_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["bio_gw", np.sum(firm_df["cap_bio_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["hydro_gw", np.sum(firm_df["cap_hydro_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["nuclear_gw", np.sum(firm_df["cap_nuclear_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["--------", " "]
    out_df_national.loc[len(out_df_national.index)] = ["onshore_gw", np.sum(out_df["onshore_wind_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["offshore_gw", np.sum(out_df["offshore_wind_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["utility_gw", np.sum(out_df["utility_solar_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["distributed_gw", np.sum(out_df["distributed_solar_gw"])]
    for str_tech in ["phs", "bat", "caes", "vrb"]:
        out_df_national.loc[len(out_df_national.index)] = [f"cap_{str_tech}_gw", np.sum(out_df[f"cap_{str_tech}_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["--------", " "]
    out_df_national.loc[len(out_df_national.index)] = ["old_trans_cap", tra_df.iloc[0, 1]]
    out_df_national.loc[len(out_df_national.index)] = ["add_trans_cap", tra_df.iloc[1, 1]]
    out_df_national.loc[len(out_df_national.index)] = ["tot_trans_cap", tra_df.iloc[2, 1]]
    out_df_national.loc[len(out_df_national.index)] = ["--------", " "]
    out_df_national.loc[len(out_df_national.index)] = ["fossil_share", np.sum(firm_df["electricity_coal_unabated"] +
                                                                            firm_df["electricity_coal_ccs"] +
                                                                            firm_df["electricity_chp_ccs"] +
                                                                            firm_df["electricity_gas_unabated"] +
                                                                            firm_df["electricity_gas_ccs"]) / 1e3 /
                                                       np.sum(out_df["demand_twh"])]
    out_df_national.loc[len(out_df_national.index)] = ["renewable_gw", np.sum(out_df["onshore_wind_gw"] +
                                                                              out_df["offshore_wind_gw"] +
                                                                              out_df["utility_solar_gw"] +
                                                                              out_df["distributed_solar_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["storage_gw", np.sum(out_df["cap_phs_gw"] +
                                                                            out_df["cap_bat_gw"] +
                                                                            out_df["cap_caes_gw"] +
                                                                            out_df["cap_vrb_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["solar_gw", np.sum(out_df["utility_solar_gw"] +
                                                                          out_df["distributed_solar_gw"])]
    out_df_national.loc[len(out_df_national.index)] = ["wind_gw", np.sum(out_df["onshore_wind_gw"] +
                                                                         out_df["offshore_wind_gw"])]

    # Write output
    out_df.round(3).to_csv(
        os.path.join(out_output_processed_path, f"summary_provincial_{curr_year}.csv"),
        header=True, index=False)
    out_df_national.round(3).to_csv(
        os.path.join(out_output_processed_path, f"summary_national_{curr_year}.csv"),
        header=True, index=False)


def obtain_simulation_summary(vre_year, res_tag, year_list):
    # Specify file paths
    stat_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}")

    # Read 2020 initial VRE conditions
    solar_2020 = pd.read_csv(os.path.join(work_dir, "data_csv", "vre_installations", "province_solar_data.csv"),
                             header=None)
    solar_2020.columns = ["province", "solar_mw_2020"]
    solar_2020 = solar_2020.sort_values(by="province").reset_index(drop=True)
    wind_2020 = pd.read_csv(os.path.join(work_dir, "data_csv", "vre_installations", "province_wind_data.csv"),
                            header=None)
    wind_2020.columns = ["province", "wind_mw_2020"]
    wind_2020 = wind_2020.sort_values(by="province").reset_index(drop=True)

    # Initialize statistics dataframes
    stat_df = pd.DataFrame()
    for curr_year in year_list:
        # Specify file paths
        out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
        out_input_path = os.path.join(out_path, "inputs")
        out_output_path = os.path.join(out_path, "outputs")
        out_output_processed_path = os.path.join(out_path, "outputs_processed")

        # Read output files
        firm_df = pd.read_csv(os.path.join(out_output_path, "emissionBreakdowns.csv"))
        out_df = pd.read_csv(os.path.join(out_output_processed_path, f"summary_provincial_{curr_year}.csv"))

        # Add yearly output to statistics dataframes
        if "province" not in stat_df.columns:
            stat_df["province"] = firm_df["province"]
        if "solar_gw_2020" not in stat_df.columns:
            stat_df["solar_gw_2020"] = solar_2020["solar_mw_2020"] / 1e2
        if "wind_gw_2020" not in stat_df.columns:
            stat_df["wind_gw_2020"] = wind_2020["wind_mw_2020"] / 1e2
        for tech in ["coal_unabated", "coal_ccs", "chp_ccs", "gas_unabated", "gas_ccs"]:
            stat_df[f"{tech}_gw_{curr_year}"] = firm_df[f"cap_{tech}_gw"]
        stat_df[f"wind_gw_{curr_year}"] = out_df["onshore_wind_gw"] + out_df["offshore_wind_gw"]
        stat_df[f"solar_gw_{curr_year}"] = out_df["utility_solar_gw"] + out_df["distributed_solar_gw"]
        stat_df[f"land_use_wind_{curr_year}"] = out_df["wind_land_use_share"]
        stat_df[f"land_use_solar_{curr_year}"] = out_df["solar_land_use_share"]
        # TODO add storage

    # Calculate annual renewable buildout rate
    for tech in ["solar", "wind"]:
        if 2030 in year_list:
            stat_df[f"r_{tech}_2021_30"] = (stat_df[f"{tech}_gw_2030"] - stat_df[f"{tech}_gw_2020"]) / 10
        if 2040 in year_list:
            stat_df[f"r_{tech}_2031_40"] = (stat_df[f"{tech}_gw_2040"] - stat_df[f"{tech}_gw_2030"]) / 10
        if 2050 in year_list:
            stat_df[f"r_{tech}_2041_50"] = (stat_df[f"{tech}_gw_2050"] - stat_df[f"{tech}_gw_2040"] +
                                            stat_df[f"{tech}_gw_2030"] / 2) / 10
        if 2060 in year_list:
            stat_df[f"r_{tech}_2051_60"] = (stat_df[f"{tech}_gw_2060"] - stat_df[f"{tech}_gw_2050"] +
                                            stat_df[f"{tech}_gw_2040"] / 2) / 10

    # Add national summary to provincial statistics
    stat_df.loc[len(stat_df.index)] = "-"
    for col in stat_df.columns:
        if col == "province":
            stat_df.loc[-1, col] = "National"
        elif "land_use" in col:
            stat_df.loc[-1, col] = np.mean(stat_df[col][0:-2])
        else:
            stat_df.loc[-1, col] = np.sum(stat_df[col][0:-2])

    # Initialize national statistics
    stat_national_df = pd.DataFrame()
    for curr_year in year_list:
        # Specify file paths
        out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
        out_output_processed_path = os.path.join(out_path, "outputs_processed")

        # Read output files
        out_df_national = pd.read_csv(os.path.join(out_output_processed_path, f"summary_national_{curr_year}.csv"))

        # Save results to national statistics
        if len(stat_national_df.columns) == 0:
            stat_national_df["item"] = out_df_national["item"]
        stat_national_df[f"{curr_year}"] = out_df_national["value"]

    # Write output
    stat_df = stat_df.set_index("province")
    stat_df = stat_df[sorted(stat_df.columns)]
    stat_df = stat_df.reset_index(drop=False)
    stat_df.round(decimals=3).to_csv(
        os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", "stat_provincial.csv"), header=True, index=False)
    stat_national_df.round(decimals=3).to_csv(
        os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", "stat_national.csv"), header=True, index=False)


def LoadProfile(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    # Specify parameters
    wind_year = VreYearSplit(vre_year)[0]
    solar_year = VreYearSplit(vre_year)[1]
    wind_years = SplitMultipleVreYear(wind_year)
    solar_years = SplitMultipleVreYear(solar_year)
    year_count = len(wind_years)

    # Read province name list
    provins = []
    f_provins = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')
    next(f_provins)
    for line in f_provins:
        line = line.replace('\n', '')
        line = line.split(',')
        provins.append(line[1])
    f_provins.close()

    # Read initial scenario parameters
    with open(os.path.join(out_input_path, "scen_params.json")) as fin:
        scen_params = json.load(fin)
    fin.close()

    # Read firm resource parameters
    nuclearBeta = scen_params["nuclear"]["beta"]
    nuclearMR = scen_params["nuclear"]["must_run"]

    fhour_seed = open(os.path.join(out_input_path, "hour_seed.csv"), 'r+')
    hour_seed = []
    for line in fhour_seed:
        line = line.replace('\n', '')
        line = eval(line)
        hour_seed.append(line)
    fhour_seed.close()

    fwinter_hour = open(work_dir + 'data_csv' + dir_flag + 'simulation_meta/winter_hour.csv', 'r+')
    winter_hour = []
    for line in fwinter_hour:
        line = line.replace('\n', '')
        line = eval(line)
        winter_hour.append(line)
    fwinter_hour.close()

    # Read firm resource capacities
    chp_ccs = {}
    chp_ccs_df = pd.read_csv(os.path.join(out_path, "chp_ccs.csv"))
    for idx in range(len(provins)):
        province = chp_ccs_df["province"].iloc[idx]
        chp_ccs[province] = 0.001 * chp_ccs_df[f"{curr_year}_cap_mw"].iloc[idx]

    nuclear = {}
    nuclear_df = pd.read_csv(os.path.join(out_path, "nuclear.csv"))
    for idx in range(len(provins)):
        province = nuclear_df["province"].iloc[idx]
        nuclear[province] = np.ones(len(hour_seed)) * \
                            nuclearBeta * nuclearMR * nuclear_df[f"{curr_year}_cap_mw"].iloc[idx] * 0.001

    beccs_cap = {}
    beccs_df = pd.read_csv(os.path.join(out_path, "beccs.csv"))
    for idx in range(len(provins)):
        province = beccs_df["province"].iloc[idx]
        beccs_cap[province] = beccs_df[f"{curr_year}_cap_gw"].iloc[idx]

    # Read operation results
    l1 = {}
    l2 = {'Nationwide': np.zeros(len(hour_seed))}
    l3 = {'Nationwide': np.zeros(len(hour_seed))}
    l4 = {'Nationwide': np.zeros(len(hour_seed))}

    beccs = {'Nationwide': np.zeros(len(hour_seed))}
    coal = {'Nationwide': np.zeros(len(hour_seed))}
    wind = {'Nationwide': np.zeros(len(hour_seed))}
    solar = {'Nationwide': np.zeros(len(hour_seed))}

    # transfer out
    trans_out = {'l1': {}, 'l2': {}, 'l3': {}, 'l4': {}, 'wind': {}, 'solar': {}}

    for et in ['l1', 'l2', 'l3', 'wind', 'solar', 'l4']:
        for pro in provins:
            f_to = open(os.path.join(out_output_path, 'trans_out', et, pro + '.csv'), 'r+')
            if pro not in trans_out[et]:
                trans_out[et][pro] = [0] * 8760 * year_count
            for line in f_to:
                line = line.replace('\n', '')
                line = eval(line)
                trans_out[et][pro][line[0]] = line[1]
            f_to.close()

    trans_out['Nationwide'] = {
        'l1': np.zeros(len(hour_seed)),
        'l2': np.zeros(len(hour_seed)),
        'l3': np.zeros(len(hour_seed)),
        'l4': np.zeros(len(hour_seed)),
        'wind': np.zeros(len(hour_seed)),
        'solar': np.zeros(len(hour_seed))}

    for pro in provins:
        for et in ['l1', 'l2', 'l3', 'l4', 'wind', 'solar']:
            trans_out[et][pro] = np.array(trans_out[et][pro])

    for et in ['l1', 'l2', 'l3', 'l4', 'wind', 'solar']:
        for pro in provins:
            trans_out['Nationwide'][et] = trans_out['Nationwide'][et] + trans_out[et][pro]

            # integration to grid
    for pro in provins:
        fl1 = open(os.path.join(out_output_path, 'load_conv', 'l1', pro + '.csv'), 'r+')
        if pro not in l1:
            l1[pro] = []
        for line in fl1:
            line = line.replace('\n', '')
            line = eval(line)
            if line[0] in hour_seed:
                l1[pro].append(line[1])
        fl1.close()

        fl2 = open(os.path.join(out_output_path, 'load_conv', 'l2', pro + '.csv'), 'r+')
        if pro not in l2.keys():
            l2[pro] = []
        for line in fl2:
            line = line.replace('\n', '')
            line = eval(line)
            if line[0] in hour_seed:
                l2[pro].append(line[1])
        fl2.close()

        fl3 = open(os.path.join(out_output_path, 'load_conv', 'l3', pro + '.csv'), 'r+')
        if pro not in l3.keys():
            l3[pro] = []
        for line in fl3:
            line = line.replace('\n', '')
            line = eval(line)
            if line[0] in hour_seed:
                l3[pro].append(line[1])
        fl3.close()

        fl4 = open(os.path.join(out_output_path, 'load_conv', 'l4', pro + '.csv'), 'r+')
        if pro not in l4.keys():
            l4[pro] = []
        for line in fl4:
            line = line.replace('\n', '')
            line = eval(line)
            if line[0] in hour_seed:
                l4[pro].append(line[1])
        fl4.close()

        fwind = open(os.path.join(out_output_path, 'inte_wind', pro + '.csv'), 'r+')
        if pro not in wind.keys():
            wind[pro] = []
        for line in fwind:
            line = line.replace('\n', '')
            line = eval(line)
            if line[0] in hour_seed:
                wind[pro].append(line[1])
        fwind.close()

        fsolar = open(os.path.join(out_output_path, 'inte_solar', pro + '.csv'), 'r+')
        if pro not in solar.keys():
            solar[pro] = []
        for line in fsolar:
            line = line.replace('\n', '')
            line = eval(line)
            if line[0] in hour_seed:
                solar[pro].append(line[1])
        fsolar.close()

    # charge
    charge_l2 = {'phs': {}, 'bat': {}, 'Nationwide': np.zeros(len(hour_seed))}
    charge_l3 = {'phs': {}, 'bat': {}, 'Nationwide': np.zeros(len(hour_seed))}
    charge_wind = {'phs': {}, 'bat': {}, 'Nationwide': np.zeros(len(hour_seed))}
    charge_solar = {'phs': {}, 'bat': {}, 'Nationwide': np.zeros(len(hour_seed))}

    for pro in provins:
        for st in ['phs', 'bat']:
            f_charge_l2 = open(os.path.join(out_output_path, 'es_char', st, 'l2', pro + '.csv'), 'r+')
            if pro not in charge_l2[st]:
                charge_l2[st][pro] = []
            for line in f_charge_l2:
                line = line.replace('\n', '')
                line = eval(line)
                if line[0] in hour_seed:
                    charge_l2[st][pro].append(line[1])
            f_charge_l2.close()

            f_charge_l3 = open(os.path.join(out_output_path, 'es_char', st, 'l3', pro + '.csv'), 'r+')
            if pro not in charge_l3[st]:
                charge_l3[st][pro] = []
            for line in f_charge_l3:
                line = line.replace('\n', '')
                line = eval(line)
                if line[0] in hour_seed:
                    charge_l3[st][pro].append(line[1])
            f_charge_l3.close()

            f_charge_wind = open(os.path.join(out_output_path, 'es_char', st, 'wind', pro + '.csv'), 'r+')
            if pro not in charge_wind[st]:
                charge_wind[st][pro] = []
            for line in f_charge_wind:
                line = line.replace('\n', '')
                line = eval(line)
                if line[0] in hour_seed:
                    charge_wind[st][pro].append(line[1])
            f_charge_wind.close()

            f_charge_solar = open(os.path.join(out_output_path, 'es_char', st, 'solar', pro + '.csv'), 'r+')
            if pro not in charge_solar[st]:
                charge_solar[st][pro] = []
            for line in f_charge_solar:
                line = line.replace('\n', '')
                line = eval(line)
                if line[0] in hour_seed:
                    charge_solar[st][pro].append(line[1])
            f_charge_solar.close()

    # discharge
    discharge = {'phs': {}, 'bat': {}, 'Nationwide': np.zeros(len(hour_seed))}
    for pro in provins:
        for st in ['phs', 'bat']:
            if pro not in discharge[st]:
                discharge[st][pro] = []

            f_dis = open(os.path.join(out_output_path, 'es_inte', st, pro + '.csv'), 'r+')
            for line in f_dis:
                line = line.replace('\n', '')
                line = eval(line)
                if line[0] in hour_seed:
                    discharge[st][pro].append(line[1])
            f_dis.close()

    # must run generator
    for pro in provins:
        if pro not in coal:
            # coal[pro] = trans_out['l1'][pro]  # TODO Remove transfer out from l1
            coal[pro] = np.zeros(len(hour_seed))

        if pro not in beccs:
            beccs[pro] = np.zeros(len(hour_seed))

        for h in range(len(l1[pro])):
            coal[pro][h] += l1[pro][h]
            beccs[pro][h] = beccs_cap[pro] * scen_params['ccs']['beccs_cf']

            if h in winter_hour:
                coal[pro][h] += chp_ccs[pro] / (1 - scen_params['ccs']['coal_loss'])  # TODO apply efficiency losses

    # count the nationwide data
    for pro in provins:
        l2[pro] = np.array(l2[pro])

        l3[pro] = np.array(l3[pro])
        l3[pro] = l3[pro] + nuclear[pro]

        # l4[pro] = np.array(l4[pro]) + trans_out['l4'][pro]  # TODO Remove transfer out from l4
        l4[pro] = np.array(l4[pro])
        wind[pro] = np.array(wind[pro])
        solar[pro] = np.array(solar[pro])

        for st in ['phs', 'bat']:
            discharge[st][pro] = np.array(discharge[st][pro])
            charge_l2[st][pro] = np.array(charge_l2[st][pro])
            charge_l3[st][pro] = np.array(charge_l3[st][pro])
            charge_wind[st][pro] = np.array(charge_wind[st][pro])
            charge_solar[st][pro] = np.array(charge_solar[st][pro])

    for pro in provins:
        l2['Nationwide'] = l2['Nationwide'] + l2[pro]
        l3['Nationwide'] = l3['Nationwide'] + l3[pro]
        l4['Nationwide'] = l4['Nationwide'] + l4[pro]

        beccs['Nationwide'] = beccs['Nationwide'] + beccs[pro]

        coal['Nationwide'] = coal['Nationwide'] + coal[pro]

        wind['Nationwide'] = wind['Nationwide'] + wind[pro]
        solar['Nationwide'] = solar['Nationwide'] + solar[pro]

        for st in ['phs', 'bat']:
            discharge['Nationwide'] = discharge['Nationwide'] + discharge[st][pro]
            charge_l2['Nationwide'] = charge_l2['Nationwide'] + charge_l2[st][pro]
            charge_l3['Nationwide'] = charge_l3['Nationwide'] + charge_l3[st][pro]
            charge_wind['Nationwide'] = charge_wind['Nationwide'] + charge_wind[st][pro]
            charge_solar['Nationwide'] = charge_solar['Nationwide'] + charge_solar[st][pro]

    charge_tot = charge_l2['Nationwide'] + charge_l3['Nationwide'] + charge_wind['Nationwide'] + charge_solar[
        'Nationwide']

    save_as_pkl = {'hydro': l2,
                   'bio': beccs,
                   'gas': l4,
                   'coal': coal,
                   'nuclear': l3,
                   'wind': wind,
                   'solar': solar,
                   'discharge': discharge,
                   'trans_out': trans_out,
                   'charge_tot': charge_tot,
                   'charge_l2': charge_l2,
                   'charge_l3': charge_l3,
                   'charge_wind': charge_wind,
                   'charge_solar': charge_solar}

    with open(os.path.join(out_output_processed_path, 'load.pkl'), 'wb+') as fout:
        pickle.dump(save_as_pkl, fout)
    fout.close()


def curtailed(vre_year, res_tag, curr_year, re):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    # Specify parameters
    wind_year = VreYearSplit(vre_year)[0]
    solar_year = VreYearSplit(vre_year)[1]
    wind_years = SplitMultipleVreYear(wind_year)
    solar_years = SplitMultipleVreYear(solar_year)
    year_count = len(wind_years)

    # Read province name list
    provins = []
    f_provins = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')
    next(f_provins)
    for line in f_provins:
        line = line.replace('\n', '')
        line = line.split(',')
        provins.append(line[1])
    f_provins.close()

    # Read initial scenario parameters
    with open(os.path.join(out_input_path, "scen_params.json")) as fin:
        scen_params = json.load(fin)
    fin.close()

    if scen_params['vre']['aggregated'] == 0:
        wind_year = VreYearSplit(vre_year)[0]
        solar_year = VreYearSplit(vre_year)[1]

        cell_pkl = {'solar': 'solar_cell.pkl',
                    'wind': 'wind_cell.pkl'}
        with open(os.path.join(out_input_path, cell_pkl[re]), 'rb') as fin:
            cell = pickle.load(fin)
        fin.close()
    else:
        with open(work_dir + 'data_pkl' + dir_flag + 'aggregated_cell.pkl', 'rb+') as fin:
            cell = pickle.load(fin)
        fin.close()
        cell = cell[re]

    # Read seed hours of the optimization task
    fhour_seed = open(os.path.join(out_input_path, 'hour_seed.csv'), 'r+')
    hour_seed = []
    for line in fhour_seed:
        line = line.replace('\n', '')
        line = eval(line)
        hour_seed.append(line)
    fhour_seed.close()

    cell_res = {'solar': 'x_solar', 'wind': 'x_wind'}
    cell_cof = {}
    for pro in provins:
        if pro not in cell_cof:
            cell_cof[pro] = {}
        f_cell_res = open(os.path.join(out_output_path, cell_res[re], pro + '.csv'), 'r+')
        for line in f_cell_res:
            line = line.replace('\n', '')
            line = eval(line)
            cell_cof[pro][line[0]] = line[1]
        f_cell_res.close()

    power_gen = {}
    for pro in provins:
        if pro not in power_gen:
            power_gen[pro] = np.zeros(8760 * year_count)
        for c in cell_cof[pro]:
            for h in range(8760 * year_count):
                power_gen[pro][h] += (cell_cof[pro][c] * cell['provin_cf_sort'][pro][c][3] * cell['cf_prof'][pro][c][h])

    power_inUse_load, power_inUse_charge, power_inUse_trans = {}, {}, {}
    power_inUse = {}

    storage_tech = ['phs', 'bat']
    if scen_params['storage']['with_caes']:
        storage_tech.append('caes')
    if scen_params['storage']['with_vrb']:
        storage_tech.append('vrb')

    for pro in provins:
        if pro not in power_inUse:
            power_inUse[pro] = []
        if pro not in power_inUse_load:
            power_inUse_load[pro] = np.zeros(8760)
        if pro not in power_inUse_charge:
            power_inUse_charge[pro] = np.zeros(8760)
        if pro not in power_inUse_trans:
            power_inUse_trans[pro] = np.zeros(8760)

        f_re_inte = open(os.path.join(out_output_path, 'inte_' + re, pro + '.csv'), 'r+')
        for line in f_re_inte:
            line = line.replace('\n', '')
            line = eval(line)
            power_inUse[pro].append(line[1])
            power_inUse_load[pro][line[0]] += line[1]
        f_re_inte.close()

        for st in storage_tech:
            f_re_sto = open(os.path.join(out_output_path, 'es_char', st, re, pro + '.csv'), 'r+')
            for line in f_re_sto:
                line = line.replace('\n', '')
                line = eval(line)
                power_inUse[pro][line[0]] += line[1]
                power_inUse_charge[pro][line[0]] += line[1]
            f_re_sto.close()

    for root, dirs, files in os.walk(os.path.join(out_output_path, 'trans_out', re)):
        for file in files:
            ftrans = open(root + dir_flag + file, 'r+')
            pro = file.replace('.csv', '')
            for line in ftrans:
                line = line.replace('\n', '')
                line = eval(line)
                power_inUse[pro][line[0]] += line[1]
                power_inUse_trans[pro][line[0]] += line[1]
            ftrans.close()

    power_curt = {}
    power_curt_rate = {}
    nation_curt = 0
    nation_gen = 0

    for pro in provins:
        pro_curt = 0
        pro_gen = 0
        if pro not in power_curt:
            power_curt[pro] = []
        for h in range(8760 * year_count):
            power_curt[pro].append(power_gen[pro][h] - power_inUse[pro][h])

            if h in hour_seed:
                nation_curt += power_gen[pro][h] - power_inUse[pro][h]
                nation_gen += power_gen[pro][h]

                pro_curt += power_gen[pro][h] - power_inUse[pro][h]
                pro_gen += power_gen[pro][h]

        if re == 'wind':
            if pro != 'Xizang':
                power_curt_rate[pro] = pro_curt / pro_gen
            else:
                power_curt_rate[pro] = 0
        else:
            power_curt_rate[pro] = pro_curt / pro_gen

    # Save processed results
    f_nation_curt = open(os.path.join(out_output_processed_path, re + '_curt.csv'), 'w+')
    f_nation_curt.write('%s,%s,%s\n' % (nation_curt, nation_gen, 100 * (nation_curt / nation_gen)))
    f_nation_curt.close()

    folder = makeDir(os.path.join(out_output_processed_path, 'curtailment', re))
    save_as_pkl = {'power_gen': power_gen,
                   'power_inUse': power_inUse,
                   'power_inUse_load': power_inUse_load,
                   'power_inUse_charge': power_inUse_charge,
                   'power_inUse_trans': power_inUse_trans,
                   'power_curt': power_curt,
                   'power_curt_rate': power_curt_rate,
                   'nation_curt': nation_curt,
                   'nation_gen': nation_gen}
    with open(os.path.join(out_output_processed_path, "curtailment", 'curtailed_' + re + '.pkl'), 'wb+') as fout:
        pickle.dump(save_as_pkl, fout)
    fout.close()

    for pro in provins:
        f_curt = open(os.path.join(folder, pro + '.csv'), 'w+')
        for h in range(year_count * 8760):
            f_curt.write('%s,%s,%s,%s\n' % (h, power_gen[pro][h], power_inUse[pro][h], power_curt[pro][h]))
        f_curt.close()

    print(re,
          "national curtailment is", np.round(nation_curt, 2),
          "national generation is", np.round(nation_gen, 2),
          "curtailment rate is", np.round(nation_curt / nation_gen, 2))


def CurtailedSplitVRE(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")

    # Specify parameters
    wind_year = VreYearSplit(vre_year)[0]
    solar_year = VreYearSplit(vre_year)[1]
    wind_years = SplitMultipleVreYear(wind_year)
    solar_years = SplitMultipleVreYear(solar_year)
    year_count = len(wind_years)

    # Read province name list
    provins = []
    f_provins = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')
    next(f_provins)
    for line in f_provins:
        line = line.replace('\n', '')
        line = line.split(',')
        provins.append(line[1])
    f_provins.close()

    power_gen = {'solar': {'nation': 0}, 'wind': {'nation': 0}}
    power_curt = {'solar': {'nation': 0}, 'wind': {'nation': 0}}

    wind_curt = 0
    solar_curt = 0

    power_curt_rate = {'solar': {}, 'wind': {}}

    for pro in provins:
        curt_info = {
            'solar_gen': [],
            'solar_curt': [],
            'wind_gen': [],
            'wind_curt': []
        }

        curt_split = {
            'solar': [],
            'wind': []
        }

        f_sc = open(os.path.join(out_output_processed_path, 'curtailment', 'solar', pro + '.csv'), 'r+')
        for line in f_sc:
            line = line.replace('\n', '')
            line = eval(line)
            curt_info['solar_gen'].append(line[1])
            curt_info['solar_curt'].append(line[3])
        f_sc.close()

        f_wc = open(os.path.join(out_output_processed_path, 'curtailment', 'wind', pro + '.csv'), 'r+')
        for line in f_wc:
            line = line.replace('\n', '')
            line = eval(line)
            curt_info['wind_gen'].append(line[1])
            curt_info['wind_curt'].append(line[3])
        f_wc.close()

        solar_curt += sum(curt_info['solar_curt'])
        wind_curt += sum(curt_info['wind_curt'])

        for h in range(year_count * 8760):
            if curt_info['solar_curt'][h] > 0 and curt_info['wind_curt'][h] > 0:
                total_curt = curt_info['solar_curt'][h] + curt_info['wind_curt'][h]
                total_gen = curt_info['solar_gen'][h] + curt_info['wind_gen'][h]

                curt_split['solar'].append(total_curt * (curt_info['solar_gen'][h] / total_gen))
                curt_split['wind'].append(total_curt * (curt_info['wind_gen'][h] / total_gen))
            else:
                curt_split['solar'].append(curt_info['solar_curt'][h])
                curt_split['wind'].append(curt_info['wind_curt'][h])

        power_gen['solar'][pro] = sum(curt_info['solar_gen'])
        power_gen['solar']['nation'] += power_gen['solar'][pro]

        power_curt['solar'][pro] = sum(curt_split['solar'])
        power_curt['solar']['nation'] += power_curt['solar'][pro]

        power_gen['wind'][pro] = sum(curt_info['wind_gen'])
        power_gen['wind']['nation'] += power_gen['wind'][pro]

        power_curt['wind'][pro] = sum(curt_split['wind'])
        power_curt['wind']['nation'] += power_curt['wind'][pro]

        if power_gen['solar'][pro] != 0:
            power_curt_rate['solar'][pro] = 100 * power_curt['solar'][pro] / power_gen['solar'][pro]
        else:
            power_curt_rate['solar'][pro] = 0

        if power_gen['wind'][pro] != 0:
            power_curt_rate['wind'][pro] = 100 * power_curt['wind'][pro] / power_gen['wind'][pro]
        else:
            power_curt_rate['wind'][pro] = 0

    power_curt_rate['solar']['nation'] = 100 * power_curt['solar']['nation'] / power_gen['solar']['nation']
    power_curt_rate['wind']['nation'] = 100 * power_curt['wind']['nation'] / power_gen['wind']['nation']

    # Save processed outputs
    save_as_pkl = power_curt_rate['solar']
    with open(os.path.join(out_output_processed_path, 'curtailment', 'curtailed_solar_split.pkl'),
              'wb+') as fout:
        pickle.dump(save_as_pkl, fout)
    fout.close()

    save_as_pkl = power_curt_rate['wind']
    with open(os.path.join(out_output_processed_path, 'curtailment', 'curtailed_wind_split.pkl'),
              'wb+') as fout:
        pickle.dump(save_as_pkl, fout)
    fout.close()

    # Save wind/solar province-level curtailment outputs
    f_sc_pro = open(os.path.join(out_output_processed_path, 'curtailment', 'solar_curt.csv'), 'w+')
    for pro in power_curt_rate['solar']:
        f_sc_pro.write(pro + ',' + str(power_curt_rate['solar'][pro]) + '\n')
    f_sc_pro.close()

    f_wc_pro = open(os.path.join(out_output_processed_path, 'curtailment', 'wind_curt.csv'), 'w+')
    for pro in power_curt_rate['wind']:
        f_wc_pro.write(pro + ',' + str(power_curt_rate['wind'][pro]) + '\n')
    f_wc_pro.close()

    # Save wind/solar nation-level curtailment outputs
    f_sc_out = open(os.path.join(out_output_processed_path, 'solar_curt.csv'), 'a+')
    f_sc_out.write(
        '%s,%s,%s\n' % (
            power_curt['solar']['nation'], power_gen['solar']['nation'], power_curt_rate['solar']['nation']))
    f_sc_out.close()

    f_wc_out = open(os.path.join(out_output_processed_path, 'wind_curt.csv'), 'a+')
    f_wc_out.write(
        '%s,%s,%s\n' % (power_curt['wind']['nation'], power_gen['wind']['nation'], power_curt_rate['wind']['nation']))
    f_wc_out.close()


def reInteInfo(wind_year, solar_year, inte_target, re):
    provins = getProvinName()

    res_dir = res_dir = getResDir(wind_year, solar_year, inte_target) + dir_flag

    inte_res = {'solar': 'inte_solar', 'wind': 'inte_wind'}

    f_inte_summary = open(res_dir + inte_res[re] + dir_flag + 'inte_summary.csv', 'w+')

    f_inte_summary.write('province,max inte, max inte time\n')

    for pro in provins:
        f_inte = open(res_dir + inte_res[re] + dir_flag + pro + '.csv', 'r+')
        max_inte = -10
        for line in f_inte:
            line = line.replace('\n', '')
            line = eval(line)
            if line[1] > max_inte:
                max_inte = line[1]
                max_index = line[0]

        f_inte_summary.write('%s,%s,%s\n' % (pro, max_inte, max_index))
        f_inte.close()

    f_inte_summary.close()


def VREValueDistribution(vreYear, resTag):
    res_dir = getResDir(vreYear, resTag) + dir_flag

    wind_year = VreYearSplit(vreYear)[0]
    solar_year = VreYearSplit(vreYear)[1]

    with open(work_dir + 'data_pkl' + dir_flag + 'wind_cell_' + wind_year + '.pkl', 'rb+') as fin:
        wind_cell = pickle.load(fin)
    fin.close()

    with open(work_dir + 'data_pkl' + dir_flag + 'solar_cell_' + solar_year + '.pkl', 'rb+') as fin:
        solar_cell = pickle.load(fin)
    fin.close()

    wind_cell_value = {}
    solar_cell_value = {}

    provinces = wind_cell['provin_cf_sort'].keys()

    for pro in provinces:
        if pro not in wind_cell_value:
            wind_cell_value[pro] = []
        if pro not in solar_cell_value:
            solar_cell_value[pro] = []

        sp = []

        f_sp = open(res_dir + 'shadow_prices' + dir_flag + pro + '.csv', 'r+')

        for line in f_sp:
            line = line.replace('\n', '')
            line = eval(line)
            if line[1] < 0:
                sp.append(0)
            else:
                sp.append(round(line[1], 2))

        f_sp.close()

        sp = np.array(sp)

        for cell in range(len(wind_cell['provin_cf_sort'][pro])):
            value = sum(sp * wind_cell['cf_prof'][pro][cell])
            power = sum(wind_cell['cf_prof'][pro][cell])

            wind_cell_value[pro].append([
                wind_cell['provin_cf_sort'][pro][cell][7],
                wind_cell['provin_cf_sort'][pro][cell][8],
                value / (wind_cell['provin_cf_sort'][pro][cell][5] * power)
            ])

        for cell in range(len(solar_cell['provin_cf_sort'][pro])):
            if solar_cell['provin_cf_sort'][pro][cell][2] == 0:
                value = sum(sp * solar_cell['cf_prof'][pro][cell])
                power = sum(solar_cell['cf_prof'][pro][cell])

                solar_cell_value[pro].append([
                    solar_cell['provin_cf_sort'][pro][cell][7],
                    solar_cell['provin_cf_sort'][pro][cell][8],
                    value / (solar_cell['provin_cf_sort'][pro][cell][5] * power)
                ])

    with open(res_dir + 'wind_cell_value_' + wind_year + '.pkl', 'wb+') as fout:
        pickle.dump(wind_cell_value, fout)
    fout.close()

    with open(res_dir + 'solar_cell_value_' + solar_year + '.pkl', 'wb+') as fout:
        pickle.dump(solar_cell_value, fout)
    fout.close()


def loadSheddingInfo(vre_year, res_tag):
    res_dir = getResDir(vre_year, res_tag) + dir_flag

    wind_year = VreYearSplit(vre_year)[0]
    solar_year = VreYearSplit(vre_year)[1]

    wind_years = SplitMultipleVreYear(wind_year)
    solar_years = SplitMultipleVreYear(solar_year)

    year_count = len(wind_years)

    provins = []
    f_provins = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv', 'r+')
    next(f_provins)
    for line in f_provins:
        line = line.replace('\n', '')
        line = line.split(',')
        provins.append(line[1])
    f_provins.close()

    load_shedding = {}

    for pro in provins:
        if pro not in load_shedding:
            load_shedding[pro] = []
        f_ls = open(res_dir + 'load_shedding' + dir_flag + pro + '.csv', 'r+')
        for line in f_ls:
            line = line.replace('\n', '')
            line = line.split(',')

            load_shedding[pro].append(eval(line[1]))
        f_ls.close()

    save_as_pkl = {}
    f_res_ls = open(res_dir + 'load_shedding' + dir_flag + 'by_province.csv', 'w+')

    for pro in load_shedding:
        f_res_ls.write(pro + ',' + str(sum(load_shedding[pro])) + '\n')
        save_as_pkl[pro] = sum(load_shedding[pro])
    f_res_ls.close()

    with open(res_dir + 'load_shedding' + dir_flag + 'by_province.pkl', 'wb+') as fout:
        pickle.dump(save_as_pkl, fout)
    fout.close()


def TotalEnergyToHour(vre_year, res_tag):
    res_dir = getResDir(vre_year, res_tag) + dir_flag

    wind_year = VreYearSplit(vre_year)[0]
    solar_year = VreYearSplit(vre_year)[1]

    wind_years = SplitMultipleVreYear(wind_year)
    solar_years = SplitMultipleVreYear(solar_year)

    year_count = len(wind_years)

    mean_dem = {}

    with open(work_dir + 'data_pkl' + dir_flag + 'province_demand_full_2060.pkl', 'rb+') as fin:
        full_dem = pickle.load(fin)
    fin.close()

    provinces = full_dem.keys()

    for pro in full_dem:
        mean_dem[pro] = sum(full_dem[pro]) / len(full_dem[pro])

    es_tot = {'phs': {}, 'bat': {}, 'lds': {}}

    es_hour = {'phs': {}, 'bat': {}, 'lds': {}}

    for st in es_tot:
        for pro in provinces:
            if pro not in es_tot[st]:
                es_tot[st][pro] = np.zeros(8760 * year_count)

            for et in ['conv', 're']:
                f_est = open(res_dir + 'es_tot' + dir_flag + st + dir_flag + et + dir_flag + pro + '.csv', 'r+')
                for line in f_est:
                    line = line.replace('\n', '')
                    line = line.split(',')
                    es_tot[st][pro][eval(line[0])] += eval(line[1])
                f_est.close()

            es_hour[st][pro] = es_tot[st][pro] / mean_dem[pro]

    with open(res_dir + 'es_hour.pkl', 'wb+') as fout:
        pickle.dump(es_hour, fout)
    fout.close()


def convTransCount(vre_year, res_tag):
    res_dir = getResDir(vre_year, res_tag) + dir_flag

    count = 0

    for root, dirs, files in os.walk(res_dir + 'load_conv' + dir_flag + 'l2'):
        for file in files:
            f = open(res_dir + 'load_conv' + dir_flag + 'l2' + dir_flag + file, 'r+')
            for line in f:
                line = line.replace('\n', '')
                line = line.split(',')
                count += eval(line[1])

            f.close()

    print(count)


def CountAggegragetionConnectionCost(vre_year, res_tag):
    res_dir = getResDir(vre_year, res_tag) + dir_flag

    VreResFile = {'wind': 'wind_info.csv', 'solar': 'solar_info.csv'}

    AggResFile = {'wind': 'aggregated_wind_info.csv', 'solar': 'aggregated_solar_info.csv'}

    with open(work_dir + 'data_pkl' + dir_flag + 'wind_cell_2015.pkl', 'rb+') as fin:
        WindCell = pickle.load(fin)
    fin.close()

    with open(work_dir + 'data_pkl' + dir_flag + 'solar_cell_2015.pkl', 'rb+') as fin:
        SolarCell = pickle.load(fin)
    fin.close()

    VreCellFile = {'wind': WindCell, 'solar': SolarCell}

    with open(work_dir + 'data_pkl' + dir_flag + 'aggregated_cell.pkl', 'rb+') as fin:
        AggCell = pickle.load(fin)
    fin.close()

    with open(res_dir + 'scen_params.pkl', 'rb+') as fin:
        params = pickle.load(fin)
    fin.close()

    SpurCapExFixed = params['trans']['capex_spur_fixed'] * getCRF(7.4, 25)
    SpurCapExVar = params['trans']['capex_spur_var'] * getCRF(7.4, 25)

    TrunkCapExFixed = params['trans']['capex_trunk_fixed'] * getCRF(7.4, 25)
    TrunkCapExVar = params['trans']['capex_trunk_var'] * getCRF(7.4, 25)

    SpurTrunkCost = 0

    for re in ['wind', 'solar']:
        f_agg_res = open(res_dir + AggResFile[re], 'r+')

        for line in f_agg_res:
            line = line.replace('\n', '')
            line = line.split(',')
            k = eval(line[0])
            x = eval(line[1])

            pro = line[8]

            CellInfo = AggCell[re]['provin_cf_sort'][pro][k]

            cap = CellInfo[3]

            SpurDis = CellInfo[11]
            TrunkDis = CellInfo[12]

            MaxCf = max(AggCell[re]['cf_prof'][pro][k])

            SpurCost = x * cap * MaxCf * SpurDis * SpurCapExVar + x * cap * MaxCf * SpurCapExFixed

            TrunkCost = x * cap * MaxCf * TrunkDis * TrunkCapExVar + x * cap * MaxCf * TrunkCapExFixed

            SpurTrunkCost -= (SpurCost + TrunkCost)

    for re in ['wind', 'solar']:
        f_vre_res = open(res_dir + VreResFile[re], 'r+')

        for line in f_vre_res:
            line = line.replace('\n', '')
            line = line.split(',')
            k = eval(line[0])
            x = eval(line[1])

            pro = line[8]

            CellInfo = VreCellFile[re]['provin_cf_sort'][pro][k]

            cap = CellInfo[3]

            SpurDis = CellInfo[11]
            TrunkDis = CellInfo[12]

            MaxCf = max(VreCellFile[re]['cf_prof'][pro][k])

            SpurCost = x * cap * MaxCf * SpurDis * SpurCapExVar + x * cap * MaxCf * SpurCapExFixed

            TrunkCost = x * cap * MaxCf * TrunkDis * TrunkCapExVar + x * cap * MaxCf * TrunkCapExFixed

            SpurTrunkCost += (SpurCost + TrunkCost)

    save_as_pickle = {
        'spur_cost': SpurCost,
        'trunk_cost': TrunkCost
    }

    with open(res_dir + 'SpurTrunkCost.pkl', 'wb+') as fout:
        pickle.dump(save_as_pickle, fout)
    fout.close()

    print(SpurTrunkCost)


def TransCapDis(vre_year, res_tag):  # mw-km
    res_dir = getResDir(vre_year, res_tag) + dir_flag

    with open(work_dir + 'data_pkl' + dir_flag + 'model_exovar.pkl', 'rb+') as fin:
        exovar = pickle.load(fin)
    fin.close()

    trans_dis = exovar['trans_dis']

    tot_cap_dis = {}

    f_tot_trans_cap = open(res_dir + 'total_trans_cap.csv', 'r+')

    for line in f_tot_trans_cap:
        line = line.replace('\n', '')
        line = line.split(',')
        tot_cap_dis[(line[0], line[1])] = eval(line[2]) * trans_dis[(line[0], line[1])] / 1000
    f_tot_trans_cap.close()

    new_cap_dis = {}

    f_new_trans_cap = open(res_dir + 'new_trans_cap' + dir_flag + 'cap_trans_new.csv', 'r+')

    for line in f_new_trans_cap:
        line = line.replace('\n', '')
        line = line.split(',')
        new_cap_dis[(line[0], line[1])] = eval(line[2]) * trans_dis[(line[0], line[1])] / 1000
    f_new_trans_cap.close()

    with open(res_dir + 'trans_cap_dis.pkl', 'wb+') as fout:
        pickle.dump({'total': tot_cap_dis, 'new': new_cap_dis}, fout)
    fout.close()

    # print(sum(tot_cap_dis.values()),sum(new_cap_dis.values()))


def VreGenPotential(vre_year, vre):
    with open(work_dir + 'data_pkl' + dir_flag + vre + '_cell_' + vre_year + '.pkl', 'rb+') as fin:
        cell = pickle.load(fin)
    fin.close()

    cap_potential = 0

    for pro in cell['provin_cf_sort']:
        for c in cell['provin_cf_sort'][pro]:
            cap_potential += c[3]

    print(cap_potential)


def CountProvincialAnnualDemand(vre_year, res_tag, curr_year):
    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")

    with open(os.path.join(out_input_path, f'province_demand_full_{curr_year}.pkl'), 'rb+') as fin:
        pro_demand = pickle.load(fin)
    fin.close()

    NW = ['Gansu', 'Ningxia', 'Qinghai', 'Shaanxi', 'Xizang', 'Xinjiang']
    NE = ['Jilin', 'Heilongjiang', 'Liaoning', 'EestInnerMongolia']

    count_total = 0
    count_minor = 0
    for pro in pro_demand:
        count_total += sum(pro_demand[pro])
        if pro in NW or pro in NE:
            count_minor += sum(pro_demand[pro])

    print("Nation total demand: ", count_total)
    print("NW and NE total demand: ", count_minor)


if __name__ == '__main__':

    """
    Before running the test cases below, make sure that the output files exist.
    If not, use main.py to obtain optimization outputs.
    """

    # Test the following functions
    res_tag = "test_0822_365days_all_years_b1"
    vre_year = "w2015_s2015"
    curr_year = 2060

    # cellResInfo(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    # TransCap(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    # update_storage_capacity(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    TransInfo(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    # curtailed(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year, re="wind")
    # curtailed(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year, re="solar")
    # CurtailedSplitVRE(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    # CountProvincialAnnualDemand(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    # LoadProfile(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)

    # obtain_output_summary(vre_year="w2015_s2015", res_tag="test_0725_365days_all_years", curr_year=2030)
    # obtain_output_summary(vre_year="w2015_s2015", res_tag="test_0725_365days_all_years", curr_year=2040)
    # obtain_output_summary(vre_year="w2015_s2015", res_tag="test_0725_365days_all_years", curr_year=2050)
    # obtain_output_summary(vre_year="w2015_s2015", res_tag="test_0725_365days_all_years", curr_year=2060)
    # obtain_simulation_summary(vre_year="w2015_s2015", res_tag="test_0725_365days_all_years",
    #                           year_list=[2030, 2040, 2050, 2060])

    # TODO legacy code below
    '''for res_tag in ['Base1220','Aggeragated_VRE','BAT_OC_0.5X','BAT_OC_2X','Solar_OC_0.5X',
                    'Solar_OC_2X','Solar_PF_0.5','Transmission_0.5X','Transmission_2X',
                    'Wind_PF_0.5','Resv_0.1','Resv_within_province','Nuclear_MR_0.5','Nuclear_150GW',
                    'Neg_emission_0','Neg_emission_4','Neg_emission_2','Gas_cap_0.5X',
                    'Demand_1.2X','With_CAES','With_VRB']:
        cellResInfo(vre_tag,res_tag)'''
    # storage
    # averageStorageLength(vre_tag,scenario_tag)

    # wind solar
    # CountAggegragetionConnectionCost(vre_tag,scenario_tag)
    # VreGenPotential('2015','wind')
    # VreGenPotential('2015','Solar')
    # TransCapDis('w2015_s2015',scenario_tag)
    # VREValueDistribution(vre_tag,scenario_tag)
    # loadSheddingInfo(vre_tag,scenario_tag) #only for load shedding scenario
    # TotalEnergyToHour(vre_tag,scenario_tag)
    # convTransCount(vre_tag,scenario_tag)
