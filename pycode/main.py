import os
import pickle
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from callUtility import dirFlag, makeDir, getWorkDir, VreYearSplit, SplitMultipleVreYear
from initData import initCellData, initDemLayer, initModelExovar, getCRF, seedHour, SpurTrunkDis
from clearupData import averageStorageLength, cellResInfo, TransCap, TransInfo, LoadProfile, \
    curtailed, CurtailedSplitVRE, TotalEnergyToHour


def interProvinModel(vre_year, res_tag, init_data, is8760, curr_year, scen_params):
    dir_flag = dirFlag()
    work_dir = getWorkDir()

    # Specify file paths
    out_path = os.path.join(work_dir, "data_res", f"{res_tag}_{vre_year}", str(curr_year))
    out_input_path = os.path.join(out_path, "inputs")
    out_output_path = os.path.join(out_path, "outputs")
    out_output_processed_path = os.path.join(out_path, "outputs_processed")
    print("Results saved to {}".format(out_output_path))

    # Obtain secondary parameters
    wind_year = VreYearSplit(vre_year)[0]
    solar_year = VreYearSplit(vre_year)[1]

    # Specify the paths of the json/pkl files required for optimization
    path_scenario_parameters = os.path.join(out_input_path, "scen_params.json")
    path_wind_cell = os.path.join(out_input_path, "wind_cell.pkl")
    path_solar_cell = os.path.join(out_input_path, "solar_cell.pkl")
    path_province_full_demand = os.path.join(out_input_path, f"province_demand_full_{curr_year}.pkl")
    path_layer_cap_load = os.path.join(out_input_path, "layer_cap_load.pkl")
    path_model_exovar = os.path.join(out_input_path, "model_exovar.pkl")

    # Read parameters from the scenario parameter json file
    weighted_average_cost_of_capital = scen_params["finance"]["weighted_average_cost_of_capital"]  # percentage
    lifespan_storage_phs = scen_params["lifespan"]["storage_phs"]  # yrs
    lifespan_storage_bat = scen_params["lifespan"]["storage_bat"]  # yrs
    lifespan_transmission_spur = scen_params["lifespan"]["transmission_spur"]  # yrs
    lifespan_transmission_trunk = scen_params["lifespan"]["transmission_trunk"]  # yrs
    hydro_capex = scen_params["hydro"]["capex"]  # RMB/kW
    hydro_opex_fixed = scen_params["hydro"]["opex"]  # RMB/kW-yr
    hydro_lifespan = scen_params["hydro"]["lifespan"]  # yrs
    nuclear_capex = scen_params["nuclear"]["capex"]  # RMB/kW
    nuclear_opex_fixed = scen_params["nuclear"]["opex"]  # RMB/kW-yr
    nuclear_lifespan = scen_params["nuclear"]["lifespan"]  # yrs
    beccs_capex = scen_params["beccs"]["capex"]  # RMB/kW
    beccs_opex_fixed = scen_params["beccs"]["opex"]  # RMB/kW-yr
    beccs_lifespan = scen_params["beccs"]["lifespan"]  # yrs
    coal_capex = scen_params["coal"]["capex"]  # RMB/kW
    coal_opex_fixed = scen_params["coal"]["opex"]  # RMB/kW-yr
    coal_ccs_capex = scen_params["ccs"]["capex_coal_ccs"]  # RMB/kW
    coal_ccs_opex_fixed = scen_params["ccs"]["opex_coal_ccs"]  # RMB/kW-yr
    coal_lifespan = scen_params["coal"]["lifespan"]  # yrs
    gas_capex = scen_params["gas"]["capex"]  # RMB/kW
    gas_opex_fixed = scen_params["gas"]["opex"]  # RMB/kW-yr
    gas_ccs_capex = scen_params["ccs"]["capex_gas_ccs"]  # RMB/kW
    gas_ccs_opex_fixed = scen_params["ccs"]["opex_gas_ccs"]  # RMB/kW-yr
    gas_lifespan = scen_params["gas"]["lifespan"]  # yrs

    # Set parameters for ramp-up/down and reserve costs for each technology layer
    ru_c = {'l1': 50 * 0.001, 'l1_unabated': 50 * 0.001, 'l1_ccs': 50 * 0.001, 'l1_chp': 50 * 0.001,
            'l2': 0, 'l3': 200 * 0.001, 'l4': 10 * 0.001, 'l4_unabated': 10 * 0.001, 'l4_ccs': 10 * 0.001}  # RMB/kWh
    rd_c = {'l1': 0, 'l1_unabated': 0, 'l1_ccs': 0, 'l1_chp': 0,
            'l2': 0, 'l3': 0, 'l4': 0, 'l4_unabated': 0, 'l4_ccs': 0}  # RMB/kWh
    resv_p = {'l1': 24 * 0.001, 'l1_unabated': 24 * 0.001, 'l1_ccs': 24 * 0.001, 'l1_chp': 24 * 0.001,
              'l2': 0, 'l3': 25 * 0.001, 'l4': 26 * 0.001, 'l4_unabated': 26 * 0.001, 'l4_ccs': 26 * 0.001}  # RMB/kWh

    # Save input scenario parameter file
    with open(path_scenario_parameters, 'w+') as fp:
        json.dump(scen_params, fp)
    fp.close()

    # Read input scenario parameters TODO: this section is not used
    if scen_params['vre']['inter_annual'] == 1:
        wind_years = SplitMultipleVreYear(wind_year)
        solar_years = SplitMultipleVreYear(solar_year)
        save_in_data_pkl = 1
        year_count = len(wind_years)
    else:
        wind_years = [wind_year]
        solar_years = [solar_year]
        year_count = 1

    # Initialize optimization hours
    if is8760 == 0:
        seedHour(vre_year=vre_year,
                 years=scen_params["optimization_hours"]["years"],
                 step=scen_params["optimization_hours"]["step"],
                 days=scen_params["optimization_hours"]["days"],
                 res_tag=res_tag, curr_year=curr_year)
    else:
        seedHour(vre_year=vre_year,
                 years=year_count,
                 step=28,
                 days=15,
                 res_tag=res_tag, curr_year=curr_year)
    hour_seed = pd.read_csv(os.path.join(out_input_path, "hour_seed.csv"), header=None).iloc[:, 0].to_list()
    hour_end = hour_seed[-1]
    with open(os.path.join(out_input_path, "hour_pre.pkl"), 'rb+') as fin:
        hour_pre = pickle.load(fin)
    fin.close()
    Hour = 8760 * year_count
    print(f'Year count: {year_count}, final hour: {hour_end}')

    # # Initialize wind/solar cell pickle files if init_data is 1
    # if init_data:
    #     print('Initializing data...')
    #     if scen_params['vre']['aggregated'] == 0:
    #         print("Init cell Data - Wind and Solar")
    #         if scen_params['vre']['inter_annual'] == 0:
    #             initCellData(
    #                 vre='wind',
    #                 vre_year_single=wind_year,
    #                 equip=equip_wind,
    #                 other=other_wind,
    #                 om=om_wind,
    #                 cap_scale=scen_params['vre']['cap_scale_wind'],
    #                 cap_scale_east=scen_params['vre']['cap_scale_wind_ep'],
    #                 hour_seed=hour_seed,  # GetHourSeed(vre_year, res_tag),
    #                 res_tag=res_tag,
    #                 vre_year=vre_year,
    #                 curr_year=curr_year
    #             )
    #             initCellData(
    #                 vre='solar',
    #                 vre_year_single=solar_year,
    #                 equip=equip_solar,
    #                 other=other_solar,
    #                 om=om_solar,
    #                 cap_scale=scen_params['vre']['cap_scale_pv'],
    #                 cap_scale_east=scen_params['vre']['cap_scale_pv_ep'],
    #                 hour_seed=hour_seed,  # GetHourSeed(vre_year, res_tag),
    #                 res_tag=res_tag,
    #                 vre_year=vre_year,
    #                 curr_year=curr_year
    #             )
    #         else:
    #             initMultipleYearCell(
    #                 'solar',
    #                 solar_year,
    #                 equip_solar,
    #                 other_solar,
    #                 om_solar,
    #                 scen_params['vre']['cap_scale_pv'],
    #                 res_tag,
    #                 vre_year,
    #                 save_in_data_pkl
    #             )
    #             initMultipleYearCell(
    #                 'wind',
    #                 wind_year,
    #                 equip_wind,
    #                 other_wind,
    #                 om_wind,
    #                 scen_params['vre']['cap_scale_wind'],
    #                 res_tag,
    #                 vre_year,
    #                 save_in_data_pkl
    #             )
    #
    #     print('Data initialization completed...')

    # Read VRE cell data (integrated or aggregated)
    if scen_params['vre']['aggregated'] == 0:
        with open(path_wind_cell, 'rb') as fin:
            wind_cell = pickle.load(fin)
        fin.close()
        with open(path_solar_cell, 'rb') as fin:
            solar_cell = pickle.load(fin)
        fin.close()
    else:
        pass

    # Read province hourly demand data
    with open(path_province_full_demand, 'rb+') as fin:
        pro_dem_full = pickle.load(fin)
    fin.close()

    # Firm resource capacity limit by province (nuclear, coal, hydro, gas, beccs, bio)
    with open(path_layer_cap_load, 'rb') as fin:
        layer_cap_load = pickle.load(fin)
    fin.close()
    layer_cf_norm = layer_cap_load['layer_cf_norm']
    layer_cap_total = layer_cap_load['layer_cap_total']
    layer_cap_hourly = layer_cap_load['layer_cap_hourly']

    # Read exogenous and endogenous variable values
    with open(path_model_exovar, 'rb') as fin:
        model_exovar = pickle.load(fin)
    fin.close()
    cap_trans = model_exovar['inter_pro_trans']  # capacity of the existing transmission lines
    trans_new = model_exovar['new_trans']  # if there are existing lines
    trans_dis = model_exovar['trans_dis']  # Distances between provinces
    trans_voltage = model_exovar['trans_voltage']
    capex_trans_cap = model_exovar['capex_trans_cap']
    capex_trans_dis = model_exovar['capex_trans_dis']
    grid_pro = model_exovar['grid_pro']
    phs_ub = model_exovar['phs_ub']
    phs_lb = model_exovar['phs_lb']
    bat_lb = model_exovar['bat_lb']
    caes_lb = model_exovar['caes_lb']
    vrb_lb = model_exovar['vrb_lb']
    lds_lb = {"caes": caes_lb, "vrb": vrb_lb}
    hydro_lb = model_exovar['hydro_lb']
    nuclear_lb = model_exovar['nuclear_lb']
    beccs_lb = model_exovar['beccs_lb']
    bio_lb = model_exovar['bio_lb']

    # Obtain province list in China
    provins = []
    provin_abbrev = open(work_dir + 'data_csv' + dir_flag + 'geography/China_provinces_hz.csv')
    next(provin_abbrev)
    for line in provin_abbrev:
        line = line.replace('\n', '')
        line = line.split(',')
        provins.append(line[1])
    provins = sorted(provins)
    provin_abbrev.close()

    # Obtain winter hours in China
    winter_hour = []
    f_wh = open(work_dir + 'data_csv' + dir_flag + 'simulation_meta/winter_hour.csv', 'r+')
    for line in f_wh:
        line = line.replace('\n', '')
        line = eval(line)
        winter_hour.append(line)
    f_wh.close()

    # Identify transmission line connections between provinces
    trans_to = {}
    trans_from = {}
    for pro in cap_trans:
        if pro[0] not in trans_to:
            trans_to[pro[0]] = []
        if pro[1] not in trans_from:
            trans_from[pro[1]] = []
        trans_to[pro[0]].append(pro[1])
        trans_from[pro[1]].append(pro[0])

    # Read parameters for storage (BAT, PHS)
    capex_power_phs = scen_params['storage']['capex_power_phs'] * \
                      getCRF(weighted_average_cost_of_capital, lifespan_storage_phs)
    capex_power_bat = scen_params['storage']['capex_power_bat'] * \
                      getCRF(weighted_average_cost_of_capital, lifespan_storage_bat)
    fixed_omc_phs = scen_params['storage']['fixed_omc_phs']
    fixed_omc_bat = scen_params['storage']['fixed_omc_bat']
    var_omc_phs = scen_params['storage']['var_omc_phs']  # yuan/kwh
    var_omc_bat = scen_params['storage']['var_omc_bat']
    rt_effi_phs = scen_params['storage']['rt_effi_phs']
    rt_effi_bat = scen_params['storage']['rt_effi_bat']
    duration_phs = scen_params['storage']['duration_phs']
    duration_bat = scen_params['storage']['duration_bat']
    capex_annual_power_phs = capex_power_phs + fixed_omc_phs
    capex_annual_power_bat = capex_power_bat + fixed_omc_bat
    sdiss_phs = scen_params['storage']['sdiss_phs']
    sdiss_bat = scen_params['storage']['sdiss_bat']

    # Read parameters for storage (Long-duration)
    with_lds = scen_params['storage']['with_lds']  # 是否有长时储能
    lds = []
    if scen_params['storage']['with_caes']:
        lds.append('caes')
    if scen_params['storage']['with_vrb']:
        lds.append('vrb')
    capex_power_lds = {}
    capex_energy_lds = {}
    fixed_omc_lds = {}
    var_omc_lds = {}
    rt_effi_lds = {}
    duration_lds = {}
    capex_annual_power_lds = {}
    sdiss_lds = {}
    for st in lds:
        lifespan_storage_lds = scen_params['storage']['span_lds'][st]
        capex_power_lds[st] = scen_params['storage']['capex_power_lds'][st] * \
                              getCRF(weighted_average_cost_of_capital, lifespan_storage_lds)
        capex_energy_lds[st] = scen_params['storage']['capex_energy_lds'][st] * \
                               getCRF(weighted_average_cost_of_capital, lifespan_storage_lds)
        fixed_omc_lds[st] = scen_params['storage']['fixed_omc_lds'][st]
        var_omc_lds[st] = scen_params['storage']['var_omc_lds'][st]
        rt_effi_lds[st] = scen_params['storage']['rt_effi_lds'][st]
        duration_lds[st] = scen_params['storage']['duration_lds'][st]
        capex_annual_power_lds[st] = capex_power_lds[st] + fixed_omc_lds[st] + duration_lds[st] * capex_energy_lds[st]
        sdiss_lds[st] = scen_params['storage']['sdiss_lds'][st]

    cap_phs = {}
    cap_bat = {}
    cap_lds = {}

    charge_phs = {'wind': {}, 'solar': {}, 'l2': {}, 'l3': {}}
    charge_bat = {'wind': {}, 'solar': {}, 'l2': {}, 'l3': {}}
    charge_lds = {}

    dischar_phs = {}
    dischar_bat = {}
    dischar_lds = {}

    resv_bat = {}
    resv_lds = {}
    resv_phs = {}

    tot_energy_phs = {}
    tot_energy_bat = {}
    tot_energy_lds = {}

    for st in lds:
        cap_lds[st] = {}
        charge_lds[st] = {'wind': {}, 'solar': {}, 'l2': {}, 'l3': {}}
        dischar_lds[st] = {}
        resv_lds[st] = {}
        tot_energy_lds[st] = {}

    # Read parameters for transmission lines
    trans_loss = scen_params['trans']['trans_loss']
    capex_spur_fixed = year_count * scen_params['trans']['capex_spur_fixed'] * \
                       getCRF(weighted_average_cost_of_capital, lifespan_transmission_spur)
    capex_spur_var = scen_params['trans']['capex_spur_var'] * \
                     getCRF(weighted_average_cost_of_capital, lifespan_transmission_spur)  # yuan/(km*kw)
    capex_trunk_fixed = year_count * scen_params['trans']['capex_trunk_fixed'] * \
                        getCRF(weighted_average_cost_of_capital, lifespan_transmission_trunk)
    capex_trunk_var = scen_params['trans']['capex_trunk_var'] * \
                      getCRF(weighted_average_cost_of_capital, lifespan_transmission_trunk)

    # Read reserve requirements
    vre_resv = scen_params['resv']['vre_resv']  # VRE generation
    demand_resv = scen_params['resv']['demand_resv']  # Firm generation

    # Read CCS parameters
    coal_ccs_loss = scen_params['ccs']['coal_loss']
    gas_ccs_loss = scen_params['ccs']['gas_loss']
    bio_ccs_loss = scen_params['ccs']['bio_loss']

    # Set decision variables
    # L1: coal, L2: hydro, L3: nuclear, L4: gas
    ru = {'l1_unabated': {}, 'l1_ccs': {}, 'l1_chp': {}, 'l2': {}, 'l3': {}, 'l4_unabated': {}, 'l4_ccs': {}}
    rd = {'l1_unabated': {}, 'l1_ccs': {}, 'l1_chp': {}, 'l2': {}, 'l3': {}, 'l4_unabated': {}, 'l4_ccs': {}}
    load_conv = {'l1_unabated': {}, 'l1_ccs': {}, 'l1_chp': {}, 'l2': {}, 'l3': {}, 'l4_unabated': {}, 'l4_ccs': {}}
    load_resv = {'l1_unabated': {}, 'l1_ccs': {}, 'l1_chp': {}, 'l2': {}, 'l3': {}, 'l4_unabated': {}, 'l4_ccs': {}}
    trans_out = {'l1_unabated': {}, 'l1_ccs': {}, 'l1_chp': {}, 'l2': {}, 'l3': {}, 'l4_unabated': {}, 'l4_ccs': {},
                 'wind': {}, 'solar': {}}

    # Set model parameters
    interProvinModel = gp.Model('lp')  # Model name
    interProvinModel.setParam('Method', 2)  # Method=2 barrier for QP; method=3 concurrent for LP
    interProvinModel.setParam('Crossover', 0)  # Use value 0 to disable crossover
    # interProvinModel.setParam('PreSparsify', 2)  # Value 2 to reduce non-zero variables in pre-solved models
    # interProvinModel.setParam('MIPGap', 1e-1)
    # interProvinModel.setParam('QCPDual', 1)  # Product of two variables when calculating emissions
    # interProvinModel.setParam('NonConvex', 2)  # QCP constraints do not have PSD Q

    # Calculate the total count fo wind/solar cells within each province
    wind_cell_num = {}
    solar_cell_num = {}
    for pro in provins:
        wind_cell_num[pro] = len(wind_cell['provin_cf_sort'][pro])
        solar_cell_num[pro] = len(solar_cell['provin_cf_sort'][pro])
    if not scen_params['vre']['wind_with_xz']:  # No wind in Tibet in practice
        wind_cell_num['Xizang'] = 0

    # Set decision variables for transfered electricity
    load_trans = {}
    cap_trans_new = {}
    for pro in cap_trans:
        load_trans[pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)

    for pro in trans_new:
        if scen_params['trans']['is_full_inter_province'] == 0:
            if trans_new[pro] == 0:
                cap_trans_new[pro] = interProvinModel.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS)
            else:
                cap_trans_new[pro] = interProvinModel.addVar(lb=0, vtype=GRB.CONTINUOUS)
        else:
            if pro[0] != pro[1]:
                cap_trans_new[pro] = interProvinModel.addVar(lb=0, vtype=GRB.CONTINUOUS)
            else:
                cap_trans_new[pro] = interProvinModel.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS)

    # Set decision variables for renewable and firm operations
    for pro in provins:
        # Set decision variables of ramp ups / downs, load offset, load reserves, and transferred electricity
        for l in ["l1_unabated", "l1_ccs", "l1_chp", "l2", "l3", "l4_unabated", "l4_ccs"]:
            ru[l][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
            rd[l][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
            load_conv[l][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
            load_resv[l][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
            trans_out[l][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)

        for l in ['wind', 'solar']:
            trans_out[l][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)

    print("Model update #1...")
    interProvinModel.update()

    # Set decision variables for renewable capacities
    x_wind, x_solar, inte_wind, inte_solar = {}, {}, {}, {}
    is_chp_online = {}
    load_shedding = {}
    for pro in provins:
        # Set decision variables of wind/solar capacity
        x_wind[pro] = interProvinModel.addVars(wind_cell_num[pro], lb=0, ub=1, vtype=GRB.CONTINUOUS)
        x_solar[pro] = interProvinModel.addVars(solar_cell_num[pro], lb=0, ub=1, vtype=GRB.CONTINUOUS)
        inte_wind[pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
        inte_solar[pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)

        # Set decision variables of storage capacity
        cap_phs[pro] = interProvinModel.addVar(lb=phs_lb[pro], ub=phs_ub[pro], vtype=GRB.CONTINUOUS)
        cap_bat[pro] = interProvinModel.addVar(lb=bat_lb[pro], vtype=GRB.CONTINUOUS)
        for st in lds:
            cap_lds[st][pro] = interProvinModel.addVar(lb=lds_lb[st][pro], vtype=GRB.CONTINUOUS)

        if scen_params['shedding']['with_shedding'] == 1:
            load_shedding[pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
        else:
            load_shedding[pro] = interProvinModel.addVars(Hour, lb=0, ub=0, vtype=GRB.CONTINUOUS)

        for et in ['wind', 'solar', 'l2', 'l3']:
            charge_phs[et][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
            charge_bat[et][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
            for st in lds:
                charge_lds[st][et][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)

        dischar_phs[pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
        dischar_bat[pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
        for st in lds:
            dischar_lds[st][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)

        resv_bat[pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
        resv_phs[pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
        for st in lds:
            resv_lds[st][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)

        tot_energy_phs[pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
        tot_energy_bat[pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)
        for st in lds:
            tot_energy_lds[st][pro] = interProvinModel.addVars(Hour, lb=0, vtype=GRB.CONTINUOUS)

    print("Model update #2...")
    interProvinModel.update()

    # Set decision variables for firm capacities
    cap_gw = {}
    cap_hourly = {}
    for pro in provins:
        cap_gw[pro] = {}
        cap_hourly[pro] = {}
        for tech in ["coal_unabated", "coal_ccs", "chp_ccs", "hydro", "nuclear",
                     "bio", "beccs", "gas_unabated", "gas_ccs"]:
            # Initialize firm capacity decision variables
            cap_gw[pro][tech] = interProvinModel.addVar(lb=0, vtype=GRB.CONTINUOUS)

            # Obtain maximum hourly capacity (capacity factors x total capacity)
            cap_hourly[pro][tech] = layer_cf_norm[pro][f"{tech}_hourly_cf"] * cap_gw[pro][tech]

        # Add constraints: equal to initial firm resource capacity if exogenous
        if not scen_params["scenario"]["endogenize_firm_capacity"]:
            for tech in ["coal_unabated", "coal_ccs", "chp_ccs", "hydro", "nuclear",
                         "bio", "beccs", "gas_unabated", "gas_ccs"]:
                interProvinModel.addConstr(cap_gw[pro][tech] == layer_cap_total[pro][tech])

        # Add constraints: firm resource capacity upper/lower bounds if endogenous
        else:
            # Coal capacity should not exceed the fleet size after natural retirement
            emission_factors = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions",
                                                        "coal_natural_retire", f"coal_{curr_year}_pre.csv"))
            interProvinModel.addConstr(cap_gw[pro]["coal_unabated"] + cap_gw[pro]["coal_ccs"] <=
                                       emission_factors[emission_factors["province"] == pro]["cap_gw_ub"].values[0])

            # Gas capacity should not exceed the fleet size after natural retirement
            emission_factors = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions",
                                                        "gas_natural_retire", f"gas_{curr_year}_pre.csv"))
            interProvinModel.addConstr(cap_gw[pro]["gas_unabated"] + cap_gw[pro]["gas_ccs"] <=
                                       emission_factors[emission_factors["province"] == pro]["cap_gw_ub"].values[0])

            # CHP CCS capacity should be pre-solved
            if scen_params["scenario"]["ccs_start_year"] > curr_year:
                interProvinModel.addConstr(cap_gw[pro]["chp_ccs"] == 0)
            elif scen_params["scenario"]["heating_electrification"] == "heat_pump":
                interProvinModel.addConstr(cap_gw[pro]["chp_ccs"] == 0)
                # Add induced electricity demands if heat pumps replace CHP CCS plants
                provincial_cop = pd.read_csv(os.path.join(work_dir, "data_csv", "demand_assumptions",
                                                          "province_temp_monthly_cop.csv")).set_index("hour")
                if pro in provincial_cop.columns:
                    for hr in winter_hour:
                        pro_dem_full[pro][hr] += layer_cap_total[pro]["chp_ccs"] * 1.5 / provincial_cop[pro][hr]
                # Update provincial demand files
                with open(path_province_full_demand, 'wb') as fout:
                    pickle.dump(pro_dem_full, fout)
                fp.close()
            else:
                interProvinModel.addConstr(cap_gw[pro]["chp_ccs"] == layer_cap_total[pro]["chp_ccs"])

            # Hydro/Nuclear capacity upper bounds
            interProvinModel.addConstr(cap_gw[pro]["hydro"] <= layer_cap_total[pro]["hydro"])
            interProvinModel.addConstr(cap_gw[pro]["nuclear"] <= layer_cap_total[pro]["nuclear"])

            # # Nuclear capacity should not exceed the planned capacity by 2060
            # nuclear_ub = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "nuclear_2060.csv"),
            #                          header=None)
            # nuclear_ub.columns = ["province", "cap_mw"]
            # interProvinModel.addConstr(cap_gw[pro]["nuclear"] <=
            #                            nuclear_ub[nuclear_ub["province"] == pro]["cap_mw"].values[0] / 1e3)
            #
            # # Hydro capacity should not exceed the planned capacity by 2060
            # hydro_ub = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "hydro_2060.csv"))
            # interProvinModel.addConstr(cap_gw[pro]["hydro"] <=
            #                            hydro_ub[hydro_ub["province"] == pro]["2060_cap_mw"].values[0] / 1e3)

            # BECCS capacity should not exceed the planned capacity by 2060
            if scen_params["scenario"]["emission_target"] == "15C":
                beccs_ub = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "beccs_2060_15C.csv"),
                                       header=None)
            else:
                beccs_ub = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions", "beccs_2060.csv"),
                                       header=None)
            beccs_ub.columns = ["province", "cap_gw"]
            interProvinModel.addConstr(cap_gw[pro]["beccs"] <=
                                       beccs_ub[beccs_ub["province"] == pro]["cap_gw"].values[0])

            # BIO capacity should be pre-solved
            interProvinModel.addConstr(cap_gw[pro]["bio"] == layer_cap_total[pro]["bio"])

            # Hydro/Nuclear/BECCS capacity lower bounds
            interProvinModel.addConstr(cap_gw[pro]["hydro"] >= min(hydro_lb[pro], layer_cap_total[pro]["hydro"]))
            interProvinModel.addConstr(cap_gw[pro]["nuclear"] >= nuclear_lb[pro])
            interProvinModel.addConstr(cap_gw[pro]["beccs"] >= beccs_lb[pro])

            # CCS capacity = 0 if before retrofit starts or CCS enters the market
            if scen_params["scenario"]["ccs_start_year"] > curr_year:
                interProvinModel.addConstr(cap_gw[pro]["coal_ccs"] == 0)
                interProvinModel.addConstr(cap_gw[pro]["gas_ccs"] == 0)
                interProvinModel.addConstr(cap_gw[pro]["beccs"] == 0)

            # In NDC mode, gas capacity in 2035 should be pre-solved
            if curr_year == 2035:
                interProvinModel.addConstr(cap_gw[pro]["gas_unabated"] + cap_gw[pro]["gas_ccs"] ==
                                           emission_factors[emission_factors["province"] == pro]["cap_gw_ub"].values[0])

    # Calculate cost terms
    # Wind generating cost 风电总生产成本
    wind_gen_cost = [wind_cell['provin_cf_sort'][pro][c][5] *
                     wind_cell['provin_cf_sort'][pro][c][4] *
                     x_wind[pro][c]
                     for pro in provins for c in range(wind_cell_num[pro])]

    # Solar generating cost 光电总生产成本
    solar_gen_cost = [solar_cell['provin_cf_sort'][pro][c][5] *  # LCOE of each cell
                      solar_cell['provin_cf_sort'][pro][c][4] *  # max NWh = maximum potential of each cell * cf
                      x_solar[pro][c]  # Share of potential used
                      for pro in provins for c in range(solar_cell_num[pro])]

    # Storage fixed cost 储能系统固定总成本
    fixed_phs_cost = [year_count * capex_annual_power_phs * cap_phs[pro] for pro in provins]
    fixed_bat_cost = [year_count * capex_annual_power_bat * cap_bat[pro] for pro in provins]
    fixed_lds_cost = [year_count * capex_annual_power_lds[st] * cap_lds[st][pro] for st in lds for pro in provins]

    # Storage variable cost 储能系统可变成本
    var_phs_cost = [var_omc_phs * dischar_phs[pro][h] for pro in provins for h in hour_seed]
    var_bat_cost = [var_omc_bat * dischar_bat[pro][h] for pro in provins for h in hour_seed]
    var_lds_cost = [var_omc_lds[st] * dischar_lds[st][pro][h] for st in lds for pro in provins for h in hour_seed]

    # Ramp-up/down costs 爬坡与退坡成本
    ramp_up_cost = [ru_c[l] * ru[l][pro][h]
                    for l in ["l1_unabated", "l1_ccs", "l1_chp", "l2", "l3", "l4_unabated", "l4_ccs"]
                    for pro in provins for h in hour_seed]
    ramp_dn_cost = [rd_c[l] * rd[l][pro][h]
                    for l in ["l1_unabated", "l1_ccs", "l1_chp", "l2", "l3", "l4_unabated", "l4_ccs"]
                    for pro in provins for h in hour_seed]

    # Reserve costs 储备总成本
    resv_cost = [resv_p[l] * load_resv[l][pro][h]
                 for l in ["l1_unabated", "l1_ccs", "l1_chp", "l2", "l3", "l4_unabated", "l4_ccs"]
                 for pro in provins for h in hour_seed]

    # Calculate annualized capital costs, fixed O&M costs, and variable costs
    coal_cost_fixed_per_hour = \
        (coal_capex * getCRF(weighted_average_cost_of_capital, coal_lifespan) + coal_opex_fixed) / 8760
    coal_ccs_cost_fixed_per_hour = \
        ((coal_capex + coal_ccs_capex) * getCRF(weighted_average_cost_of_capital,
                                                coal_lifespan) + coal_ccs_opex_fixed) / 8760
    coal_ccs_cost_fixed = [coal_ccs_cost_fixed_per_hour *
                           cap_gw[pro]["coal_ccs"] *
                           scen_params["optimization_hours"]["days"] * 24 for pro in provins]
    coal_unabated_cost_fixed = [coal_cost_fixed_per_hour *
                                cap_gw[pro]["coal_unabated"] *
                                scen_params["optimization_hours"]["days"] * 24 for pro in provins]
    coal_cost_var = [scen_params['coal']['var_cost'] *
                     (load_conv['l1_unabated'][pro][h] + trans_out['l1_unabated'][pro][h] +
                      load_conv['l1_ccs'][pro][h] + trans_out['l1_ccs'][pro][h] +
                      load_conv['l1_chp'][pro][h] + trans_out['l1_chp'][pro][h])
                     for pro in provins for h in hour_seed]

    gas_cost_fixed_per_hour = \
        (gas_capex * getCRF(weighted_average_cost_of_capital, gas_lifespan) + gas_opex_fixed) / 8760
    gas_ccs_cost_fixed_per_hour = \
        ((gas_capex + gas_ccs_capex) * getCRF(weighted_average_cost_of_capital,
                                              gas_lifespan) + gas_ccs_opex_fixed) / 8760
    gas_ccs_cost_fixed = [gas_ccs_cost_fixed_per_hour *
                          cap_gw[pro]["gas_ccs"] *
                          scen_params["optimization_hours"]["days"] * 24 for pro in provins]
    gas_unabated_cost_fixed = [gas_cost_fixed_per_hour *
                               cap_gw[pro]["gas_unabated"] *
                               scen_params["optimization_hours"]["days"] * 24 for pro in provins]
    gas_cost_var = [scen_params['gas']['var_cost'] *
                    (load_conv['l4_unabated'][pro][h] + trans_out['l4_unabated'][pro][h] +
                     load_conv['l4_ccs'][pro][h] + trans_out['l4_ccs'][pro][h])
                    for pro in provins for h in hour_seed]

    hydro_cost_fixed_per_hour = \
        (hydro_capex * getCRF(weighted_average_cost_of_capital, hydro_lifespan) + hydro_opex_fixed) / 8760
    hydro_cost_fixed = [hydro_cost_fixed_per_hour *
                        cap_gw[pro]["hydro"] *
                        scen_params["optimization_hours"]["days"] * 24 for pro in provins]
    hydro_cost_var = [scen_params["hydro"]["var_cost"] * (load_conv['l2'][pro][h] + trans_out['l2'][pro][h] +
                                                          charge_phs['l2'][pro][h] + charge_bat['l2'][pro][h] +
                                                          gp.quicksum([charge_lds[st]['l2'][pro][h] for st in lds]))
                      for pro in provins for h in hour_seed]

    beccs_cost_fixed_per_hour = \
        (beccs_capex * getCRF(weighted_average_cost_of_capital, beccs_lifespan) + beccs_opex_fixed) / 8760
    beccs_cost_fixed = [beccs_cost_fixed_per_hour *
                        cap_gw[pro]["beccs"] *
                        scen_params["optimization_hours"]["days"] * 24 for pro in provins]
    bio_cost_var_must_run = [scen_params['beccs']['var_cost'] *
                             (cap_gw[pro]["beccs"] + cap_gw[pro]["bio"]) * scen_params['beccs']['must_run'] *
                             scen_params["optimization_hours"]["days"] * 24 for pro in provins]

    nuclear_cost_fixed_hour = \
        (nuclear_capex * getCRF(weighted_average_cost_of_capital, nuclear_lifespan) + nuclear_opex_fixed) / 8760
    nuclear_cost_fixed = [nuclear_cost_fixed_hour *
                          cap_gw[pro]["nuclear"] *
                          scen_params["optimization_hours"]["days"] * 24 for pro in provins]
    nuclear_cost_var = [scen_params['nuclear']['var_cost'] *
                        (load_conv['l3'][pro][h] + trans_out['l3'][pro][h] +
                         charge_phs['l3'][pro][h] + charge_bat['l3'][pro][h] +
                         gp.quicksum([charge_lds[st]['l3'][pro][h] for st in lds]))
                        for pro in provins for h in hour_seed]

    cpv_solar = {}
    for pro in provins:
        cpv_solar[pro] = []
        for c in range(solar_cell_num[pro]):
            if solar_cell['provin_cf_sort'][pro][c][2] == 0:
                cpv_solar[pro].append(c)

    # Calculate wind/solar spur/trunk line connection costs
    spur_cost_wind, spur_cost_solar, trunk_cost = [], [], []
    if scen_params['vre']['aggregated'] == 0:
        # VRE spur line costs for cell-substation connections 格点至变电站成本
        spur_cost_wind = [(x_wind[pro][c] - wind_cell['provin_cf_sort'][pro][c][13]) *
                          wind_cell['provin_cf_sort'][pro][c][3] *
                          wind_cell['provin_cf_sort'][pro][c][11] *
                          max(wind_cell['cf_prof'][pro][c]) * capex_spur_var
                          + (x_wind[pro][c] - wind_cell['provin_cf_sort'][pro][c][13]) *
                          wind_cell['provin_cf_sort'][pro][c][3] *
                          max(wind_cell['cf_prof'][pro][c]) * capex_spur_fixed
                          for pro in provins for c in range(wind_cell_num[pro])]

        spur_cost_solar = [
            (x_solar[pro][c] - solar_cell['provin_cf_sort'][pro][c][13]) *
            solar_cell['provin_cf_sort'][pro][c][3] *
            solar_cell['provin_cf_sort'][pro][c][11] *
            max(solar_cell['cf_prof'][pro][c]) * capex_spur_var +
            (x_solar[pro][c] - solar_cell['provin_cf_sort'][pro][c][13]) *
            solar_cell['provin_cf_sort'][pro][c][3] *
            max(solar_cell['cf_prof'][pro][c]) * capex_spur_fixed
            for pro in provins for c in cpv_solar[pro]
        ]

        # VRE trunk line costs for substation-node connections 变电站至主干网成本
        sub_cell = {}
        for pro in wind_cell['sub_cell_info']:
            if pro != 'Xizang':
                if pro not in sub_cell:
                    sub_cell[pro] = {}
                for s in wind_cell['sub_cell_info'][pro]:
                    if s not in sub_cell[pro]:
                        sub_cell[pro][s] = {}

                    sub_cell[pro][s]['wind_cell'] = wind_cell['sub_cell_info'][pro][s]['cell']
                    sub_cell[pro][s]['dis'] = wind_cell['sub_cell_info'][pro][s]['dis']

                    if s in solar_cell['sub_cell_info'][pro]:
                        sub_cfs = (
                                wind_cell['sub_cell_info'][pro][s]['cap'] * wind_cell['sub_cell_info'][pro][s]['cf'] + \
                                solar_cell['sub_cell_info'][pro][s]['cap'] * solar_cell['sub_cell_info'][pro][s]['cf'])

                        sub_cap = wind_cell['sub_cell_info'][pro][s]['cap'] + solar_cell['sub_cell_info'][pro][s]['cap']

                        sub_cell[pro][s]['max_cf'] = max(sub_cfs) / sub_cap
                        sub_cell[pro][s]['solar_cell'] = solar_cell['sub_cell_info'][pro][s]['cell']
                    else:
                        sub_cell[pro][s]['max_cf'] = max(wind_cell['sub_cell_info'][pro][s]['cf']) / \
                                                     wind_cell['sub_cell_info'][pro][s]['cap']
                        sub_cell[pro][s]['solar_cell'] = []

        for pro in solar_cell['sub_cell_info']:
            if pro not in sub_cell:
                sub_cell[pro] = {}
            for s in solar_cell['sub_cell_info'][pro]:
                if s not in sub_cell[pro]:
                    sub_cell[pro][s] = {}
                    sub_cell[pro][s]['solar_cell'] = solar_cell['sub_cell_info'][pro][s]['cell']
                    sub_cell[pro][s]['max_cf'] = max(solar_cell['sub_cell_info'][pro][s]['cf']) / \
                                                 solar_cell['sub_cell_info'][pro][s]['cap']
                    sub_cell[pro][s]['wind_cell'] = []
                    sub_cell[pro][s]['dis'] = solar_cell['sub_cell_info'][pro][s]['dis']

        trunk_cost = [((x_wind[pro][w] - wind_cell['provin_cf_sort'][pro][w][13]) *
                       wind_cell['provin_cf_sort'][pro][w][3] +
                       (x_solar[pro][s] - solar_cell['provin_cf_sort'][pro][s][13]) *
                       solar_cell['provin_cf_sort'][pro][s][3]) * sub_cell[pro][sub]['max_cf'] *
                      (sub_cell[pro][sub]['dis'] * capex_trunk_var + capex_trunk_fixed)
                      for pro in provins
                      for sub in sub_cell[pro]
                      for w in sub_cell[pro][sub]['wind_cell']
                      for s in sub_cell[pro][sub]['solar_cell']]
    elif scen_params['vre']['aggregated'] == 1:
        spur_cost_wind = [(x_wind[pro][c] - wind_cell['provin_cf_sort'][pro][c][13]) *
                          wind_cell['provin_cf_sort'][pro][c][3] *
                          wind_cell['provin_cf_sort'][pro][c][11] *
                          max(wind_cell['cf_prof'][pro][c]) * capex_spur_var +
                          (x_wind[pro][c] - wind_cell['provin_cf_sort'][pro][c][13]) *
                          wind_cell['provin_cf_sort'][pro][c][3] *
                          max(wind_cell['cf_prof'][pro][c]) * capex_spur_fixed
                          for pro in provins for c in range(wind_cell_num[pro])]

        spur_cost_solar = [(x_solar[pro][c] - solar_cell['provin_cf_sort'][pro][c][13]) *
                           solar_cell['provin_cf_sort'][pro][c][3] *
                           solar_cell['provin_cf_sort'][pro][c][11] *
                           max(solar_cell['cf_prof'][pro][c]) * capex_spur_var +
                           (x_solar[pro][c] - solar_cell['provin_cf_sort'][pro][c][13]) *
                           solar_cell['provin_cf_sort'][pro][c][3] *
                           max(solar_cell['cf_prof'][pro][c]) * capex_spur_fixed
                           for pro in provins for c in range(solar_cell_num[pro])]

        trunk_cost_wind = [
            (x_wind[pro][c] - wind_cell['provin_cf_sort'][pro][c][13])
            * max(wind_cell['cf_prof'][pro][c]) * wind_cell['provin_cf_sort'][pro][c][3]
            * (
                    capex_trunk_fixed + capex_trunk_var * wind_cell['provin_cf_sort'][pro][c][12]
            )
            for pro in provins
            for c in range(wind_cell_num[pro])
        ]

        trunk_cost_solar = [
            (x_solar[pro][c] - solar_cell['provin_cf_sort'][pro][c][13])
            * max(solar_cell['cf_prof'][pro][c]) * solar_cell['provin_cf_sort'][pro][c][3]
            * (
                    capex_trunk_fixed + capex_trunk_var * solar_cell['provin_cf_sort'][pro][c][12]
            )
            for pro in provins
            for c in range(solar_cell_num[pro])
        ]

        trunk_cost = trunk_cost_wind + trunk_cost_solar

    trans_pair_count = []
    trans_cost = []
    intertrans_cost_scale = scen_params['trans']['interprovincial_scale']

    for pro in trans_new:
        if trans_new[pro] != 0 and pro not in trans_pair_count:

            trans_pair_count.append((pro[1], pro[0]))
            if pro not in trans_voltage:
                voltage = 500  # kV
            else:
                voltage = trans_voltage[pro]
            if voltage < 500:  # kV
                voltage = 500  # kV
            trans_cost.append(
                2 * intertrans_cost_scale * year_count * capex_trans_cap[voltage]
                * (cap_trans_new[pro] + cap_trans_new[(pro[1], pro[0])] + cap_trans[pro])
                + intertrans_cost_scale * year_count * capex_trans_dis[voltage]
                * trans_dis[pro] * (cap_trans_new[pro] + cap_trans_new[(pro[1], pro[0])] + cap_trans[pro])
            )

    # Calculate load shedding costs
    other_tech_cost = [0]
    if scen_params['shedding']['with_shedding'] == 1:
        load_shedding_cost = [
            load_shedding[pro][h] * scen_params['shedding']['shedding_vom']
            for pro in provins
            for h in hour_seed
        ]

        other_tech_cost += load_shedding_cost

    # Calculate offshore, onshore, DPV, and UPV generation before curtailment at the provincial level
    offshore_gen, onshore_gen, upv_gen, dpv_gen = {}, {}, {}, {}
    for pro in provins:
        offshore_gen[pro] = gp.quicksum([wind_cell['provin_cf_sort'][pro][c][4] * x_wind[pro][c]
                                         for c in range(wind_cell_num[pro])
                                         if wind_cell['provin_cf_sort'][pro][c][2] == 0])
        onshore_gen[pro] = gp.quicksum([wind_cell['provin_cf_sort'][pro][c][4] * x_wind[pro][c]
                                        for c in range(wind_cell_num[pro])
                                        if wind_cell['provin_cf_sort'][pro][c][2] == 1])
        upv_gen[pro] = gp.quicksum([solar_cell['provin_cf_sort'][pro][c][4] * x_solar[pro][c]
                                    for c in range(solar_cell_num[pro])
                                    if solar_cell['provin_cf_sort'][pro][c][2] == 0])
        dpv_gen[pro] = gp.quicksum([solar_cell['provin_cf_sort'][pro][c][4] * x_solar[pro][c]
                                    for c in range(solar_cell_num[pro])
                                    if solar_cell['provin_cf_sort'][pro][c][2] == 1])

    # Calculate renewable incentives
    if scen_params["scenario"]["ptc_mode"] == "none":
        offshore_cost_var = [offshore_gen[pro] * 0 for pro in provins]
        onshore_cost_var = [onshore_gen[pro] * 0 for pro in provins]
        upv_cost_var = [upv_gen[pro] * 0 for pro in provins]
        dpv_cost_var = [dpv_gen[pro] * 0 for pro in provins]
    elif scen_params["scenario"]["ptc_mode"] == "uniform":
        offshore_cost_var = [offshore_gen[pro] * (- 0.2) for pro in provins]
        onshore_cost_var = [onshore_gen[pro] * 0 for pro in provins]
        upv_cost_var = [upv_gen[pro] * 0 for pro in provins]
        dpv_cost_var = [dpv_gen[pro] * (- 0.2) for pro in provins]
    else:
        incentives = pd.read_csv(os.path.join(work_dir, "data_csv", "cost_assumptions", "renewable_incentives.csv"))
        offshore_cost_var = [offshore_gen[pro] * (
            - incentives[incentives["province"] == pro]["offshore_RMB_per_kWh"].values[0]) for pro in provins]
        onshore_cost_var = [onshore_gen[pro] * (
            - incentives[incentives["province"] == pro]["onshore_RMB_per_kWh"].values[0]) for pro in provins]
        upv_cost_var = [upv_gen[pro] * (
            - incentives[incentives["province"] == pro]["upv_RMB_per_kWh"].values[0]) for pro in provins]
        dpv_cost_var = [dpv_gen[pro] * (
            - incentives[incentives["province"] == pro]["dpv_RMB_per_kWh"].values[0]) for pro in provins]

    # Set optimization objective
    interProvinModel.setObjective(gp.quicksum(wind_gen_cost) +
                                  gp.quicksum(solar_gen_cost) +
                                  gp.quicksum(offshore_cost_var) +
                                  gp.quicksum(onshore_cost_var) +
                                  gp.quicksum(upv_cost_var) +
                                  gp.quicksum(dpv_cost_var) +
                                  gp.quicksum(ramp_up_cost) +
                                  gp.quicksum(ramp_dn_cost) +
                                  gp.quicksum(fixed_phs_cost) +
                                  gp.quicksum(fixed_bat_cost) +
                                  gp.quicksum(fixed_lds_cost) +
                                  gp.quicksum(var_phs_cost) +
                                  gp.quicksum(var_bat_cost) +
                                  gp.quicksum(var_lds_cost) +
                                  gp.quicksum(resv_cost) +
                                  gp.quicksum(spur_cost_wind) +
                                  gp.quicksum(spur_cost_solar) +
                                  gp.quicksum(trunk_cost) +
                                  gp.quicksum(trans_cost) +
                                  gp.quicksum(beccs_cost_fixed) +
                                  gp.quicksum(bio_cost_var_must_run) +
                                  gp.quicksum(hydro_cost_fixed) +
                                  gp.quicksum(hydro_cost_var) +
                                  gp.quicksum(nuclear_cost_fixed) +
                                  gp.quicksum(nuclear_cost_var) +
                                  gp.quicksum(coal_ccs_cost_fixed) +
                                  gp.quicksum(coal_unabated_cost_fixed) +
                                  gp.quicksum(coal_cost_var) +
                                  gp.quicksum(gas_ccs_cost_fixed) +
                                  gp.quicksum(gas_unabated_cost_fixed) +
                                  gp.quicksum(gas_cost_var) +
                                  gp.quicksum(other_tech_cost))

    print("Model update #3...")
    interProvinModel.update()

    # Add optimization constraints
    # Renewables are greater than previous year's installed capacities 大于等于现有装机
    interProvinModel.addConstrs(
        x_wind[pro][c] >= wind_cell['provin_cf_sort'][pro][c][13]
        for pro in provins for c in range(wind_cell_num[pro]))
    interProvinModel.addConstrs(
        x_solar[pro][c] >= solar_cell['provin_cf_sort'][pro][c][13]
        for pro in provins for c in range(solar_cell_num[pro]))

    # Storage operation constraints (Storage operation)
    # 期初与期末储能系统总储存能量相等
    interProvinModel.addConstrs(tot_energy_phs[pro][0] == tot_energy_phs[pro][hour_end] for pro in provins)
    interProvinModel.addConstrs(tot_energy_bat[pro][0] == tot_energy_bat[pro][hour_end] for pro in provins)

    for st in lds:
        interProvinModel.addConstrs(tot_energy_lds[st][pro][0] == tot_energy_lds[st][pro][hour_end] for pro in provins)

    # 第1小时与第2小时放电量限制
    interProvinModel.addConstrs(
        resv_phs[pro][0] + dischar_phs[pro][0] +
        resv_phs[pro][1] + dischar_phs[pro][1] <=
        tot_energy_phs[pro][0] +
        gp.quicksum([charge_phs[et][pro][0] for et in ['wind', 'solar', 'l2', 'l3']])
        for pro in provins)

    interProvinModel.addConstrs(
        resv_bat[pro][0] + dischar_bat[pro][0] +
        resv_bat[pro][1] + dischar_bat[pro][1] <=
        (1 - sdiss_bat) * tot_energy_bat[pro][0] +
        gp.quicksum([charge_bat[et][pro][0] for et in ['wind', 'solar', 'l2', 'l3']])
        for pro in provins)

    for st in lds:
        interProvinModel.addConstrs(
            resv_lds[st][pro][0] + dischar_lds[st][pro][0] +
            resv_lds[st][pro][1] + dischar_lds[st][pro][1] <=
            tot_energy_lds[st][pro][0] +
            gp.quicksum([charge_lds[st][et][pro][0] for et in ['wind', 'solar', 'l2', 'l3']])
            for pro in provins)

    # 第2小时结束后储能系统中储存的能量
    interProvinModel.addConstrs(
        tot_energy_phs[pro][1] == tot_energy_phs[pro][0] - dischar_phs[pro][0] - dischar_phs[pro][1]
        + gp.quicksum([charge_phs[et][pro][0] + charge_phs[et][pro][1] for et in ['wind', 'solar', 'l2', 'l3']])
        for pro in provins)

    interProvinModel.addConstrs(
        tot_energy_bat[pro][1] == (1 - sdiss_bat) * tot_energy_bat[pro][0] - dischar_bat[pro][0] - dischar_bat[pro][1]
        + gp.quicksum([charge_bat[et][pro][0] + charge_bat[et][pro][1] for et in ['wind', 'solar', 'l2', 'l3']])
        for pro in provins)

    for st in lds:
        interProvinModel.addConstrs(
            tot_energy_lds[st][pro][1] == tot_energy_lds[st][pro][0] - dischar_lds[st][pro][0] - dischar_lds[st][pro][1]
            + gp.quicksum(
                [charge_lds[st][et][pro][0] + charge_lds[st][et][pro][1] for et in ['wind', 'solar', 'l2', 'l3']])
            for pro in provins)

    # 第h小时结束后储能系统中存储的能量
    interProvinModel.addConstrs(
        tot_energy_phs[pro][h] == tot_energy_phs[pro][hour_pre[h]] - dischar_phs[pro][h] +
        gp.quicksum([charge_phs[et][pro][h] for et in ['wind', 'solar', 'l2', 'l3']])
        for pro in provins for h in hour_seed[2:])

    interProvinModel.addConstrs(
        tot_energy_bat[pro][h] == (1 - sdiss_bat) * tot_energy_bat[pro][hour_pre[h]] - dischar_bat[pro][h] +
        gp.quicksum([charge_bat[et][pro][h] for et in ['wind', 'solar', 'l2', 'l3']])
        for pro in provins for h in hour_seed[2:])

    for st in lds:
        interProvinModel.addConstrs(
            tot_energy_lds[st][pro][h] == tot_energy_lds[st][pro][hour_pre[h]] - dischar_lds[st][pro][h] +
            gp.quicksum([charge_lds[st][et][pro][h] for et in ['wind', 'solar', 'l2', 'l3']])
            for pro in provins for h in hour_seed[2:])
    # 放电约束
    interProvinModel.addConstrs(resv_phs[pro][h] + dischar_phs[pro][h] <= tot_energy_phs[pro][hour_pre[h]]
                                for pro in provins for h in hour_seed[2:])

    interProvinModel.addConstrs(resv_bat[pro][h] + dischar_bat[pro][h] <= tot_energy_bat[pro][hour_pre[h]]
                                for pro in provins for h in hour_seed[2:])

    for st in lds:
        interProvinModel.addConstrs(
            resv_lds[st][pro][h] + dischar_lds[st][pro][h] <= tot_energy_lds[st][pro][hour_pre[h]]
            for pro in provins for h in hour_seed[2:])

    # 储能总量约束
    interProvinModel.addConstrs(tot_energy_phs[pro][h] <= duration_phs * cap_phs[pro]
                                for pro in provins for h in hour_seed)

    interProvinModel.addConstrs(tot_energy_bat[pro][h] <= duration_bat * cap_bat[pro]
                                for pro in provins for h in hour_seed)

    for st in lds:
        interProvinModel.addConstrs(tot_energy_lds[st][pro][h] <= duration_lds[st] * cap_lds[st][pro]
                                    for pro in provins for h in hour_seed)

    # 容量约束
    interProvinModel.addConstrs(
        gp.quicksum([charge_phs[et][pro][h] for et in ['wind', 'solar', 'l2', 'l3']]) <= cap_phs[pro]
        for pro in provins for h in hour_seed)

    interProvinModel.addConstrs(rt_effi_phs * dischar_phs[pro][h] <= cap_phs[pro] for pro in provins for h in hour_seed)

    interProvinModel.addConstrs(rt_effi_phs * dischar_phs[pro][h] + rt_effi_phs * resv_phs[pro][h]
                                <= cap_phs[pro] + gp.quicksum(
        [charge_phs[et][pro][h] for et in ['wind', 'solar', 'l2', 'l3']])
                                for pro in provins for h in hour_seed)

    interProvinModel.addConstrs(
        gp.quicksum([charge_bat[et][pro][h] for et in ['wind', 'solar', 'l2', 'l3']]) <= cap_bat[pro]
        for pro in provins for h in hour_seed)

    interProvinModel.addConstrs(rt_effi_bat * dischar_bat[pro][h] <= cap_bat[pro] for pro in provins for h in hour_seed)

    interProvinModel.addConstrs(rt_effi_bat * dischar_bat[pro][h] + rt_effi_bat * resv_bat[pro][h]
                                <= cap_bat[pro] + gp.quicksum(
        [charge_bat[et][pro][h] for et in ['wind', 'solar', 'l2', 'l3']])
                                for pro in provins for h in hour_seed)

    for st in lds:
        interProvinModel.addConstrs(
            gp.quicksum([charge_lds[st][et][pro][h] for et in ['wind', 'solar', 'l2', 'l3']]) <= cap_lds[st][pro]
            for pro in provins for h in hour_seed)

        interProvinModel.addConstrs(
            rt_effi_lds[st] * dischar_lds[st][pro][h] <= cap_lds[st][pro] for pro in provins for h in hour_seed)

        interProvinModel.addConstrs(rt_effi_lds[st] * dischar_lds[st][pro][h] + rt_effi_lds[st] * resv_lds[st][pro][h]
                                    <= cap_lds[st][pro] + gp.quicksum(
            [charge_lds[st][et][pro][h] for et in ['wind', 'solar', 'l2', 'l3']])
                                    for pro in provins for h in hour_seed)

    # Renewable and transmission operation constraints (VRE output; Transmission operation)
    for pro in provins:
        if pro in trans_to.keys():
            interProvinModel.addConstrs(inte_wind[pro][h] + charge_phs['wind'][pro][h] + charge_bat['wind'][pro][h]
                                        + gp.quicksum([charge_lds[st]['wind'][pro][h] for st in lds])
                                        + trans_out['wind'][pro][h]
                                        <= gp.quicksum(
                [x_wind[pro][c] * wind_cell['provin_cf_sort'][pro][c][3] * wind_cell['cf_prof'][pro][c][h]
                 for c in range(wind_cell_num[pro])])
                                        for h in hour_seed)

            interProvinModel.addConstrs(inte_solar[pro][h] + charge_phs['solar'][pro][h] + charge_bat['solar'][pro][h]
                                        + gp.quicksum([charge_lds[st]['solar'][pro][h] for st in lds])
                                        + trans_out['solar'][pro][h]
                                        <= gp.quicksum(
                [x_solar[pro][c] * solar_cell['provin_cf_sort'][pro][c][3] * solar_cell['cf_prof'][pro][c][h]
                 for c in range(solar_cell_num[pro])])
                                        for h in hour_seed)

            interProvinModel.addConstrs(gp.quicksum([load_trans[(pro, pro2)][h] for pro2 in trans_to[pro]])
                                        == trans_out['wind'][pro][h] + trans_out['solar'][pro][h] +
                                        trans_out['l2'][pro][h] + trans_out['l3'][pro][h] +
                                        trans_out['l1_unabated'][pro][h] +
                                        (1 - coal_ccs_loss) * trans_out['l1_ccs'][pro][h] +
                                        (1 - coal_ccs_loss) * trans_out['l1_chp'][pro][h] +
                                        trans_out['l4_unabated'][pro][h] +
                                        (1 - gas_ccs_loss) * trans_out['l4_ccs'][pro][h]
                                        for h in hour_seed)

        else:
            interProvinModel.addConstrs(inte_wind[pro][h] + charge_phs['wind'][pro][h] + charge_bat['wind'][pro][h]
                                        + gp.quicksum([charge_lds[st]['wind'][pro][h] for st in lds])
                                        <=
                                        gp.quicksum([x_wind[pro][c] * wind_cell['provin_cf_sort'][pro][c][3] *
                                                     wind_cell['cf_prof'][pro][c][h]
                                                     for c in range(wind_cell_num[pro])])
                                        for h in hour_seed)

            interProvinModel.addConstrs(inte_solar[pro][h] + charge_phs['solar'][pro][h] + charge_bat['solar'][pro][h]
                                        + gp.quicksum([charge_lds[st]['solar'][pro][h] for st in lds])
                                        <= gp.quicksum(
                [x_solar[pro][c] * solar_cell['provin_cf_sort'][pro][c][3] * solar_cell['cf_prof'][pro][c][h]
                 for c in range(solar_cell_num[pro])])
                                        for h in hour_seed)

    # Demand-supply balances at the provincial level (Demand)
    demConstrs = {}
    for pro in provins:
        if pro in trans_from.keys():
            demConstrs[pro] = interProvinModel.addConstrs(
                load_conv['l1_unabated'][pro][h] +
                (1 - coal_ccs_loss) * load_conv['l1_ccs'][pro][h] +
                (1 - coal_ccs_loss) * load_conv['l1_chp'][pro][h] +
                load_conv['l2'][pro][h] +
                load_conv['l3'][pro][h] +
                load_conv['l4_unabated'][pro][h] +
                (1 - gas_ccs_loss) * load_conv['l4_ccs'][pro][h] +
                (1 - bio_ccs_loss) * cap_gw[pro]["beccs"] * scen_params['beccs']['must_run'] +
                cap_gw[pro]["bio"] * scen_params['beccs']['must_run'] +
                gp.quicksum([(1 - trans_loss) ** trans_dis[(pro1, pro)] * load_trans[(pro1, pro)][h] for pro1 in
                             trans_from[pro]]) +
                inte_wind[pro][h] +
                inte_solar[pro][h] +
                scen_params['shedding']['with_shedding'] * load_shedding[pro][h] +
                rt_effi_phs * dischar_phs[pro][h] +
                rt_effi_bat * dischar_bat[pro][h] +
                gp.quicksum([rt_effi_lds[st] * dischar_lds[st][pro][h] for st in lds])
                ==
                pro_dem_full[pro][h]
                for h in hour_seed
            )
        else:
            demConstrs[pro] = interProvinModel.addConstrs(
                load_conv['l1_unabated'][pro][h] +
                (1 - coal_ccs_loss) * load_conv['l1_ccs'][pro][h] +
                (1 - coal_ccs_loss) * load_conv['l1_chp'][pro][h] +
                load_conv['l2'][pro][h] +
                load_conv['l3'][pro][h] +
                load_conv['l4_unabated'][pro][h] +
                (1 - gas_ccs_loss) * load_conv['l4_ccs'][pro][h] +
                (1 - bio_ccs_loss) * cap_gw[pro]["beccs"] * scen_params['beccs']['must_run'] +
                cap_gw[pro]["bio"] * scen_params['beccs']['must_run'] +
                inte_wind[pro][h] +
                inte_solar[pro][h] +
                scen_params['shedding']['with_shedding'] * load_shedding[pro][h] +
                rt_effi_phs * dischar_phs[pro][h] +
                rt_effi_bat * dischar_bat[pro][h] +
                gp.quicksum([rt_effi_lds[st] * dischar_lds[st][pro][h] for st in lds])
                ==
                pro_dem_full[pro][h]
                for h in hour_seed)

    for pro_pair in cap_trans:
        interProvinModel.addConstrs(
            gp.quicksum([load_trans[(pro_pair[0], pro_pair[1])][h] + load_trans[(pro_pair[1], pro_pair[0])][h]])
            <=
            cap_trans[pro_pair] + cap_trans_new[(pro_pair[0], pro_pair[1])] + cap_trans_new[(pro_pair[1], pro_pair[0])]
            for h in hour_seed)

    print("Model update #4...")
    interProvinModel.update()

    # Layer supply should not exceed total layer capacity (Dispatchable supply)
    # Layer 1: coal
    interProvinModel.addConstrs(
        load_conv['l1_unabated'][pro][h] + load_resv['l1_unabated'][pro][h] + trans_out['l1_unabated'][pro][h]
        <= cap_hourly[pro]["coal_unabated"][h] for pro in provins for h in hour_seed)
    interProvinModel.addConstrs(
        load_conv['l1_ccs'][pro][h] + load_resv['l1_ccs'][pro][h] + trans_out['l1_ccs'][pro][h]
        <= cap_hourly[pro]["coal_ccs"][h] for pro in provins for h in hour_seed)
    interProvinModel.addConstrs(
        load_conv['l1_chp'][pro][h] + load_resv['l1_chp'][pro][h] + trans_out['l1_chp'][pro][h]
        <= cap_hourly[pro]["chp_ccs"][h] for pro in provins for h in hour_seed)
    # Layer 2: hydro
    interProvinModel.addConstrs(load_resv['l2'][pro][h] +
                                load_conv['l2'][pro][h] +
                                charge_phs['l2'][pro][h] +
                                charge_bat['l2'][pro][h] +
                                gp.quicksum([charge_lds[st]['l2'][pro][h] for st in lds]) +
                                trans_out['l2'][pro][h]
                                <= cap_hourly[pro]["hydro"][h]
                                for pro in provins for h in hour_seed)
    # Layer 3: nuclear
    interProvinModel.addConstrs(load_conv['l3'][pro][h] +
                                charge_phs['l3'][pro][h] +
                                charge_bat['l3'][pro][h] +
                                gp.quicksum([charge_lds[st]['l3'][pro][h] for st in lds]) +
                                trans_out['l3'][pro][h]
                                <= cap_hourly[pro]["nuclear"][h]
                                for pro in provins for h in hour_seed)
    # Layer 4: gas
    interProvinModel.addConstrs(
        load_conv['l4_unabated'][pro][h] + load_resv['l4_unabated'][pro][h] + trans_out['l4_unabated'][pro][h]
        <= cap_hourly[pro]["gas_unabated"][h] for pro in provins for h in hour_seed)
    interProvinModel.addConstrs(
        load_conv['l4_ccs'][pro][h] + load_resv['l4_ccs'][pro][h] + trans_out['l4_ccs'][pro][h]
        <= cap_hourly[pro]["gas_ccs"][h] for pro in provins for h in hour_seed)

    # Layer has minimum CF requirements
    # Layer 3: nuclear has a must-run requirement
    interProvinModel.addConstrs(load_conv['l3'][pro][h] +
                                charge_phs['l3'][pro][h] +
                                charge_bat['l3'][pro][h] +
                                gp.quicksum([charge_lds[st]['l3'][pro][h] for st in lds]) +
                                trans_out['l3'][pro][h]
                                >= cap_hourly[pro]["nuclear"][h] * scen_params['nuclear']["must_run"]
                                for pro in provins for h in hour_seed)
    # CHP CCS has a 100% must-run requirement in winter hours
    for pro in provins:
        for h in hour_seed:
            if h in winter_hour:
                interProvinModel.addConstr(
                    load_conv['l1_chp'][pro][h] + trans_out['l1_chp'][pro][h]  # Note: reserves does not count
                    == cap_hourly[pro]["chp_ccs"][h])
            else:
                pass

    # Demand and VRE reserve requirements at each province or grid region (Reserve capacity)
    credible_solar = 1
    credible_wind = 1
    if scen_params['resv']['vre_resv_provincialy'] == 0:
        for g in grid_pro:
            # Demand reserve constraint
            interProvinModel.addConstrs(
                gp.quicksum([
                    gp.quicksum([1 * cap_hourly[pro]["coal_unabated"][h]]) +
                    gp.quicksum([1 * cap_hourly[pro]["coal_ccs"][h]]) +
                    gp.quicksum([1 * cap_hourly[pro]["chp_ccs"][h]]) +
                    gp.quicksum([1 * cap_hourly[pro]["hydro"][h]]) +
                    gp.quicksum([1 * cap_hourly[pro]["nuclear"][h]]) +
                    gp.quicksum([1 * cap_hourly[pro]["gas_unabated"][h]]) +
                    gp.quicksum([1 * cap_hourly[pro]["gas_ccs"][h]]) +
                    gp.quicksum([1 * cap_hourly[pro]["beccs"][h]]) +
                    gp.quicksum([1 * cap_hourly[pro]["bio"][h]]) +
                    credible_wind * gp.quicksum(
                        [x_wind[pro][c] * wind_cell['provin_cf_sort'][pro][c][3] * wind_cell['cf_prof'][pro][c][h]
                         for c in range(wind_cell_num[pro])]) +
                    credible_solar * gp.quicksum(
                        [x_solar[pro][c] * solar_cell['provin_cf_sort'][pro][c][3] * solar_cell['cf_prof'][pro][c][h]
                         for c in range(solar_cell_num[pro])]) +
                    scen_params['shedding']['with_shedding'] * gp.quicksum([load_shedding[pro][h]]) +
                    rt_effi_phs * dischar_phs[pro][h] +
                    rt_effi_bat * dischar_bat[pro][h] +
                    gp.quicksum([rt_effi_lds[st] * dischar_lds[st][pro][h] for st in lds]) +
                    rt_effi_phs * resv_phs[pro][h] + rt_effi_bat * resv_bat[pro][h] +
                    gp.quicksum([rt_effi_lds[st] * resv_lds[st][pro][h] for st in lds]) +
                    gp.quicksum([(1 - trans_loss) ** trans_dis[(pro1, pro)] * load_trans[(pro1, pro)][h]
                                 for pro1 in trans_from[pro]]) -
                    trans_out['wind'][pro][h] - trans_out['solar'][pro][h] -
                    trans_out['l2'][pro][h] - trans_out['l3'][pro][h] -
                    trans_out['l4_unabated'][pro][h] - trans_out['l4_ccs'][pro][h] -
                    trans_out['l1_unabated'][pro][h] - trans_out['l1_ccs'][pro][h] - trans_out['l1_chp'][pro][h]
                    for pro in grid_pro[g]])
                >=
                gp.quicksum([(1 + demand_resv) * pro_dem_full[pro][h] for pro in grid_pro[g]])
                for h in hour_seed)

            # VRE integration reserve constraint
            interProvinModel.addConstrs(
                gp.quicksum([
                    load_resv['l1_unabated'][pro][h] + load_resv['l1_ccs'][pro][h] + load_resv['l1_chp'][pro][h] +
                    load_resv['l2'][pro][h] + load_resv['l3'][pro][h] +
                    load_resv['l4_unabated'][pro][h] + load_resv['l4_ccs'][pro][h] +
                    rt_effi_phs * resv_phs[pro][h] + rt_effi_bat * resv_bat[pro][h] +
                    gp.quicksum([rt_effi_lds[st] * resv_lds[st][pro][h] for st in lds])
                    for pro in grid_pro[g]])
                >=
                gp.quicksum([vre_resv * (inte_wind[pro][h] + inte_solar[pro][h] +
                                         trans_out['wind'][pro][h] + trans_out['solar'][pro][h])
                             for pro in grid_pro[g]])
                for h in hour_seed)
    else:
        # Demand reserve constraint
        interProvinModel.addConstrs(
            gp.quicksum([1 * cap_hourly[pro]["coal_unabated"][h]]) +
            gp.quicksum([1 * cap_hourly[pro]["coal_ccs"][h]]) +
            gp.quicksum([1 * cap_hourly[pro]["chp_ccs"][h]]) +
            gp.quicksum([1 * cap_hourly[pro]["hydro"][h]]) +
            gp.quicksum([1 * cap_hourly[pro]["nuclear"][h]]) +
            gp.quicksum([1 * cap_hourly[pro]["gas_unabated"][h]]) +
            gp.quicksum([1 * cap_hourly[pro]["gas_ccs"][h]]) +
            gp.quicksum([1 * cap_hourly[pro]["beccs"][h]]) +
            gp.quicksum([1 * cap_hourly[pro]["bio"][h]]) +
            credible_wind * gp.quicksum(
                [x_wind[pro][c] * wind_cell['provin_cf_sort'][pro][c][3] * wind_cell['cf_prof'][pro][c][h]
                 for c in range(wind_cell_num[pro])]) +
            credible_solar * gp.quicksum(
                [x_solar[pro][c] * solar_cell['provin_cf_sort'][pro][c][3] * solar_cell['cf_prof'][pro][c][h]
                 for c in range(solar_cell_num[pro])]) +
            scen_params['shedding']['with_shedding'] * load_shedding[pro][h] +
            rt_effi_phs * dischar_phs[pro][h] + rt_effi_bat * dischar_bat[pro][h] +
            gp.quicksum([rt_effi_lds[st] * dischar_lds[st][pro][h] for st in lds]) +
            rt_effi_phs * resv_phs[pro][h] + rt_effi_bat * resv_bat[pro][h] +
            gp.quicksum([rt_effi_lds[st] * resv_lds[st][pro][h] for st in lds]) +
            gp.quicksum(
                [(1 - trans_loss) ** trans_dis[(pro1, pro)] * load_trans[(pro1, pro)][h]
                 for pro1 in trans_from[pro]]) -
            trans_out['wind'][pro][h] - trans_out['solar'][pro][h] -
            trans_out['l2'][pro][h] - trans_out['l3'][pro][h] -
            trans_out['l4_unabated'][pro][h] - trans_out['l4_ccs'][pro][h] -
            trans_out['l1_unabated'][pro][h] - trans_out['l1_ccs'][pro][h] - trans_out['l1_chp'][pro][h]
            >=
            (1 + demand_resv) * pro_dem_full[pro][h]
            for pro in provins for h in hour_seed)

        # VRE reserve constraint
        interProvinModel.addConstrs(
            load_resv['l1_unabated'][pro][h] + load_resv['l1_ccs'][pro][h] + load_resv['l1_chp'][pro][h] +
            load_resv['l2'][pro][h] + load_resv['l3'][pro][h] +
            load_resv['l4_unabated'][pro][h] + load_resv['l4_ccs'][pro][h] +
            rt_effi_phs * resv_phs[pro][h] + rt_effi_bat * resv_bat[pro][h] +
            gp.quicksum([rt_effi_lds[st] * resv_lds[st][pro][h] for st in lds])
            >=
            vre_resv * (inte_wind[pro][h] + inte_solar[pro][h] + trans_out['wind'][pro][h] + trans_out['solar'][pro][h])
            for pro in provins for h in hour_seed)

    # Ramp up/down constraints for each layer (Ramp capacity)
    for pro in provins:
        for h in hour_seed[1:]:
            interProvinModel.addConstr(
                ru["l1_unabated"][pro][h] <= scen_params['ramp']["l1"] * cap_gw[pro]["coal_unabated"])
            interProvinModel.addConstr(
                rd["l1_unabated"][pro][h] <= scen_params['ramp']["l1"] * cap_gw[pro]["coal_unabated"])
            interProvinModel.addConstr(
                ru["l1_ccs"][pro][h] <= scen_params['ramp']["l1"] * cap_gw[pro]["coal_ccs"])
            interProvinModel.addConstr(
                rd["l1_ccs"][pro][h] <= scen_params['ramp']["l1"] * cap_gw[pro]["coal_ccs"])
            interProvinModel.addConstr(
                ru["l1_chp"][pro][h] <= scen_params['ramp']["l1"] * cap_gw[pro]["chp_ccs"])
            interProvinModel.addConstr(
                rd["l1_chp"][pro][h] <= scen_params['ramp']["l1"] * cap_gw[pro]["chp_ccs"])
            interProvinModel.addConstr(
                ru["l4_unabated"][pro][h] <= scen_params['ramp']["l4"] * cap_gw[pro]["gas_unabated"])
            interProvinModel.addConstr(
                rd["l4_unabated"][pro][h] <= scen_params['ramp']["l4"] * cap_gw[pro]["gas_unabated"])
            interProvinModel.addConstr(
                ru["l4_ccs"][pro][h] <= scen_params['ramp']["l4"] * cap_gw[pro]["gas_ccs"])
            interProvinModel.addConstr(
                rd["l4_ccs"][pro][h] <= scen_params['ramp']["l4"] * cap_gw[pro]["gas_ccs"])

            for l in ["l1_unabated", "l1_ccs", "l1_chp", "l4_unabated", "l4_ccs"]:
                interProvinModel.addConstr(
                    ru[l][pro][h] >=
                    load_conv[l][pro][h] + trans_out[l][pro][h] -
                    load_conv[l][pro][hour_pre[h]] - trans_out[l][pro][hour_pre[h]])
                interProvinModel.addConstr(
                    rd[l][pro][h] >=
                    load_conv[l][pro][hour_pre[h]] + trans_out[l][pro][hour_pre[h]] -
                    load_conv[l][pro][h] - trans_out[l][pro][h])

            interProvinModel.addConstr(ru["l2"][pro][h] <= scen_params['ramp']["l2"] * cap_gw[pro]["hydro"])
            interProvinModel.addConstr(rd["l2"][pro][h] <= scen_params['ramp']["l2"] * cap_gw[pro]["hydro"])
            interProvinModel.addConstr(ru["l3"][pro][h] <= scen_params['ramp']["l3"] * cap_gw[pro]["nuclear"])
            interProvinModel.addConstr(rd["l3"][pro][h] <= scen_params['ramp']["l3"] * cap_gw[pro]["nuclear"])

            for l in ['l2', 'l3']:
                interProvinModel.addConstr(
                    ru[l][pro][h] >=
                    load_conv[l][pro][h] + charge_phs[l][pro][h] + charge_bat[l][pro][h] +
                    gp.quicksum([charge_lds[st][l][pro][h] for st in lds]) + trans_out[l][pro][h] -
                    load_conv[l][pro][hour_pre[h]] - charge_phs[l][pro][hour_pre[h]] - charge_bat[l][pro][hour_pre[h]] -
                    gp.quicksum([charge_lds[st][l][pro][hour_pre[h]] for st in lds]) - trans_out[l][pro][hour_pre[h]])
                interProvinModel.addConstr(
                    rd[l][pro][h] >=
                    load_conv[l][pro][hour_pre[h]] + charge_phs[l][pro][hour_pre[h]] + charge_bat[l][pro][hour_pre[h]] +
                    gp.quicksum([charge_lds[st][l][pro][hour_pre[h]] for st in lds]) + trans_out[l][pro][hour_pre[h]] -
                    load_conv[l][pro][h] - charge_phs[l][pro][h] - charge_bat[l][pro][h] -
                    gp.quicksum([charge_lds[st][l][pro][h] for st in lds]) - trans_out[l][pro][h])

    # # Consider load shedding technology
    # if scen_params['shedding']['with_shedding'] == 1:
    #     with open(work_dir + 'data_pkl' + dir_flag + 'tot_dem2060.pkl', 'rb+') as fin:
    #         tot_dem = pickle.load(fin)
    #     fin.close()
    #
    #     # Total curtailed load is less than x% of average daily load
    #     interProvinModel.addConstr(
    #         gp.quicksum([load_shedding[pro][h] for pro in provins for h in hour_seed])
    #         <= year_count * (len(hour_seed) / 8760) * scen_params['shedding']['shedding_cof'] * tot_dem[
    #             'tot_dem'] / 365)

    # Consider medium-term policy goal for VRE buildout (1200 GW)
    if (scen_params["scenario"]["comply_with_medium_vre_goal"] == 1) and (curr_year == 2030):
        wind_installed_cap = [wind_cell['provin_cf_sort'][pro][c][3] * x_wind[pro][c]
                              for pro in provins for c in range(wind_cell_num[pro])]
        solar_installed_cap = [solar_cell['provin_cf_sort'][pro][c][3] * x_solar[pro][c]
                               for pro in provins for c in range(solar_cell_num[pro])]
        interProvinModel.addConstr(
            gp.quicksum(wind_installed_cap + solar_installed_cap) >= 1200
        )

    # Add 2030 policy goal for VRE buildout in "ndc" mode (2000 GW)
    if (scen_params["scenario"]["run_mode"] == "ndc") and (curr_year == 2030):
        wind_installed_cap = [wind_cell['provin_cf_sort'][pro][c][3] * x_wind[pro][c]
                              for pro in provins for c in range(wind_cell_num[pro])]
        offshore_installed_cap = [wind_cell['provin_cf_sort'][pro][c][3] * x_wind[pro][c]
                                  for pro in provins for c in range(wind_cell_num[pro])
                                  if wind_cell['provin_cf_sort'][pro][c][2] == 0]
        onshore_installed_cap = [wind_cell['provin_cf_sort'][pro][c][3] * x_wind[pro][c]
                                 for pro in provins for c in range(wind_cell_num[pro])
                                 if wind_cell['provin_cf_sort'][pro][c][2] == 1]
        solar_installed_cap = [solar_cell['provin_cf_sort'][pro][c][3] * x_solar[pro][c]
                               for pro in provins for c in range(solar_cell_num[pro])]
        upv_installed_cap = [solar_cell['provin_cf_sort'][pro][c][3] * x_solar[pro][c]
                             for pro in provins for c in range(solar_cell_num[pro])
                             if solar_cell['provin_cf_sort'][pro][c][2] == 0]
        dpv_installed_cap = [solar_cell['provin_cf_sort'][pro][c][3] * x_solar[pro][c]
                             for pro in provins for c in range(solar_cell_num[pro])
                             if solar_cell['provin_cf_sort'][pro][c][2] == 1]
        interProvinModel.addConstr(
            gp.quicksum(wind_installed_cap + solar_installed_cap) >= 2000
        )

        # Add 2024 EOY installations as the deployment lower bound
        interProvinModel.addConstr(gp.quicksum(offshore_installed_cap) >= 38)
        interProvinModel.addConstr(gp.quicksum(dpv_installed_cap) >= 280)
        interProvinModel.addConstr(gp.quicksum(cap_bat[pro] for pro in provins) >= 35.3)

        # interProvinModel.addConstr(gp.quicksum(onshore_installed_cap) >= 812)
        interProvinModel.addConstr(gp.quicksum(offshore_installed_cap) >= 109)
        # interProvinModel.addConstr(gp.quicksum(upv_installed_cap) >= 682)
        interProvinModel.addConstr(gp.quicksum(dpv_installed_cap) >= 468)
        interProvinModel.addConstr(gp.quicksum(cap_phs[pro] for pro in provins) >= 86)
        interProvinModel.addConstr(gp.quicksum(cap_bat[pro] for pro in provins) >= 154)

        # # Add 10% energy storage mandate at the provincial level
        # for pro in provins:
        #     interProvinModel.addConstr(
        #         cap_phs[pro] + cap_bat[pro] >=
        #         0.1 * (gp.quicksum([wind_cell['provin_cf_sort'][pro][c][3] * x_wind[pro][c]
        #                             for c in range(wind_cell_num[pro])]) +
        #                gp.quicksum([solar_cell['provin_cf_sort'][pro][c][3] * x_solar[pro][c]
        #                             for c in range(solar_cell_num[pro])])
        #                ))

    # Add coal emission targets as another constraint
    # BECCS + BIO + Gas Unabated/CCS + CHP_CCS + Coal Unabated/CCS emissions <= annual emission targets
    emission_scenario = scen_params["scenario"]["emission_target"]
    emission_factor_method = scen_params["scenario"]["emission_factor_method"]
    if scen_params["scenario"]["run_mode"] == "ndc":
        emission_targets = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions",
                                                    f"power_sector_emission_{emission_scenario}_ndc.csv"))
    else:
        emission_targets = pd.read_csv(os.path.join(work_dir, "data_csv", "capacity_assumptions",
                                                    f"power_sector_emission_{emission_scenario}.csv"))
    emission_factors = pd.read_csv(
        os.path.join(work_dir, "data_csv", "capacity_assumptions",
                     "coal_natural_retire", f"coal_{curr_year}_pre.csv"))
    # Emissions from BECCS and BIO
    emission_beccs, emission_bio = [], []
    for pro in provins:
        emission_beccs.append(
            8760 *  # annual run hours
            scen_params['beccs']['must_run'] *  # capacity factor
            cap_gw[pro]["beccs"] *  # BECCS capacity in GW at the provincial level
            0.88 *  # CCS capture rate
            -1.0 / 1e3  # kgCO2 per kWh
        )
        emission_bio.append(
            8760 *  # annual run hours
            scen_params['beccs']['must_run'] *  # capacity factor
            cap_gw[pro]["bio"] *  # BIO capacity in GW at the provincial level
            0.35 / 1e3  # kgCO2 per kWh
        )

    # Emissions from Gas (Unabated and Gas CCS)
    electricity_gas_unabated, electricity_gas_ccs = [], []
    emission_gas_unabated, emission_gas_ccs = [], []
    for pro in provins:
        # Obtain generated electricity data
        electricity_gas_unabated_value = gp.quicksum(
            [load_conv['l4_unabated'][pro][h] + trans_out['l4_unabated'][pro][h] for h in hour_seed])  # GWh
        electricity_gas_ccs_value = gp.quicksum(
            [load_conv['l4_ccs'][pro][h] + trans_out['l4_ccs'][pro][h] for h in hour_seed])  # GWh
        # Store electricity and emission results
        electricity_gas_unabated.append(electricity_gas_unabated_value)
        electricity_gas_ccs.append(electricity_gas_ccs_value)
        emission_gas_unabated.append(electricity_gas_unabated_value * 0.5 / 1e3)
        emission_gas_ccs.append(electricity_gas_ccs_value * 0.5 / 1e3 * (1 - 0.88))
    # Emissions from Coal (Coal Unabated, CCS, and CHP)
    electricity_coal_unabated, electricity_coal_ccs, electricity_chp = [], [], []
    emission_coal_unabated, emission_coal_ccs, emission_chp_ccs = [], [], []
    for pro in provins:
        # Obtain generated electricity data
        electricity_coal_unabated_value = gp.quicksum(
            [load_conv['l1_unabated'][pro][h] + trans_out['l1_unabated'][pro][h] for h in hour_seed])  # GWh
        electricity_coal_ccs_value = gp.quicksum(
            [load_conv['l1_ccs'][pro][h] + trans_out['l1_ccs'][pro][h] for h in hour_seed])  # GWh
        electricity_chp_value = gp.quicksum(
            [load_conv['l1_chp'][pro][h] + trans_out['l1_chp'][pro][h] for h in hour_seed])  # GWh
        # Read provincial coal plant emission factor
        cap_weighted_ef = emission_factors[emission_factors["province"] == pro]["cap_weighted_ef"].values[0]
        if cap_weighted_ef == 0:
            cap_weighted_ef = np.ma.average(emission_factors["cap_weighted_ef"],
                                            weights=emission_factors["cap_gw_ub"])
        # Store electricity and emission results
        electricity_coal_unabated.append(electricity_coal_unabated_value)
        electricity_coal_ccs.append(electricity_coal_ccs_value)
        electricity_chp.append(electricity_chp_value)
        emission_coal_unabated.append(electricity_coal_unabated_value * cap_weighted_ef / 1e3)
        emission_coal_ccs.append(electricity_coal_ccs_value * cap_weighted_ef / 1e3 * (1 - 0.88))
        emission_chp_ccs.append(electricity_chp_value * cap_weighted_ef / 1e3 * (1 - 0.88))

    # Total emissions must <= annual emission target
    annual_emission_target = emission_targets[emission_targets["Year"] == curr_year]["CO2_emission"].values[0]  # mt
    if (scen_params["scenario"]["ccs_start_year"] > curr_year) and (annual_emission_target < 0):
        # Update annual emission target (if negative) if no CCS is available
        annual_emission_target = 0
    interProvinModel.addConstr(
        gp.quicksum(emission_beccs) + gp.quicksum(emission_bio) +
        gp.quicksum(emission_gas_unabated) + gp.quicksum(emission_gas_ccs) +
        gp.quicksum(emission_coal_unabated) + gp.quicksum(emission_coal_ccs) + gp.quicksum(emission_chp_ccs)
        <= annual_emission_target)

    # For coal unabated, coal CCS, gas unabated, and gas CCS, minimum CF is 5% at each province
    min_cf = 0.05
    for i in range(len(provins)):
        pro = provins[i]
        interProvinModel.addConstr(
            cap_gw[pro]["coal_unabated"] * min_cf * scen_params["optimization_hours"]["days"] * 24 <=
            electricity_coal_unabated[i])
        interProvinModel.addConstr(
            cap_gw[pro]["coal_ccs"] * min_cf * scen_params["optimization_hours"]["days"] * 24 <=
            electricity_coal_ccs[i])
        interProvinModel.addConstr(
            cap_gw[pro]["chp_ccs"] * min_cf * scen_params["optimization_hours"]["days"] * 24 <=
            electricity_chp[i])
        interProvinModel.addConstr(
            cap_gw[pro]["gas_unabated"] * min_cf * scen_params["optimization_hours"]["days"] * 24 <=
            electricity_gas_unabated[i])
        interProvinModel.addConstr(cap_gw[pro]["gas_ccs"] * min_cf * scen_params["optimization_hours"]["days"] * 24 <=
                                   electricity_gas_ccs[i])

    # Solve the optimization model
    print('Start to solve...')
    interProvinModel.optimize()
    print('Print model attributes...')
    print('Total number of variables: ' + str(interProvinModel.numVars))
    print('Total number of integer variables: ' + str(interProvinModel.NumIntVars))
    print('Total number of binary variables: ' + str(interProvinModel.NumBinVars))
    print('Total number of constraints: ' + str(interProvinModel.NumConstrs))
    print('Total number of quadratic constraints: ' + str(interProvinModel.NumQConstrs))
    print('Total number of general constraints: ' + str(interProvinModel.NumGenConstrs))
    print('If the model is a MIP: ' + str(interProvinModel.IsMIP))
    print('If the model is a QP: ' + str(interProvinModel.IsQP))
    print('If the model is a QCP: ' + str(interProvinModel.IsQCP))

    # Write objective value
    f_objV = open(os.path.join(out_output_path, 'objValue.csv'), 'w+')
    f_objV.write('%s,%s,%s\n' % ("item", "million RMB", "Var/Constant"))
    f_objV.write('%s,%s\n' % ("objValue", np.round(interProvinModel.objVal, 2)))
    f_objV.write('%s,%s,%s\n' % ("wind_gen_cost", np.round(gp.quicksum(wind_gen_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("solar_gen_cost", np.round(gp.quicksum(solar_gen_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("offshore_cost_var", np.round(gp.quicksum(offshore_cost_var).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("onshore_cost_var", np.round(gp.quicksum(onshore_cost_var).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("upv_cost_var", np.round(gp.quicksum(upv_cost_var).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("dpv_cost_var", np.round(gp.quicksum(dpv_cost_var).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("ramp_up_cost", np.round(gp.quicksum(ramp_up_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("ramp_dn_cost", np.round(gp.quicksum(ramp_dn_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("fixed_phs_cost", np.round(gp.quicksum(fixed_phs_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("fixed_bat_cost", np.round(gp.quicksum(fixed_bat_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("fixed_lds_cost", np.round(gp.quicksum(fixed_lds_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("var_phs_cost", np.round(gp.quicksum(var_phs_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("var_bat_cost", np.round(gp.quicksum(var_bat_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("var_lds_cost", np.round(gp.quicksum(var_lds_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("resv_cost", np.round(gp.quicksum(resv_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("spur_cost_wind", np.round(gp.quicksum(spur_cost_wind).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("spur_cost_solar", np.round(gp.quicksum(spur_cost_solar).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("trunk_cost", np.round(gp.quicksum(trunk_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("trans_cost", np.round(gp.quicksum(trans_cost).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("beccs_cost_fixed", np.round(gp.quicksum(beccs_cost_fixed).getValue(), 2), "Const"))
    f_objV.write(
        '%s,%s,%s\n' % ("bio_cost_var_must_run", np.round(gp.quicksum(bio_cost_var_must_run).getValue(), 2), "Const"))
    f_objV.write('%s,%s,%s\n' % ("hydro_cost_fixed", np.round(gp.quicksum(hydro_cost_fixed).getValue(), 2), "Const"))
    f_objV.write('%s,%s,%s\n' % ("hydro_cost_var", np.round(gp.quicksum(hydro_cost_var).getValue(), 2), "Var"))
    f_objV.write(
        '%s,%s,%s\n' % ("nuclear_cost_fixed", np.round(gp.quicksum(nuclear_cost_fixed).getValue(), 2), "Const"))
    f_objV.write('%s,%s,%s\n' % ("nuclear_cost_var", np.round(gp.quicksum(nuclear_cost_var).getValue(), 2), "Var"))
    f_objV.write(
        '%s,%s,%s\n' % ("coal_ccs_cost_fixed", np.round(gp.quicksum(coal_ccs_cost_fixed).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % (
    "coal_unabated_cost_fixed", np.round(gp.quicksum(coal_unabated_cost_fixed).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("coal_cost_var", np.round(gp.quicksum(coal_cost_var).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("gas_cost_fixed", np.round(gp.quicksum(gas_ccs_cost_fixed).getValue(), 2), "Var"))
    f_objV.write(
        '%s,%s,%s\n' % ("gas_unabated_cost_fixed", np.round(gp.quicksum(gas_unabated_cost_fixed).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("gas_cost_var", np.round(gp.quicksum(gas_cost_var).getValue(), 2), "Var"))
    f_objV.write('%s,%s,%s\n' % ("other_tech_cost", np.round(gp.quicksum(other_tech_cost).getValue(), 2), "Const"))
    f_objV.close()

    # Write emission results
    f_emisV = open(os.path.join(out_output_path, 'emissionValue.csv'), 'w+')
    f_emisV.write('%s,%s\n' % ("emission_target_mt", np.round(annual_emission_target, 2)))
    f_emisV.write('%s,%s\n' % ("emission_beccs_mt", np.round(gp.quicksum(emission_beccs).getValue(), 2)))
    f_emisV.write('%s,%s\n' % ("emission_bio_mt", np.round(gp.quicksum(emission_bio).getValue(), 2)))
    f_emisV.write('%s,%s\n' % ("emission_gas_unabated_mt", np.round(gp.quicksum(emission_gas_unabated).getValue(), 2)))
    f_emisV.write('%s,%s\n' % ("emission_gas_ccs_mt", np.round(gp.quicksum(emission_gas_ccs).getValue(), 2)))
    f_emisV.write(
        '%s,%s\n' % ("emission_coal_unabated_mt", np.round(gp.quicksum(emission_coal_unabated).getValue(), 2)))
    f_emisV.write('%s,%s\n' % ("emission_coal_ccs_mt", np.round(gp.quicksum(emission_coal_ccs).getValue(), 2)))
    f_emisV.write('%s,%s\n' % ("emission_chp_ccs_mt", np.round(gp.quicksum(emission_chp_ccs).getValue(), 2)))
    f_emisV.close()

    # Write provincial breakdowns of emissions and capacity factors
    emission_breakdown = pd.DataFrame(
        columns=["province",
                 "cap_coal_unabated_gw", "cap_coal_ccs_gw", "cap_chp_ccs_gw", "cap_beccs_gw", "cap_bio_gw",
                 "cap_hydro_gw", "cap_nuclear_gw", "cap_gas_unabated_gw", "cap_gas_ccs_gw",
                 "electricity_coal_unabated", "electricity_coal_ccs", "electricity_chp_ccs",
                 "electricity_gas_unabated", "electricity_gas_ccs",
                 "electricity_onshore", "electricity_offshore", "electricity_utility", "electricity_distributed",
                 "cf_coal_unabated", "cf_coal_ccs", "cf_chp_ccs", "cf_gas_unabated", "cf_gas_ccs",
                 "emission_beccs_mt", "emission_bio_mt",
                 "emission_coal_unabated_mt", "emission_coal_ccs_mt", "emission_chp_ccs_mt",
                 "emission_gas_unabated_mt", "emission_gas_ccs_mt"])
    emission_breakdown["province"] = provins
    for i in range(len(provins)):
        pro = provins[i]
        for tech in ["coal_unabated", "coal_ccs", "chp_ccs", "hydro", "nuclear",
                     "beccs", "bio", "gas_unabated", "gas_ccs"]:
            emission_breakdown.loc[i, f"cap_{tech}_gw"] = np.round(cap_gw[pro][tech].x, 2)

        emission_breakdown.loc[i, "electricity_coal_unabated"] = np.round(electricity_coal_unabated[i].getValue(), 2)
        emission_breakdown.loc[i, "electricity_coal_ccs"] = np.round(electricity_coal_ccs[i].getValue(), 2)
        emission_breakdown.loc[i, "electricity_chp_ccs"] = np.round(electricity_chp[i].getValue(), 2)
        emission_breakdown.loc[i, "electricity_gas_unabated"] = np.round(electricity_gas_unabated[i].getValue(), 2)
        emission_breakdown.loc[i, "electricity_gas_ccs"] = np.round(electricity_gas_ccs[i].getValue(), 2)

        emission_breakdown.loc[i, "electricity_onshore"] = np.round(onshore_gen[pro].getValue(), 2)
        emission_breakdown.loc[i, "electricity_offshore"] = np.round(offshore_gen[pro].getValue(), 2)
        emission_breakdown.loc[i, "electricity_utility"] = np.round(upv_gen[pro].getValue(), 2)
        emission_breakdown.loc[i, "electricity_distributed"] = np.round(dpv_gen[pro].getValue(), 2)

        emission_breakdown.loc[i, "emission_beccs_mt"] = np.round(emission_beccs[i].getValue(), 2)
        emission_breakdown.loc[i, "emission_bio_mt"] = np.round(emission_bio[i].getValue(), 2)
        emission_breakdown.loc[i, "emission_coal_unabated_mt"] = np.round(emission_coal_unabated[i].getValue(), 2)
        emission_breakdown.loc[i, "emission_coal_ccs_mt"] = np.round(emission_coal_ccs[i].getValue(), 2)
        emission_breakdown.loc[i, "emission_chp_ccs_mt"] = np.round(emission_chp_ccs[i].getValue(), 2)
        emission_breakdown.loc[i, "emission_gas_unabated_mt"] = np.round(emission_gas_unabated[i].getValue(), 2)
        emission_breakdown.loc[i, "emission_gas_ccs_mt"] = np.round(emission_gas_ccs[i].getValue(), 2)

    for tech in ["coal_unabated", "coal_ccs", "chp_ccs", "gas_unabated", "gas_ccs"]:
        emission_breakdown[f"cf_{tech}"] = emission_breakdown[f"electricity_{tech}"] / \
                                           emission_breakdown[f"cap_{tech}_gw"] / \
                                           (scen_params["optimization_hours"]["days"] * 24)

    emission_breakdown.round(2).to_csv(os.path.join(out_output_path, 'emissionBreakdowns.csv'))

    # Write model results
    print("Writing optimization results to files...")

    folder_shadow_price = makeDir(os.path.join(out_output_path, 'shadow_prices'))
    for pro in demConstrs:
        f_sp = open(os.path.join(folder_shadow_price, pro + '.csv'), 'w+')
        shadow_prices = interProvinModel.getAttr('Pi', demConstrs[pro])
        for h in shadow_prices:
            f_sp.write(str(h) + ',' + str(shadow_prices[h]) + '\n')
        f_sp.close()

    for l in load_conv:
        folder = makeDir(os.path.join(out_output_path, 'load_conv', l))
        for pro in load_conv[l]:
            res_load_conv = open(os.path.join(folder, pro + '.csv'), 'w+')
            for h in load_conv[l][pro]:
                res_load_conv.write('%s,%s\n' % (h, load_conv[l][pro][h].x))
            res_load_conv.close()

    for l in load_resv:
        folder = makeDir(os.path.join(out_output_path, 'load_resv', l))
        for pro in load_resv[l]:
            res_load_resv = open(os.path.join(folder, pro + '.csv'), 'w+')
            for h in load_resv[l][pro]:
                res_load_resv.write('%s,%s\n' % (h, load_resv[l][pro][h].x))
            res_load_resv.close()

    if scen_params['shedding']['with_shedding'] == 1:
        folder = makeDir(os.path.join(out_output_path, 'load_shedding'))
        for pro in load_shedding:
            res_load_shedding = open(os.path.join(folder, pro + '.csv'), 'w+')
            for h in load_shedding[pro]:
                res_load_shedding.write('%s,%s\n' % (h, load_shedding[pro][h].x))
            res_load_shedding.close()

    cell_cof_dict = {'x_wind': x_wind, 'x_solar': x_solar}
    for file in cell_cof_dict:
        folder = makeDir(os.path.join(out_output_path, file))
        for pro in cell_cof_dict[file]:
            res_cell_cof = open(os.path.join(folder, pro + '.csv'), 'w+')
            for c in cell_cof_dict[file][pro]:
                res_cell_cof.write('%s,%s\n' % (c, cell_cof_dict[file][pro][c].x))
            res_cell_cof.close()

    inte_dict = {'inte_wind': inte_wind, 'inte_solar': inte_solar}
    for file in inte_dict:
        folder = makeDir(os.path.join(out_output_path, file))
        for pro in inte_dict[file]:
            res_inte = open(os.path.join(folder, pro + '.csv'), 'w+')
            for i in inte_dict[file][pro]:
                res_inte.write('%s,%s\n' % (i, inte_dict[file][pro][i].x))
            res_inte.close()

    es_cap = {'phs': cap_phs, 'bat': cap_bat}
    es_cap_count = {'phs': 0, 'bat': 0, 'caes': 0, 'vrb': 0}
    for file in es_cap:
        folder = makeDir(os.path.join(out_output_path, 'es_cap'))
        res_store = open(os.path.join(folder, 'es_' + file + '_cap.csv'), 'w+')
        for pro in es_cap[file]:
            res_store.write('%s,%s\n' % (pro, es_cap[file][pro].x))
            es_cap_count[file] += es_cap[file][pro].x
        res_store.close()

    for file in cap_lds:
        folder = makeDir(os.path.join(out_output_path, 'es_cap'))
        res_store = open(os.path.join(folder, 'es_' + file + '_cap.csv'), 'w+')
        for pro in cap_lds[file]:
            res_store.write('%s,%s\n' % (pro, cap_lds[file][pro].x))
            es_cap_count[file] += cap_lds[file][pro].x
        res_store.close()

    f_es_tot_cap = open(os.path.join(out_output_path, 'es_cap', 'es_tot_cap.csv'), 'w+')
    for st in es_cap_count:
        f_es_tot_cap.write(st + ',' + str(es_cap_count[st]) + '\n')
    f_es_tot_cap.close()

    es_char = {'phs': charge_phs, 'bat': charge_bat}
    for file in es_char:
        for et in es_char[file]:
            folder = makeDir(os.path.join(out_output_path, 'es_char', file, et))
            for pro in es_char[file][et]:
                res_es_char = open(os.path.join(folder, pro + '.csv'), 'w+')
                for h in es_char[file][et][pro]:
                    res_es_char.write('%s,%s\n' % (h, es_char[file][et][pro][h].x))
                res_es_char.close()

    for file in charge_lds:
        for et in charge_lds[file]:
            folder = makeDir(os.path.join(out_output_path, 'es_char', file, et))
            for pro in charge_lds[file][et]:
                res_es_char = open(os.path.join(folder, pro + '.csv'), 'w+')
                for h in charge_lds[file][et][pro]:
                    res_es_char.write('%s,%s\n' % (h, charge_lds[file][et][pro][h].x))
                res_es_char.close()

    es_inte = {'phs': dischar_phs, 'bat': dischar_bat}
    rtrip_efi = {'phs': rt_effi_phs, 'bat': rt_effi_bat}

    for file in es_inte:
        folder = makeDir(os.path.join(out_output_path, 'es_inte', file))
        for pro in es_inte[file]:
            res_es_inte = open(os.path.join(folder, pro + '.csv'), 'w+')
            for h in es_inte[file][pro]:
                res_es_inte.write('%s,%s\n' % (h, rtrip_efi[file] * es_inte[file][pro][h].x))
            res_es_inte.close()

    for file in dischar_lds:
        folder = makeDir(os.path.join(out_output_path, 'es_inte', file))
        for pro in dischar_lds[file]:
            res_es_inte = open(os.path.join(folder, pro + '.csv'), 'w+')
            for h in dischar_lds[file][pro]:
                res_es_inte.write('%s,%s\n' % (h, rt_effi_lds[file] * dischar_lds[file][pro][h].x))
            res_es_inte.close()

    folder = makeDir(os.path.join(out_output_path, 'resv_phs'))
    for pro in resv_phs:
        res_resv_phs = open(os.path.join(folder, pro + '.csv'), 'w+')

        for h in resv_phs[pro]:
            res_resv_phs.write('%s,%s\n' % (h, resv_phs[pro][h].x))

        res_resv_phs.close()

    for et in trans_out:
        folder = makeDir(os.path.join(out_output_path, 'trans_out', et))
        for pro in trans_out[et]:
            f_trans_out = open(os.path.join(folder, pro + '.csv'), 'w+')

            for h in trans_out[et][pro]:
                f_trans_out.write('%s,%s\n' % (h, trans_out[et][pro][h].x))
            f_trans_out.close()

    folder = makeDir(os.path.join(out_output_path, 'resv_bat'))
    for pro in resv_bat:
        res_resv_bat = open(os.path.join(folder, pro + '.csv'), 'w+')

        for h in resv_bat[pro]:
            res_resv_bat.write('%s,%s\n' % (h, resv_bat[pro][h].x))
        res_resv_bat.close()

    for st in resv_lds:
        folder = makeDir(os.path.join(out_output_path, 'resv_' + st))
        for pro in resv_lds[st]:
            res_resv_lds = open(os.path.join(folder, pro + '.csv'), 'w+')

            for h in resv_lds[st][pro]:
                res_resv_lds.write('%s,%s\n' % (h, resv_lds[st][pro][h].x))
            res_resv_lds.close()

    folder = makeDir(os.path.join(out_output_path, 'load_trans'))
    for pro in load_trans:
        res_trans = open(os.path.join(folder, pro[0] + '_' + pro[1] + '.csv'), 'w+')
        for h in load_trans[pro]:
            res_trans.write('%s,%s\n' % (h, load_trans[pro][h].x))
        res_trans.close()

    es_tot = {'phs': tot_energy_phs, 'bat': tot_energy_bat}

    for file in es_tot:
        folder = makeDir(os.path.join(out_output_path, 'es_tot', file))
        for pro in es_tot[file]:
            res_es_tot = open(os.path.join(folder, pro + '.csv'), 'w+')
            for h in es_tot[file][pro]:
                res_es_tot.write('%s,%s\n' % (h, es_tot[file][pro][h].x))
            res_es_tot.close()

    for file in tot_energy_lds:
        folder = makeDir(os.path.join(out_output_path, 'es_tot', file))
        for pro in tot_energy_lds[file]:
            res_es_tot = open(os.path.join(folder, pro + '.csv'), 'w+')
            for h in tot_energy_lds[file][pro]:
                res_es_tot.write('%s,%s\n' % (h, tot_energy_lds[file][pro][h].x))
            res_es_tot.close()

    folder = makeDir(os.path.join(out_output_path, 'new_trans_cap'))
    res_new_cap_trans = open(os.path.join(folder, 'cap_trans_new.csv'), 'w+')
    for pro in cap_trans_new:
        res_new_cap_trans.write('%s,%s,%s\n' % (pro[0], pro[1], cap_trans_new[pro].x))
    res_new_cap_trans.close()

    # Combine l1 and l4 output if endogenous
    for item in ["load_conv", "load_resv", "trans_out"]:
        folder = makeDir(os.path.join(out_output_path, item, "l1"))
        for pro in provins:
            l1_unabated = pd.read_csv(os.path.join(out_output_path, item, "l1_unabated", f"{pro}.csv"))
            l1_ccs = pd.read_csv(os.path.join(out_output_path, item, "l1_ccs", f"{pro}.csv"))
            l1_chp = pd.read_csv(os.path.join(out_output_path, item, "l1_chp", f"{pro}.csv"))
            l1 = pd.DataFrame()
            l1["hour"] = hour_seed
            l1["power"] = l1_unabated.iloc[:, 1] + l1_ccs.iloc[:, 1] + l1_chp.iloc[:, 1]
            l1.to_csv(os.path.join(folder, f"{pro}.csv"), index=False, header=False)

        folder = makeDir(os.path.join(out_output_path, item, "l4"))
        for pro in provins:
            l4_unabated = pd.read_csv(os.path.join(out_output_path, item, "l4_unabated", f"{pro}.csv"))
            l4_ccs = pd.read_csv(os.path.join(out_output_path, item, "l4_ccs", f"{pro}.csv"))
            l4 = pd.DataFrame()
            l4["hour"] = hour_seed
            l4["power"] = l4_unabated.iloc[:, 1] + l4_ccs.iloc[:, 1]
            l4.to_csv(os.path.join(folder, f"{pro}.csv"), index=False, header=False)

    # Process optimization results
    print('Processing optimization results...')


if __name__ == '__main__':
    """
    Before running the test cases below, make sure that the input files exist.
    If not, use initData.py to initialize input files required for optimizations.
    """

    # Specify scenario parameters
    work_dir = getWorkDir()
    with open(os.path.join(work_dir, "data_csv", "scen_params_template.json")) as f:
        scen_params = json.load(f)

    # Run single-year test case
    interProvinModel(vre_year="w2015_s2015",  # which year of solar and wind data to use
                     res_tag="test_0708_5days_2050only",  # the folder name
                     init_data=0,  # if data will be re-initialized; 0 = we are using pre-initialized data
                     is8760=0,  # if we are optimizing for the entire year
                     curr_year=2050,  # which year we are optimizing for
                     scen_params=scen_params  # scenario parameters
                     )
