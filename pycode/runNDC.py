import datetime
import os
import json

import pandas as pd
import numpy as np

from callUtility import getWorkDir
from multiYearAutomation import MultiYearAutomation
from initData import seedHour, initCellData, initDemLayer, initModelExovar
from main import interProvinModel
from clearupData import cellResInfo, TransCap, update_storage_capacity, TransInfo, curtailed, \
    obtain_output_summary, obtain_simulation_summary

# # Step 0: Set parameters (if run remotely using shell scripts)
# optimization_days = int(sys.argv[1])  # 365
# start_year = int(sys.argv[2])  # 2050
# year_count = int(sys.argv[3])  # 1
# node = str(sys.argv[4])  # "home"
# emission_target = str(sys.argv[5])  # "2C"
# ccs_start_year = int(sys.argv[6])  # 2040
# heating_electrification = str(sys.argv[7])  # "chp_ccs"
# renewable_cost_decline = str(sys.argv[8])  # "baseline"
# endogenize_firm_capacity = str(sys.argv[9])  # 1
# demand_sensitivity = str(sys.argv[10])  # "none"
# ccs_retrofit_cost = float(sys.argv[11])  # 3500
# run_mode = str(sys.argv[12])  # "ndc"
# vre_year = str(sys.argv[13])  # "w2024_s2024"
# ptc_mode = str(sys.argv[14])  # ["none", "uniform", "provincial"]
# custom_tag = ""

# Step 0: Set parameters (if run locally on your laptop)
optimization_days = 7
start_year = 2030
year_count = 2  # [1, 2, 3, 4]
node = "pc"
emission_target = "2C"  # ["2C", "15C"]
ccs_start_year = 2040  # [2040, 2050, 2060, 2070]
heating_electrification = "heat_pump"  # ["chp_ccs", "heat_pump"]
renewable_cost_decline = "baseline"  # ["baseline", "conservative", "low_wind", "low_bat"]
endogenize_firm_capacity = 0  # [0, 1]
demand_sensitivity = "none"  # ["none", "2030b_2035p10", "2030b_2035m10",
    # "2030m10_2035b", "2030m10_2035p10", "2030m10_2035m10"]
ccs_retrofit_cost = float(3500)  # 3500 ~ 10000
run_mode = "ndc"  # ["ndc"]
vre_year = "w2015_s2015"
ptc_mode = "none"
custom_tag = ""

year_list = [2030, 2035]

# Read parameters
date = datetime.date.today().strftime("%m%d")
if year_count == 1:
    res_tag = f"test_{date}_{optimization_days}days_{str(start_year)}only_{node}_{custom_tag}"
else:
    res_tag = f"test_{date}_{optimization_days}days_all_years_{node}"

optimization_step = 1
year_start = 2025
year_end = 2060
year_step = 5

# Step 1: generate multi-year inputs
multiyear_input = MultiYearAutomation(yr_start=year_start,
                                      yr_end=year_end,
                                      yr_step=year_step,
                                      res_tag=res_tag,
                                      vre_year=vre_year,
                                      emission_target=emission_target,
                                      demand_sensitivity=demand_sensitivity,
                                      run_mode=run_mode
                                      )
multiyear_input.automate_inputs()

# Step 2: read the template of scenario parameters
work_dir = getWorkDir()
with open(os.path.join(work_dir, "data_csv", "scen_params_template.json")) as f:
    scen_params = json.load(f)
f.close()

# Step 3: initialize exogenous parameters
for idx in range(len(multiyear_input.yr_req)):
    # Specify parameters
    curr_year = multiyear_input.yr_req[idx]
    last_year = multiyear_input.yr_req[idx - 1] if idx != 0 else 2020

    # Specify folder paths
    out_input_path = os.path.join(multiyear_input.out_path, str(curr_year), "inputs")

    # 3.1 Update and save input scenario parameters
    scen_params["scenario"]["comply_with_medium_vre_goal"] = 1
    scen_params["scenario"]["endogenize_firm_capacity"] = 0 if curr_year == 2030 else 1
    scen_params["scenario"]["ccs_start_year"] = ccs_start_year
    scen_params["scenario"]["emission_target"] = emission_target
    scen_params["scenario"]["heating_electrification"] = heating_electrification
    scen_params["scenario"]["renewable_cost_decline"] = renewable_cost_decline
    scen_params["scenario"]["demand_sensitivity"] = demand_sensitivity
    scen_params["scenario"]["run_mode"] = run_mode
    scen_params["scenario"]["ptc_mode"] = ptc_mode
    scen_params["scenario"]["emission_factor_method"] = "mean"
    scen_params["ccs"]["capex_coal_ccs"] = ccs_retrofit_cost
    scen_params["ccs"]["capex_gas_ccs"] = ccs_retrofit_cost

    # Baseline cost decline assumptions
    capex_on_wind_list = np.linspace(7700, 3000, len(multiyear_input.yr_req) + 1)
    capex_off_wind_list = np.linspace(15000, 5400, len(multiyear_input.yr_req) + 1)
    capex_pv_list = np.linspace(5300, 1500, len(multiyear_input.yr_req) + 1)
    capex_dpv_list = np.linspace(5300, 2000, len(multiyear_input.yr_req) + 1)
    capex_bat_list = np.linspace(6000, 2700, len(multiyear_input.yr_req) + 1)
    capex_caes_list = np.linspace(24000, 4800, len(multiyear_input.yr_req) + 1)
    capex_vrb_list = np.linspace(25600, 3000, len(multiyear_input.yr_req) + 1)

    # Update cost decline assumptions based on scenario names
    if curr_year == 2035:
        if scen_params["scenario"]["renewable_cost_decline"] == "conservative":
            # Conservative cost declines based on NREL projections (2023)
            capex_on_wind_list = [6525] * (len(multiyear_input.yr_req) + 1)
            capex_off_wind_list = [12600] * (len(multiyear_input.yr_req) + 1)
            capex_pv_list = [4350] * (len(multiyear_input.yr_req) + 1)
            capex_dpv_list = [4475] * (len(multiyear_input.yr_req) + 1)
            capex_bat_list = [5175] * (len(multiyear_input.yr_req) + 1)
            capex_caes_list = [19200] * (len(multiyear_input.yr_req) + 1)
            capex_vrb_list = [19950] * (len(multiyear_input.yr_req) + 1)
        elif scen_params["scenario"]["renewable_cost_decline"] == "low_wind":
            capex_on_wind_list = np.linspace(6000, 3000, len(multiyear_input.yr_req) + 1)
            capex_off_wind_list = np.linspace(12000, 5400, len(multiyear_input.yr_req) + 1)
        elif scen_params["scenario"]["renewable_cost_decline"] == "low_bat":
            capex_bat_list = np.linspace(4000, 2700, len(multiyear_input.yr_req) + 1)
    else:
        pass

    scen_params["vre"]["capex_equip_on_wind"] = capex_on_wind_list[idx + 1] - 800
    scen_params["vre"]["capex_om_on_wind"] = np.linspace(170, 45, len(multiyear_input.yr_req) + 1)[idx + 1]
    scen_params["vre"]["capex_equip_off_wind"] = capex_off_wind_list[idx + 1] - 1600
    scen_params["vre"]["capex_om_off_wind"] = np.linspace(715, 81, len(multiyear_input.yr_req) + 1)[idx + 1]
    scen_params["vre"]["capex_equip_pv"] = capex_pv_list[idx + 1] - 400
    scen_params["vre"]["capex_om_pv"] = np.linspace(85, 7.5, len(multiyear_input.yr_req) + 1)[idx + 1]
    scen_params["vre"]["capex_equip_dpv"] = capex_dpv_list[idx + 1] - 600
    scen_params["vre"]["capex_om_dpv"] = np.linspace(107, 10, len(multiyear_input.yr_req) + 1)[idx + 1]
    scen_params['storage']['capex_power_phs'] = np.linspace(3840, 3840, len(multiyear_input.yr_req) + 1)[idx + 1]
    scen_params['storage']['capex_power_bat'] = capex_bat_list[idx + 1]
    scen_params['storage']['capex_power_lds']['caes'] = capex_caes_list[idx + 1]
    scen_params['storage']['capex_power_lds']['vrb'] = capex_vrb_list[idx + 1]

    # 3.2 Initialize optimization hours
    scen_params["optimization_hours"]["step"] = optimization_step
    scen_params["optimization_hours"]["days"] = optimization_days
    seedHour(vre_year=vre_year,
             years=scen_params["optimization_hours"]["years"],
             step=scen_params["optimization_hours"]["step"],
             days=scen_params["optimization_hours"]["days"],
             res_tag=res_tag, curr_year=curr_year)
    hour_seed = pd.read_csv(os.path.join(out_input_path, "hour_seed.csv"), header=None).iloc[:, 0].to_list()

    # Save the updated scenario parameters
    with open(os.path.join(out_input_path, "scen_params.json"), 'w') as fp:
        json.dump(scen_params, fp)

    # 3.3 Initialize demand layers
    initDemLayer(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year, scen_params=scen_params)

# Step 4: initialize parameters that have endogenous elements
# Only run optimization per 10 or 5 years, hence the break-up from previous steps
for idx in range(len(year_list)):
    # Specify parameters
    curr_year = year_list[idx]
    last_year = curr_year - 5

    # Specify folder paths
    out_input_path = os.path.join(multiyear_input.out_path, str(curr_year), "inputs")

    # Read scenario parameters and optimization hours
    hour_seed = pd.read_csv(os.path.join(out_input_path, "hour_seed.csv"), header=None).iloc[:, 0].to_list()
    with open(os.path.join(out_input_path, "scen_params.json")) as f:
        scen_params = json.load(f)

    # 4.1 Initialize VRE cell data
    initCellData(vre="wind", vre_year_single=vre_year.split("_")[0][1:], hour_seed=hour_seed, res_tag=res_tag,
                 vre_year=vre_year, curr_year=curr_year, last_year=last_year, scen_params=scen_params)
    initCellData(vre="solar", vre_year_single=vre_year.split("_")[1][1:], hour_seed=hour_seed, res_tag=res_tag,
                 vre_year=vre_year, curr_year=curr_year, last_year=last_year, scen_params=scen_params)

    # 4.2 Initialize layer capacity
    initModelExovar(vre_year=vre_year, res_tag=res_tag,
                    curr_year=curr_year, last_year=last_year, scen_params=scen_params)

    # Step 5: run optimization for the current year
    interProvinModel(vre_year=vre_year, res_tag=res_tag, init_data=0, is8760=0,
                     curr_year=curr_year, scen_params=scen_params)

    # Step 6: post-process optimization outputs
    cellResInfo(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    TransCap(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    update_storage_capacity(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    TransInfo(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)
    curtailed(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year, re="wind")
    curtailed(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year, re="solar")

    # Step 7: obtain provincial and national output summary
    obtain_output_summary(vre_year=vre_year, res_tag=res_tag, curr_year=curr_year)

# Step 8: obtain provincial and national output summary for the whole simulation years
obtain_simulation_summary(vre_year=vre_year, res_tag=res_tag, year_list=year_list)

# Step 9: remove excessive huge files
for idx in range(len(multiyear_input.yr_req)):
    try:
        # Specify folder paths
        curr_year = multiyear_input.yr_req[idx]
        out_input_path = os.path.join(multiyear_input.out_path, str(curr_year), "inputs")
        print(out_input_path)

        # Delete VRE cell pickle files due to storage space limits
        os.remove(os.path.join(out_input_path, "solar_cell.pkl"))
        os.remove(os.path.join(out_input_path, "wind_cell.pkl"))
    except:
        pass
