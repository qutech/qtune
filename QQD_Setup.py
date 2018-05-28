#%% Test script for autotuner on dqd1
from qtune import sm
from qtune import sm_tune_qqd
from qtune.experiment import Measurement


# Start/Connect to MATLB
# Make sure SpecialMeasure and Tunedata are ready to go
matlab = sm.SpecialMeasureMatlab(connect='MATLAB_7144')
# TODO check for global vars
# matlab.engine.local_tune_setup(nargout=0)

# create QQd Instance
qqd = sm_tune_qqd.SMTuneQQD(matlab)

#%% Setup Measurements
dqd1_meas_line = Measurement('line', index=1,
                             loadFile=r'\\Janeway\User AG Bluhm\Common\GaAs\Triton 200\Backup\DATA\\tune\\run_0001\line\sm_line_3_2018_05_09_16_02_05.mat')

dqd1_meas_lead = Measurement('lead', index=1,
                             loadFile=r'\\Janeway\User AG Bluhm\Common\GaAs\Triton 200\Backup\DATA\\tune\\run_0001\lead\sm_lead_1_2018_05_25_16_22_59.mat')

dqd1_meas_chrg = Measurement('chrg', index=1,
                             loadFile=r'\\Janeway\User AG Bluhm\Common\GaAs\Triton 200\Backup\DATA\\tune\\run_0001\chrg\sm_chrg_1_2018_05_28_01_45_09.mat')

dqd1_meas_sensor = Measurement('sensor', index=1,
                               loadFile=r'\\Janeway\User AG Bluhm\Common\GaAs\Triton 200\Backup\DATA\\tune\\run_0001\sensor\sm_sensor_1_LB_LT_2018_03_29_00_01_56_1_1.mat',
                               changeGateVoltages=0)

dqd1_meas_sensor_2d = Measurement('sensor 2d', index=1,
                                  loadFile=r'\\Janeway\User AG Bluhm\Common\GaAs\Triton 200\Backup\DATA\\tune\\run_0001\sensor_2d\sm_sensor_2d_1_LB_LT_2018_03_29_00_01_38_1_1.mat',
                                  changeGateVoltages=0)

dqd1_meas_stp = Measurement('stp', index=1,
                            loadFile=r'\\Janeway\User AG Bluhm\Common\GaAs\Triton 200\Backup\DATA\\tune\\run_0001\stp\sm_stp_1_2018_05_25_19_23_31.mat')

dqd1_meas_tl = Measurement('tl', index=1,
                           loadFile=r'\\Janeway\User AG Bluhm\Common\GaAs\Triton 200\Backup\DATA\tune\run_0001\tl\sm_tl_1_2018_05_25_18_46_51.mat')



#%% Test Measurements
ret = qqd.measure(dqd1_meas_line)
ret = qqd.measure(dqd1_meas_lead)
ret = qqd.measure(dqd1_meas_chrg)
#ret = qqd.measure(dqd1_meas_sensor)
#ret = qqd.measure(dqd1_meas_sensor_2d)
ret = qqd.measure(dqd1_meas_stp)
ret = qqd.measure(dqd1_meas_tl)

#%% Create Evaluators
dqd1_tunnel_coupling = sm_tune_qqd.SMQQDPassThru(experiment=qqd,
                                    measurements=[dqd1_meas_line],
                                    parameters=['tunnel_coupling'])

dqd1_lead_time = sm_tune_qqd.SMQQDPassThru(experiment=qqd,
                                    measurements=[dqd1_meas_lead],
                                    parameters=['lead time'])

dqd1_origin = sm_tune_qqd.SMQQDPassThru(experiment=qqd,
                                    measurements=[dqd1_meas_chrg],
                                    parameters=['origin'])

dqd1_stp = sm_tune_qqd.SMQQDPassThru(experiment=qqd,
                                    measurements=[dqd1_meas_stp],
                                    parameters=['stp point'])

dqd1_tl = sm_tune_qqd.SMQQDPassThru(experiment=qqd,
                                    measurements=[dqd1_meas_tl],
                                    parameters=['tl point'])


#%% Test Evaluators
dqd1_tunnel_coupling.evaluate()
dqd1_lead_time.evaluate()
dqd1_origin.evaluate()
dqd1_stp.evaluate()
dqd1_tl.evaluate()

#%% Setup autotuner