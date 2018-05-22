#%%
from qtune import sm
from qtune import sm_tune_qqd
from qtune.experiment import Measurement


# Start/Connect to MATLB
# Make sure SpecialMeasure and Tunedata are ready to go
matlab = sm.SpecialMeasureMatlab()
# TODO check for global vars

# create QQd Instance
qqd = sm_tune_qqd.SMTuneQQD(matlab)
#%%
# Test measurements
qqd.measure(Measurement('line',index=1))
qqd.measure(Measurement('lead',index=1))
qqd.measure(Measurement('resp',index=1))
qqd.measure(Measurement('chrg',index=1))
qqd.measure(Measurement('sensor',index=1))

#%%
# Create Evaluators for sensor 1
sensor1 = sm_tune_qqd.SMQQDSensor(experiment=qqd, measurements=(Measurement('sensor',index=1),) )
sensor1_2d = sm_tune_qqd.SMQQDSensor2d(experiment=qqd, measurements=(Measurement('sensor_2d',index=1),) )

# Create Evaluators for sensor 2
sensor2 = sm_tune_qqd.SMQQDSensor(experiment=qqd, measurements=(Measurement('sensor',index=2),) )
sensor2_2d = sm_tune_qqd.SMQQDSensor2d(experiment=qqd, measurements=(Measurement('sensor_2d',index=2),) )

# Create evaluators for dots 1 and 2
dqd1_tunnel_coupling = sm_tune_qqd.SMQQDLineScan(experiment=qqd, measurements=(Measurement('line',index=1),) )
dqd1_lead_A = sm_tune_qqd.SMQQDLeadScan(experiment=qqd, measurements=(Measurement('lead',index=1),) )
dqd1_lead_B = sm_tune_qqd.SMQQDLeadScan(experiment=qqd, measurements=(Measurement('lead',index=1),) )
dqd1_chrg = sm_tune_qqd.SMQQDChrgScan(experiment=qqd, measurements=(Measurement('chrg',index=1),) )

# Create evaluators for dots 2 and 3
dqd2_tunnel_coupling = sm_tune_qqd.SMQQDLineScan(experiment=qqd, measurements=(Measurement('line',index=2),) )
dqd2_lead_A = sm_tune_qqd.SMQQDLeadScan(experiment=qqd, measurements=(Measurement('lead',index=2),) )
dqd2_lead_B = sm_tune_qqd.SMQQDLeadScan(experiment=qqd, measurements=(Measurement('lead',index=2),) )
dqd2_chrg = sm_tune_qqd.SMQQDChrgScan(experiment=qqd, measurements=(Measurement('chrg',index=2),) )

# Create evaluators for dots 3 and 4
dqd3_tunnel_coupling = sm_tune_qqd.SMQQDLineScan(experiment=qqd, measurements=(Measurement('line',index=3),) )
dqd3_lead_A = sm_tune_qqd.SMQQDLeadScan(experiment=qqd, measurements=(Measurement('lead',index=3),) )
dqd3_lead_B = sm_tune_qqd.SMQQDLeadScan(experiment=qqd, measurements=(Measurement('lead',index=3),) )
dqd3_chrg = sm_tune_qqd.SMQQDChrgScan(experiment=qqd, measurements=(Measurement('chrg',index=3),) )