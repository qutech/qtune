from qtune.sm import ChargeDiagram, BasicDQD

class Autotuner:
    def __init__(self, dqd: BasicDQD, solver=None):
        self.charge_diagram = ChargeDiagram(dqd, dqd._matlab)
        self.charge_diagram.measure_positions()
        self.charge_diagram.initialize_kalman()

        self.solver = solver
