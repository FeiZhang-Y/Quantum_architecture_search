"""
Quantum Gates
"""


class Gate:
    def __init__(self, qubits, name, para_gate):
        """
        :param qubits:  How many qubits the gate act on?
        :param name: Gate name
        :param para_gate: bool type, indicate whether this gate is a parameter gate
        :return:
        """
        self.qubits = qubits
        self.name = name
        self.para_gate = para_gate
        self.act_on = None  # which qubits this gate act on, like: [1],[1,2]
