#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from typing import Tuple

import pandas as pd
from opt_einsum import contract_path
import tensornetwork as tn
import tensorcircuit as tc


def get_circuit_dataframe(circuit: tc.Circuit):
    n_gates = circuit.gate_count()
    n_cnot = circuit.gate_count(["cnot"])
    n_mc = circuit.gate_count(["multicontrol"])
    # avoid int overflow and for better formatting
    n_qubits = circuit.circuit_param["nqubits"]
    flop = count_circuit_flop(circuit)

    df_dict = {
        "#qubits": n_qubits,
        "#gates": [n_gates],
        "#CNOT": [n_cnot],
        "#multicontrol": [n_mc],
        "depth": [circuit.to_qiskit().depth()],
        "#FLOP": [flop],
    }
    return pd.DataFrame(df_dict)


def count_circuit_flop(circuit: tc.Circuit):
    # tensor network contraction flops by greedy algorithm
    nodes = circuit._copy()[0]
    input_set_list = [set([id(e) for e in node.edges]) for node in nodes]
    array_list = [node.tensor for node in nodes]
    output_set = set([id(e) for e in tn.get_subgraph_dangling(nodes)])
    args = []
    for i in range(len(nodes)):
        args.extend([array_list[i], input_set_list[i]])
    args.append(output_set)
    _, desc = contract_path(*args)
    return desc.opt_cost


def evolve_pauli(circuit: tc.Circuit, pauli_string: Tuple, theta: float):
    # pauli_string in openfermion.QubitOperator.terms format
    for idx, symbol in pauli_string:
        if symbol == "X":
            circuit.H(idx)
        elif symbol == "Y":
            circuit.SD(idx)
            circuit.H(idx)
        elif symbol == "Z":
            continue
        else:
            raise ValueError(f"Invalid Pauli String: {pauli_string}")

    for i in range(len(pauli_string) - 1):
        circuit.CNOT(pauli_string[i][0], pauli_string[i + 1][0])
    circuit.rz(pauli_string[-1][0], theta=theta)

    for i in reversed(range(len(pauli_string) - 1)):
        circuit.CNOT(pauli_string[i][0], pauli_string[i + 1][0])

    for idx, symbol in pauli_string:
        if symbol == "X":
            circuit.H(idx)
        elif symbol == "Y":
            circuit.H(idx)
            circuit.S(idx)
        elif symbol == "Z":
            continue
        else:
            raise ValueError(f"Invalid Pauli String: {pauli_string}")
    return circuit


def multicontrol_ry(theta, ctrl=(0, 1, 0)):
    # ancilla-free multicontrolled Ry, generalized to an arbitrary number of controls
    # https://arxiv.org/pdf/2005.14475.pdf, see also
    # https://github.com/tequilahub/tequila/blob/master/src/tequila/quantumchemistry/chemistry_tools.py
    n_ctrl = len(ctrl)
    c = tc.Circuit(n_ctrl + 1)
    controls = list(range(n_ctrl))
    target = n_ctrl

    # controls with ctrl == 0 are flipped so that the recursive construction below,
    # which always controls on |1>, implements the requested 0/1 pattern
    flipped = [q for q, b in zip(controls, ctrl) if b == 0]
    for q in flipped:
        c.x(q)

    def cry_recursive(dcontrol, angle, case):
        if not dcontrol:
            c.ry(target, theta=angle)
            return
        aux, rest = dcontrol[0], dcontrol[1:]
        if case:
            cry_recursive(rest, angle / 2, True)
            c.h(aux)
            c.cnot(target, aux)
            cry_recursive(rest, -angle / 2, False)
            c.cnot(target, aux)
            c.h(aux)
        else:
            c.h(aux)
            c.cnot(target, aux)
            cry_recursive(rest, -angle / 2, False)
            c.cnot(target, aux)
            c.h(aux)
            cry_recursive(rest, angle / 2, True)

    cry_recursive(controls, theta, True)

    for q in flipped:
        c.x(q)
    return c
