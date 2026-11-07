#!/usr/bin/env python3

# Quantum Nash Equilibrium



"""
Quantum Nash Equilibrium V1 — Loto 7/39.

Generator:
    P(x | h) = sum_s pi[s] * P_s(x | h)

Za svaku kvantnu strategiju:
    P_s(x | h) = product_j T_s,j(x[j-1], x[j]; h) / Z_s(h)

T su Bornove verovatnoce dobijene iz Qiskit amplituda.
Z je zbir tezina svih legalnih sedmoclanih kombinacija.

Normalizacija je zajednicka za celu kombinaciju.
Prelazi se ne normalizuju zasebno po legalnim nastavcima.

Nash igra:
    min_pi max_w
        w @ (M.T @ pi - target) - 0.5 * ||w||^2

    pi >= 0
    sum(pi) = 1

Kvantna kola su deterministicka i fiksna.
Uce se tezine mesavine njihovih raspodela.
Kriticar je neprekidni linearni funkcional zajednickih karakteristika.

Nash gap odnosi se na ovu ogranicenu igru.
Ne predstavlja meru prediktivne preciznosti.

Sedam pozicija koristi sekvencijalna sestokubitna kola.
Ne simulira se zajednicko spregnuto stanje od 42 kubita.
Nema slucajnog uzorkovanja ili generisanja kombinacija.

Ulaz:
    tri aktuelna CSV fajla, bez zaglavlja;
    sedam razlicitih brojeva 1..39 po redu;
    redovi od najstarijeg ka najnovijem.

Broj redova iz naziva fajla nije uslov za ucenje.
Koriste se svi validni redovi iz samog CSV fajla.
"""



import os

for _thread_setting in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_setting] = "1"

import csv
import hashlib
import io
import itertools
import math
from pathlib import Path
import random
import re
import sys

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, Statevector
from threadpoolctl import threadpool_limits

random.seed(39)
np.random.seed(39)

DATA_DIR = Path("/data")

STRATEGIES = 16
WINDOWS = (1, 3)

BATCH = 48
MAP_BATCH = 8192

GAP_TOL = 1.0e-9
MAX_GAME_STEPS = 100000


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def make_circuit(strategy, position):
    register = QuantumRegister(6, "position_" + str(position + 1))
    circuit = QuantumCircuit(register)

    for layer in range(2):
        for qubit in range(6):
            phase = (
                (strategy + 1) * (qubit + 2) * (layer + 1)
                + 3 * (position + 1)
            )
            angle = math.pi * math.sin(phase * math.sqrt(2.0))
            circuit.ry(angle, qubit)

        for qubit in range(5):
            circuit.cx(qubit, qubit + 1)

        circuit.cx(5, 0)

    return circuit


def make_bank():
    bank = np.empty((STRATEGIES, 7, 64, 64), dtype=np.float64)

    for strategy in range(STRATEGIES):
        for position in range(7):
            operator = Operator(make_circuit(strategy, position)).data

            require(
                np.max(np.abs(operator.imag)) < 1.0e-13,
                "Kvantni operator nije realan.",
            )

            matrix = operator.real

            require(
                np.max(np.abs(matrix.T @ matrix - np.eye(64))) < 1.0e-12,
                "Provera ortogonalnosti nije prosla.",
            )

            bank[strategy, position] = matrix

    return bank


def discover_files():
    require(DATA_DIR.is_dir(), "CSV direktorijum ne postoji.")

    pattern = re.compile(
        r"^loto7_(\d+)_k(\d+)(?:_(loto|loto_plus)_(\d+))?\.csv$"
    )

    groups = {}

    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue

        match = pattern.fullmatch(path.name)

        if match is None:
            continue

        total = int(match[1])
        round_id = int(match[2])
        kind = match[3] or "all"
        named_count = int(match[4]) if match[4] else total

        groups.setdefault((total, round_id), {})[kind] = (
            path,
            named_count,
        )

    complete = [
        key
        for key, items in groups.items()
        if set(items) == {"all", "loto", "loto_plus"}
    ]

    require(
        bool(complete),
        "Nije pronadjen komplet od tri odgovarajuca CSV fajla.",
    )

    chosen = groups[max(complete)]

    return [
        (name, *chosen[name])
        for name in ("all", "loto", "loto_plus")
    ]


def read_draws(path, named_count):
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    rows = []

    for row_number, row in enumerate(csv.reader(io.StringIO(text)), 1):
        location = path.name + ", red " + str(row_number)

        require(
            len(row) == 7,
            "CSV red nema sedam kolona: " + location,
        )

        require(
            all(re.fullmatch(r"\s*\d+\s*", value) for value in row),
            "CSV sadrzi necelobrojnu vrednost: " + location,
        )

        values = sorted(int(value) for value in row)

        require(
            1 <= values[0] and values[-1] <= 39,
            "CSV broj je van opsega 1..39: " + location,
        )

        require(
            len(set(values)) == 7,
            "CSV red sadrzi duplikate: " + location,
        )

        rows.append(values)

    require(
        len(rows) >= 50,
        "Nedovoljno validnih redova: " + path.name,
    )

    print("Stvarni broj validnih redova:", len(rows))

    if len(rows) != named_count:
        print("Broj naveden u nazivu:", named_count)
        print("Za ucenje se koriste svi validni CSV redovi.")

    data = np.asarray(rows, dtype=np.int64)
    data.setflags(write=False)

    return data, hashlib.sha256(raw).hexdigest()


def input_bases(data, times, window, n=39):
    """Za cilj u vremenu t dozvoljeni su samo redovi pre t."""
    times = np.asarray(times, dtype=np.int64)

    require(window in WINDOWS, "Nepodrzan istorijski prozor.")

    require(
        np.all(times >= 1) and np.all(times <= len(data)),
        "Neispravno istorijsko vreme.",
    )

    width = data.shape[1]
    base = np.zeros((len(times), 64), dtype=np.float64)

    for lag in range(window):
        valid = times - 1 - lag >= 0

        base[valid, 7 * lag:7 * lag + width] = (
            data[times[valid] - 1 - lag] / float(n)
        )

    base[:, 22] = np.mean(data[times - 1], axis=1) / float(n)
    base[:, 63] = 1.0

    return base


def legal_mask(position, n=39, k=7):
    previous = np.arange(n + 1)
    following = np.arange(1, n + 1)

    if position == 0:
        valid_previous = previous == 0
    else:
        valid_previous = (
            (previous >= position)
            & (previous <= n - k + position)
        )

    return (
        valid_previous[:, None]
        & (following[None, :] > previous[:, None])
        & (following[None, :] <= n - k + position + 1)
    )


def quantum_potential(base, matrix, position, n=39, k=7):
    """
    Bornove verovatnoce pre zajednicke normalizacije.

    Normalizuje se ceo ulazni kvantni vektor.
    Ne normalizuje se zasebno skup legalnih nastavaka.
    """
    base = base.copy()
    base[:, 21] = (position + 1) / float(k)

    previous = np.arange(n + 1, dtype=np.float64)

    shift = (
        previous[:, None] / float(n)
        * matrix[:n, 23][None, :]
    )

    shift[1:] += matrix[:n, 24:24 + n].T

    amplitude = (
        (base @ matrix[:n].T)[:, None, :]
        + shift[None, :, :]
    )

    base_norm = np.sum(base * base, axis=1)

    previous_norm = (
        (previous / float(n)) ** 2
        + (previous > 0).astype(np.float64)
    )

    input_norm = (
        base_norm[:, None] + previous_norm[None, :]
    )

    require(
        np.all(input_norm > 0.0),
        "Ulazna kvantna amplituda ima nultu normu.",
    )

    potential = (
        amplitude * amplitude / input_norm[:, :, None]
    )

    potential *= legal_mask(position, n, k)[None, :, :]

    require(
        np.all(np.isfinite(potential))
        and np.all(potential >= 0.0),
        "Neispravan kvantni potencijal.",
    )

    require(
        float(np.max(potential)) <= 1.0 + 1.0e-12,
        "Bornova verovatnoca je veca od jedan.",
    )

    return potential


def forward_backward(tables):
    """
    tables: (pozicija, uzorak, prethodni_broj, naredni_broj)

    Vraca particiju Z, forward i backward poruke.
    """
    k, batch, previous_count, n = tables.shape

    require(
        previous_count == n + 1,
        "Neispravne dimenzije zajednicke raspodele.",
    )

    alpha = np.zeros(
        (k + 1, batch, n + 1),
        dtype=np.float64,
    )
    alpha[0, :, 0] = 1.0

    for position in range(k):
        alpha[position + 1, :, 1:] = np.sum(
            alpha[position, :, :, None] * tables[position],
            axis=1,
        )

    partition = alpha[k].sum(axis=1)

    require(
        np.all(np.isfinite(partition))
        and np.all(partition > 0.0),
        "Zajednicka raspodela nema pozitivnu konacnu particiju.",
    )

    beta = np.ones_like(alpha)

    for position in range(k - 1, -1, -1):
        beta[position] = np.sum(
            tables[position]
            * beta[position + 1, :, None, 1:],
            axis=2,
        )

    require(
        np.allclose(
            partition,
            beta[0, :, 0],
            rtol=1.0e-11,
            atol=0.0,
        ),
        "Forward i backward particije se ne poklapaju.",
    )

    return partition, alpha, beta


def joint_at_position(tables, partition, alpha, beta, position):
    joint = (
        alpha[position, :, :, None]
        * tables[position]
        * beta[position + 1, :, None, 1:]
        / partition[:, None, None]
    )

    require(
        np.max(np.abs(joint.sum(axis=(1, 2)) - 1.0)) < 1.0e-10,
        "Normalizacija zajednicke marginale nije prosla.",
    )

    return joint


def feature_count(n=39, k=7):
    return k * n + (k - 1) * n * n + n * n


def target_features(last, target, n=39, k=7):
    result = np.zeros(feature_count(n, k), dtype=np.float64)

    for position in range(k):
        np.add.at(
            result,
            position * n + target[:, position] - 1,
            1.0 / math.sqrt(k),
        )

    edge_start = k * n

    for position in range(k - 1):
        indices = (
            edge_start
            + position * n * n
            + (target[:, position] - 1) * n
            + target[:, position + 1] - 1
        )

        np.add.at(
            result,
            indices,
            1.0 / math.sqrt(k - 1),
        )

    last_hot = np.zeros((len(last), n), dtype=np.float64)
    target_hot = np.zeros_like(last_hot)

    last_hot[np.arange(len(last))[:, None], last - 1] = 1.0
    target_hot[np.arange(len(target))[:, None], target - 1] = 1.0

    result[edge_start + (k - 1) * n * n:] = (
        last_hot.T @ target_hot
    ).ravel() / float(k)

    return result, last_hot


def collect_stats(data, start, stop, window, bank, n=39, k=7):
    require(
        1 <= start < stop <= len(data),
        "Neispravan hronoloski interval.",
    )

    strategies = len(bank)

    moments = np.zeros(
        (strategies, feature_count(n, k)),
        dtype=np.float64,
    )

    target_sum = np.zeros(feature_count(n, k), dtype=np.float64)

    log_probability = np.empty(
        (stop - start, strategies),
        dtype=np.float64,
    )

    for first in range(start, stop, BATCH):
        times = np.arange(first, min(first + BATCH, stop))
        rows = np.arange(len(times))

        base = input_bases(data, times, window, n)
        target = data[times]

        observed_features, last_hot = target_features(
            data[times - 1], target, n, k
        )

        target_sum += observed_features

        for strategy in range(strategies):
            tables = np.stack(
                [
                    quantum_potential(
                        base,
                        bank[strategy, position],
                        position,
                        n,
                        k,
                    )
                    for position in range(k)
                ],
                axis=0,
            )

            partition, alpha, beta = forward_backward(tables)

            presence = np.zeros(
                (len(times), n),
                dtype=np.float64,
            )

            lp = -np.log(partition)

            for position in range(k):
                joint = joint_at_position(
                    tables,
                    partition,
                    alpha,
                    beta,
                    position,
                )

                current = joint.sum(axis=1)

                moments[
                    strategy,
                    position * n:(position + 1) * n,
                ] += current.sum(axis=0) / math.sqrt(k)

                if position > 0:
                    offset = k * n + (position - 1) * n * n

                    moments[strategy, offset:offset + n * n] += (
                        joint[:, 1:, :].sum(axis=0).ravel()
                        / math.sqrt(k - 1)
                    )

                previous = (
                    np.zeros(len(times), dtype=np.int64)
                    if position == 0
                    else target[:, position - 1]
                )

                observed = tables[
                    position,
                    rows,
                    previous,
                    target[:, position] - 1,
                ]

                with np.errstate(divide="ignore"):
                    lp += np.log(observed)

                presence += current

            offset = k * n + (k - 1) * n * n

            moments[strategy, offset:] += (
                last_hot.T @ presence
            ).ravel() / float(k)

            log_probability[times - start, strategy] = lp

    count = stop - start

    return {
        "M": moments / count,
        "target": target_sum / count,
        "logp": log_probability,
        "count": count,
    }


def combine_stats(first, second):
    count = first["count"] + second["count"]

    return {
        "M": (
            first["M"] * first["count"]
            + second["M"] * second["count"]
        ) / count,
        "target": (
            first["target"] * first["count"]
            + second["target"] * second["count"]
        ) / count,
        "count": count,
    }


def simplex_projection(vector):
    ordered = np.sort(vector)[::-1]
    offset = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(vector) + 1)

    eligible = ordered - offset / indices > 0.0

    require(np.any(eligible), "Projekcija tezina nije uspela.")

    rho = np.flatnonzero(eligible)[-1]
    threshold = offset[rho] / float(rho + 1)

    result = np.maximum(vector - threshold, 0.0)
    result /= result.sum()

    return result


def solve_game(stats):
    matrix = stats["M"]
    target = stats["target"]

    require(
        np.all(np.isfinite(matrix))
        and np.all(np.isfinite(target)),
        "Nekonacni podaci u Nash igri.",
    )

    difference = matrix - target[None, :]
    gram = difference @ difference.T

    pi = np.zeros(len(matrix), dtype=np.float64)
    pi[int(np.argmin(np.diag(gram)))] = 1.0

    lipschitz = max(
        float(np.linalg.eigvalsh(gram)[-1]),
        1.0e-15,
    )

    extrapolated = pi.copy()
    acceleration = 1.0

    for iteration in range(MAX_GAME_STEPS):
        gradient = gram @ pi
        gap = float(pi @ gradient - np.min(gradient))

        if gap <= GAP_TOL:
            break

        if iteration < MAX_GAME_STEPS // 2:
            candidate = simplex_projection(
                extrapolated
                - (gram @ extrapolated) / lipschitz
            )

            new_acceleration = 0.5 * (
                1.0
                + math.sqrt(
                    1.0 + 4.0 * acceleration * acceleration
                )
            )

            extrapolated = candidate + (
                (acceleration - 1.0) / new_acceleration
            ) * (candidate - pi)

            pi = candidate
            acceleration = new_acceleration

        else:
            vertex = int(np.argmin(gradient))

            direction = -pi.copy()
            direction[vertex] += 1.0

            direction_residual = difference.T @ direction
            curvature = float(
                direction_residual @ direction_residual
            )

            require(
                curvature > 0.0,
                "Nash optimizacija ima nerazresen ravan smer.",
            )

            step = min(1.0, gap / curvature)

            pi += step * direction
            pi = np.maximum(pi, 0.0)
            pi /= pi.sum()

    residual = difference.T @ pi
    payoff = difference @ residual

    gap = float(pi @ payoff - np.min(payoff))

    require(
        np.isfinite(gap)
        and gap >= -1.0e-12
        and gap <= GAP_TOL * 1.01,
        "Nash tolerancija nije dostignuta: " + str(gap),
    )

    require(
        np.all(pi >= 0.0)
        and abs(float(pi.sum()) - 1.0) < 1.0e-12,
        "Neispravne tezine generatora.",
    )

    return pi, max(0.0, gap), float(np.linalg.norm(residual))


def negative_log_likelihood(logp, pi):
    active = pi > 0.0

    values = logp[:, active] + np.log(pi[active])[None, :]
    largest = np.max(values, axis=1)

    if not np.all(np.isfinite(largest)):
        return float("inf")

    log_joint = largest + np.log(
        np.exp(values - largest[:, None]).sum(axis=1)
    )

    return -float(np.mean(log_joint))


def next_tables(data, time, window, bank, n=39, k=7):
    base = input_bases(
        data,
        np.asarray([time]),
        window,
        n,
    )

    return np.stack(
        [
            np.stack(
                [
                    quantum_potential(
                        base,
                        bank[strategy, position],
                        position,
                        n,
                        k,
                    )[0]
                    for position in range(k)
                ],
                axis=0,
            )
            for strategy in range(len(bank))
        ],
        axis=0,
    )


def partitions_for_model(tables):
    result = np.empty(len(tables), dtype=np.float64)

    for strategy in range(len(tables)):
        partition, _, _ = forward_backward(
            tables[strategy, :, None, :, :]
        )
        result[strategy] = partition[0]

    return result


def active_model(tables, pi):
    require(
        len(tables) == len(pi)
        and np.all(np.isfinite(pi))
        and np.all(pi >= 0.0)
        and abs(float(pi.sum()) - 1.0) < 1.0e-12,
        "Neispravna mesavina strategija.",
    )

    active = pi > 0.0

    require(np.any(active), "Generator nema aktivne strategije.")

    return tables[active], pi[active]


def batch_combination_probability(
    tables,
    combinations,
    pi,
    partitions,
):
    size = len(combinations)

    individual = np.ones(
        (size, len(tables)),
        dtype=np.float64,
    )

    for strategy in range(len(tables)):
        previous = np.zeros(size, dtype=np.int64)

        for position in range(combinations.shape[1]):
            following = combinations[:, position]

            individual[:, strategy] *= tables[
                strategy,
                position,
                previous,
                following - 1,
            ]

            previous = following

        individual[:, strategy] /= partitions[strategy]

    return individual @ pi, individual


def exact_map(tables, pi, n=39, k=7, announce=True):
    tables, pi = active_model(tables, pi)
    partitions = partitions_for_model(tables)

    iterator = itertools.combinations(range(1, n + 1), k)

    best_probability = -1.0
    best_combination = None

    masses = np.zeros(len(tables), dtype=np.float64)

    count = 0
    total = math.comb(n, k)
    next_report = 5000000

    while True:
        items = list(itertools.islice(iterator, MAP_BATCH))

        if not items:
            break

        combinations = np.asarray(items, dtype=np.int64)

        probability, individual = batch_combination_probability(
            tables,
            combinations,
            pi,
            partitions,
        )

        require(
            np.all(np.isfinite(probability))
            and np.all(probability >= 0.0),
            "Neispravna zajednicka verovatnoca.",
        )

        masses += individual.sum(axis=0)

        index = int(np.argmax(probability))

        if probability[index] > best_probability:
            best_probability = float(probability[index])
            best_combination = tuple(
                int(value) for value in combinations[index]
            )

        count += len(items)

        if announce and count >= next_report:
            print(
                "MAP pregled:",
                count,
                "/",
                total,
                flush=True,
            )
            next_report += 5000000

    require(
        count == total,
        "Pretraga globalnog maksimuma nije potpuna.",
    )

    require(
        np.max(np.abs(masses - 1.0)) < 1.0e-9,
        "Egzaktna provera ukupne mase nije prosla.",
    )

    require(
        best_combination is not None
        and best_probability > 0.0,
        "Globalni maksimum nije pronadjen.",
    )

    return (
        best_combination,
        best_probability,
        float(masses @ pi),
    )


def mixture_marginals(tables, pi):
    strategies, k, previous_count, n = tables.shape

    require(
        len(pi) == strategies and previous_count == n + 1,
        "Neispravne dimenzije mesavine.",
    )

    nodes = np.zeros((k, n), dtype=np.float64)
    edges = np.zeros((k - 1, n, n), dtype=np.float64)

    for strategy in np.flatnonzero(pi > 0.0):
        current_tables = tables[strategy, :, None, :, :]

        partition, alpha, beta = forward_backward(current_tables)

        for position in range(k):
            joint = joint_at_position(
                current_tables,
                partition,
                alpha,
                beta,
                position,
            )[0]

            nodes[position] += (
                pi[strategy] * joint.sum(axis=0)
            )

            if position > 0:
                edges[position - 1] += (
                    pi[strategy] * joint[1:]
                )

    require(
        np.max(np.abs(nodes.sum(axis=1) - 1.0)) < 1.0e-10,
        "Normalizacija marginala mesavine nije prosla.",
    )

    return np.concatenate((nodes.ravel(), edges.ravel()))


def compare_distributions(left_tables, left_pi, right_tables, right_pi):
    left_marginals = mixture_marginals(left_tables, left_pi)
    right_marginals = mixture_marginals(right_tables, right_pi)

    marginal_difference = float(
        np.max(np.abs(left_marginals - right_marginals))
    )

    if marginal_difference > 1.0e-12:
        return marginal_difference, None

    if (
        np.array_equal(left_pi, right_pi)
        and np.array_equal(left_tables, right_tables)
    ):
        return marginal_difference, 0.0

    lt, lp = active_model(left_tables, left_pi)
    rt, rp = active_model(right_tables, right_pi)

    require(
        lt.shape[1:] == rt.shape[1:],
        "Raspodele nemaju isti prostor kombinacija.",
    )

    lz = partitions_for_model(lt)
    rz = partitions_for_model(rt)

    k = lt.shape[1]
    n = lt.shape[-1]

    iterator = itertools.combinations(range(1, n + 1), k)

    absolute_sum = 0.0
    count = 0

    while True:
        items = list(itertools.islice(iterator, MAP_BATCH))

        if not items:
            break

        combinations = np.asarray(items, dtype=np.int64)

        left_probability, _ = batch_combination_probability(
            lt, combinations, lp, lz
        )

        right_probability, _ = batch_combination_probability(
            rt, combinations, rp, rz
        )

        absolute_sum += float(
            np.abs(left_probability - right_probability).sum()
        )

        count += len(items)

    require(
        count == math.comb(n, k),
        "Poredjenje zajednickih raspodela nije potpuno.",
    )

    return marginal_difference, 0.5 * absolute_sum


def model_fingerprint(window, pi, tables):
    digest = hashlib.sha256()

    digest.update(np.asarray([window], dtype="<i8").tobytes())
    digest.update(np.asarray(pi, dtype="<f8").tobytes())
    digest.update(np.asarray(tables, dtype="<f8").tobytes())

    return digest.hexdigest()


def verify_qiskit_amplitudes(bank):
    data = np.asarray(
        [
            [1, 4, 9, 13, 21, 30, 39],
            [2, 5, 10, 17, 23, 28, 36],
            [3, 6, 12, 19, 25, 32, 38],
        ],
        dtype=np.int64,
    )

    base = input_bases(data, np.asarray([3]), 3)

    for strategy in range(len(bank)):
        for position in range(7):
            previous_values = (
                (0,)
                if position == 0
                else (position, 32 + position)
            )

            for previous in previous_values:
                vector = base[0].copy()

                vector[21] = (position + 1) / 7.0
                vector[23] = previous / 39.0

                if previous > 0:
                    vector[24 + previous - 1] = 1.0

                vector /= np.linalg.norm(vector)

                circuit = QuantumCircuit(6)
                circuit.initialize(vector, range(6))

                circuit.compose(
                    make_circuit(strategy, position),
                    inplace=True,
                )

                actual = Statevector.from_instruction(circuit).data
                expected = bank[strategy, position] @ vector

                require(
                    np.max(np.abs(actual - expected)) < 1.0e-12,
                    "Qiskit amplitude se ne poklapaju.",
                )

                probability = np.abs(actual[:39]) ** 2
                probability *= legal_mask(position)[previous]

                contracted = quantum_potential(
                    base,
                    bank[strategy, position],
                    position,
                )[0, previous]

                require(
                    np.max(
                        np.abs(probability - contracted)
                    ) < 1.0e-12,
                    "Bornovi potencijali se ne poklapaju.",
                )


def verify_reference_partition():
    """
    Kontrolni matematicki slucaj, bez istorijskih podataka.
    Jednaki ulazni potencijali proveravaju combinatoriku normalizacije.
    Ovo nije pretpostavka raspodele stvarnog modela.
    """
    for n, k in ((9, 3), (3, 3)):
        tables = np.stack(
            [
                legal_mask(position, n, k).astype(np.float64)
                / 64.0
                for position in range(k)
            ],
            axis=0,
        )

        partition, _, _ = forward_backward(
            tables[:, None, :, :]
        )

        expected_partition = math.comb(n, k) / (64.0 ** k)

        require(
            math.isclose(
                float(partition[0]),
                expected_partition,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ),
            "Kontrolna kombinatorna particija nije prosla.",
        )

        combinations = np.asarray(
            list(itertools.combinations(range(1, n + 1), k)),
            dtype=np.int64,
        )

        probability, _ = batch_combination_probability(
            tables[None, :, :, :],
            combinations,
            np.asarray([1.0]),
            partition,
        )

        require(
            np.max(
                np.abs(probability - 1.0 / math.comb(n, k))
            ) < 1.0e-12,
            "Kontrolna zajednicka normalizacija nije prosla.",
        )


def verify_small_joint(bank):
    n = 9
    k = 3

    data = np.asarray(
        [
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9],
            [1, 5, 9],
        ],
        dtype=np.int64,
    )

    sub_bank = bank[:3, :k]

    stats = collect_stats(
        data, 1, len(data), 3, sub_bank, n, k
    )

    combinations = np.asarray(
        list(itertools.combinations(range(1, n + 1), k)),
        dtype=np.int64,
    )

    explicit_moments = np.zeros_like(stats["M"])

    for time in range(1, len(data)):
        tables = next_tables(
            data, time, 3, sub_bank, n, k
        )

        partitions = partitions_for_model(tables)
        pi = np.asarray([0.2, 0.3, 0.5])

        mixed, individual = batch_combination_probability(
            tables, combinations, pi, partitions
        )

        require(
            np.max(
                np.abs(individual.sum(axis=0) - 1.0)
            ) < 1.0e-12,
            "Normalizacija malog prostora nije prosla.",
        )

        for row, combination in enumerate(combinations):
            feature, _ = target_features(
                data[time - 1:time],
                combination[None, :],
                n,
                k,
            )

            explicit_moments += (
                individual[row, :, None] * feature[None, :]
            )

        best, probability, _ = exact_map(
            tables, pi, n, k, announce=False
        )

        index = int(np.argmax(mixed))

        require(
            best == tuple(combinations[index])
            and abs(probability - mixed[index]) < 1.0e-14,
            "Globalni maksimum malog prostora se ne poklapa.",
        )

        marginal = mixture_marginals(tables, pi)

        expected_nodes = np.zeros((k, n), dtype=np.float64)
        expected_edges = np.zeros((k - 1, n, n), dtype=np.float64)

        for row, combination in enumerate(combinations):
            weight = mixed[row]

            for position in range(k):
                expected_nodes[
                    position, combination[position] - 1
                ] += weight

            for position in range(k - 1):
                expected_edges[
                    position,
                    combination[position] - 1,
                    combination[position + 1] - 1,
                ] += weight

        explicit_marginal = np.concatenate(
            (expected_nodes.ravel(), expected_edges.ravel())
        )

        require(
            np.max(
                np.abs(marginal - explicit_marginal)
            ) < 1.0e-12,
            "Marginale mesavine se ne poklapaju.",
        )

        observed_index = np.flatnonzero(
            np.all(combinations == data[time], axis=1)
        )

        require(
            len(observed_index) == 1,
            "Kontrolna kombinacija nije jednoznacna.",
        )

        expected_logp = np.log(individual[observed_index[0]])

        require(
            np.max(
                np.abs(
                    stats["logp"][time - 1] - expected_logp
                )
            ) < 1.0e-11,
            "Verovatnoca istorijskog cilja se ne poklapa.",
        )

    explicit_moments /= len(data) - 1

    require(
        np.max(
            np.abs(stats["M"] - explicit_moments)
        ) < 1.0e-12,
        "Momenti se ne poklapaju sa punom raspodelom.",
    )


def verify_game_solver():
    cases = (
        ([0.25, 0.75], [0.25, 0.75]),
        ([-0.5, 1.5], [0.0, 1.0]),
        ([1.5, -0.5], [1.0, 0.0]),
    )

    for target, expected in cases:
        stats = {
            "M": np.eye(2),
            "target": np.asarray(target, dtype=np.float64),
        }

        weights, gap, _ = solve_game(stats)

        require(
            np.max(
                np.abs(weights - np.asarray(expected))
            ) < 1.0e-7,
            "Nash optimizacija ne daje analiticko resenje.",
        )

        require(
            gap <= GAP_TOL * 1.01,
            "Analiticka provera Nash tolerancije nije prosla.",
        )


def verify_causality():
    data = np.asarray(
        [
            [1, 4, 9, 13, 21, 30, 39],
            [2, 5, 10, 17, 23, 28, 36],
            [3, 6, 12, 19, 25, 32, 38],
        ],
        dtype=np.int64,
    )

    changed = data.copy()
    changed[1:] = [1, 2, 3, 4, 5, 6, 7]

    first = input_bases(data, np.asarray([1]), 3)
    second = input_bases(changed, np.asarray([1]), 3)

    require(
        np.array_equal(first, second),
        "Buduci redovi uticu na raniji ulaz.",
    )


def run_dataset(name, data, digest, bank):
    count = len(data) - 1

    train_stop = 1 + int(count * 0.60)
    validation_stop = 1 + int(count * 0.80)

    candidates = []

    print()
    print("CSV:", name, flush=True)
    print("SHA256 ulaza:", digest)
    print("Validnih redova:", len(data))

    for window in WINDOWS:
        train = collect_stats(
            data, 1, train_stop, window, bank
        )

        validation = collect_stats(
            data, train_stop, validation_stop, window, bank
        )

        pi, gap, _ = solve_game(train)

        score = negative_log_likelihood(
            validation["logp"], pi
        )

        candidates.append(
            (score, window, train, validation)
        )

        print(
            "Prozor:",
            window,
            "validacija NLL:",
            format(score, ".10f"),
            "Nash gap:",
            format(gap, ".3e"),
            flush=True,
        )

    selected_score, window, train, validation = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )

    require(
        np.isfinite(selected_score),
        "Nijedan kandidat nema konacan validacioni NLL.",
    )

    pretest = combine_stats(train, validation)
    pi_test, _, _ = solve_game(pretest)

    test = collect_stats(
        data, validation_stop, len(data), window, bank
    )

    test_nll = negative_log_likelihood(
        test["logp"], pi_test
    )

    print("Odabrani prozor:", window)
    print(
        "Odvojeni test NLL:",
        format(test_nll, ".10f"),
        flush=True,
    )

    full = combine_stats(pretest, test)
    pi, gap, residual = solve_game(full)

    require(
        full["count"] == len(data) - 1,
        "Zavrsno ucenje ne obuhvata sve prelaze.",
    )

    tables = next_tables(
        data, len(data), window, bank
    )

    earlier = next_tables(
        data, len(data) - 1, window, bank
    )

    history_difference, history_tv = compare_distributions(
        tables, pi, earlier, pi
    )

    print("Zavrsno ucenje prelaza:", full["count"])
    print(
        "Nash gap:",
        format(gap, ".3e"),
    )
    print(
        "Odstupanje kriticarskih karakteristika:",
        format(residual, ".10f"),
    )
    print(
        "Razlika marginala pri promeni istorije:",
        format(history_difference, ".3e"),
    )

    if history_tv is not None:
        print(
            "TV razlika zajednickih raspodela:",
            format(history_tv, ".12e"),
        )

        if history_tv <= 1.0e-12:
            print(
                "AUDIT: poslednja promena istorijskog ulaza"
                " nije promenila raspodelu."
            )

    print(
        "Model SHA256:",
        model_fingerprint(window, pi, tables),
    )
    print(
        "Naucene tezine:",
        " ".join(format(value, ".17g") for value in pi),
        flush=True,
    )

    best, probability, mass = exact_map(tables, pi)

    require(
        len(best) == 7
        and len(set(best)) == 7
        and 1 <= best[0] < best[-1] <= 39,
        "Neispravna NEXT kombinacija.",
    )

    longest = 1
    consecutive = 1

    for left, right in zip(best, best[1:]):
        consecutive = (
            consecutive + 1 if right == left + 1 else 1
        )
        longest = max(longest, consecutive)

    historical_repeats = int(
        np.count_nonzero(
            np.all(data == np.asarray(best), axis=1)
        )
    )

    print(
        "Ukupna masa zajednicke raspodele:",
        format(mass, ".12f"),
    )

    if (
        longest >= 5
        or best[-1] - best[0] <= 10
        or historical_repeats > 0
    ):
        print("AUDIT obrasca globalnog maksimuma")
        print("Najduzi uzastopni niz:", longest)
        print("Raspon:", best[-1] - best[0])
        print("Istorijskih pojavljivanja:", historical_repeats)
        print(
            "Maksimum i masa provereni su"
            " preko svih legalnih kombinacija."
        )
        print(
            "Obrazac nije filtriran."
            " Prediktivni kvalitet nije potvrdjen."
        )

    return {
        "name": name,
        "best": best,
        "probability": probability,
        "pi": pi,
        "tables": tables,
        "digest": digest,
    }


def main():
    require(
        sys.version_info[:2] == (3, 11),
        "Pokreni fajl u Python 3.11 okruzenju.",
    )

    print("Quantum Nash Equilibrium V1")
    print(
        "Python:", sys.version.split()[0],
        "NumPy:", np.__version__,
        "Qiskit:", qiskit.__version__,
    )
    print("Seed:", 39)
    print("BLAS threads:", 1)
    print("Zajednicka normalizacija cele kombinacije.")

    files = discover_files()
    loaded = []

    for name, path, named_count in files:
        print("CSV putanja:", str(path), flush=True)

        data, digest = read_draws(path, named_count)
        loaded.append((name, data, digest))

    print("Provera konstrukcije...", flush=True)

    bank = make_bank()

    verify_qiskit_amplitudes(bank)
    verify_reference_partition()
    verify_small_joint(bank)
    verify_game_solver()
    verify_causality()

    print(
        "Ugradjene provere zavrsene u ovom izvrsavanju.",
        flush=True,
    )

    results = []

    for name, data, digest in loaded:
        results.append(
            run_dataset(name, data, digest, bank)
        )

    for left in range(len(results)):
        for right in range(left + 1, len(results)):
            first = results[left]
            second = results[right]

            if first["best"] != second["best"]:
                continue

            difference, total_variation = compare_distributions(
                first["tables"],
                first["pi"],
                second["tables"],
                second["pi"],
            )

            print()
            print(
                "AUDIT istog NEXT:",
                first["name"],
                second["name"],
            )
            print(
                "Razliciti CSV otisci:",
                first["digest"] != second["digest"],
            )
            print(
                "Razlika marginala naucenih raspodela:",
                format(difference, ".3e"),
            )

            if total_variation is not None:
                print(
                    "TV razlika zajednickih raspodela:",
                    format(total_variation, ".12e"),
                )

                if total_variation <= 1.0e-12:
                    print(
                        "AUDIT: ogranicena igra dala je"
                        " iste izlazne raspodele."
                    )

            print(
                "Isti maksimum je zadrzan."
                " Razlicitost kombinacija nije nametnuta."
            )

    print()
    print("Konacni NEXT rezultati")

    for result in results:
        print(
            result["name"] + ":",
            " ".join(str(value) for value in result["best"]),
        )
        print(
            "Verovatnoca prema modelu:",
            format(result["probability"], ".12e"),
        )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()



"""
Quantum Nash Equilibrium V1
Python: 3.11.13 NumPy: 1.26.4 Qiskit: 1.4.4
Seed: 39
BLAS threads: 1
Zajednicka normalizacija cele kombinacije.
CSV putanja: /data/loto7_4716_k89.csv
Stvarni broj validnih redova: 4716
CSV putanja: /data/loto7_4716_k89_loto_2980.csv
Stvarni broj validnih redova: 2979
Broj naveden u nazivu: 2980
Za ucenje se koriste svi validni CSV redovi.
CSV putanja: /data/loto7_4716_k89_loto_plus_1736.csv
Stvarni broj validnih redova: 1735
Broj naveden u nazivu: 1736
Za ucenje se koriste svi validni CSV redovi.
Provera konstrukcije...
Ugradjene provere zavrsene u ovom izvrsavanju.


CSV: all
SHA256 ulaza: deb02ea741f500e3270402730ed89565f1c9b4618d53e5a08fea2b7c2c2af31a
Validnih redova: 4716
Prozor: 1 validacija NLL: 18.9492370844 Nash gap: 6.429e-10
Prozor: 3 validacija NLL: 19.3070547526 Nash gap: 9.886e-10
Odabrani prozor: 1
Odvojeni test NLL: 19.1394182229
Zavrsno ucenje prelaza: 4715
Nash gap: 6.858e-10
Odstupanje kriticarskih karakteristika: 0.0906986380
Razlika marginala pri promeni istorije: 1.206e-02
Model SHA256: a64e0293cfa45d2aa48c43f5dcce0fce2983c90bbf3cb8ceaccac298ce2c3a36
Naucene tezine: 0.04945701883101692 0.15463028301074597 0.02258772369358818 0.062078662486780956 0.066580726415152142 0.10578292158461895 0.027786344568541871 0.045146455430295281 0.075118125928730101 0.042681858164818114 0.038402165425551915 0.053660602525334049 0.10575587448560721 0.070757156482769437 0 0.079574080966448996
MAP pregled: 5005312 / 15380937
MAP pregled: 10002432 / 15380937
MAP pregled: 15007744 / 15380937
Ukupna masa zajednicke raspodele: 1.000000000000


CSV: loto
SHA256 ulaza: f1cd81cd04660d50d9058675924da34346fc006ce8dabd84d52266c66afe17d1
Validnih redova: 2979
Prozor: 1 validacija NLL: 18.9787812637 Nash gap: 5.314e-10
Prozor: 3 validacija NLL: 19.0137424118 Nash gap: 4.864e-10
Odabrani prozor: 1
Odvojeni test NLL: 19.0761077126
Zavrsno ucenje prelaza: 2978
Nash gap: 8.993e-10
Odstupanje kriticarskih karakteristika: 0.0918295059
Razlika marginala pri promeni istorije: 1.187e-02
Model SHA256: d7e5a76099668bcf82e914ea0262272d302d5b80c7dd6a19796155d60a6a5b90
Naucene tezine: 0.049947008299239412 0.15212486050406349 0.016718601617104329 0.055866029523000599 0.065016313059127057 0.10689714921058621 0.028792186562692553 0.045357015352025687 0.077499173205160282 0.042361476725086195 0.040646271422074012 0.052614768714115821 0.10597834236347201 0.071548614695389673 0 0.088632188746862683
MAP pregled: 5005312 / 15380937
MAP pregled: 10002432 / 15380937
MAP pregled: 15007744 / 15380937
Ukupna masa zajednicke raspodele: 1.000000000000


CSV: loto_plus
SHA256 ulaza: 56225ed0f0be7e2d3e30779804d8fa45e96ecfb9d14441eb2a731dd7fff38ae6
Validnih redova: 1735
Prozor: 1 validacija NLL: 19.1051560261 Nash gap: 5.986e-10
Prozor: 3 validacija NLL: 19.4138340526 Nash gap: 9.412e-10
Odabrani prozor: 1
Odvojeni test NLL: 19.0676024924
Zavrsno ucenje prelaza: 1734
Nash gap: 5.165e-10
Odstupanje kriticarskih karakteristika: 0.0969265104
Razlika marginala pri promeni istorije: 2.518e-02
Model SHA256: 174942c486ec5748bc469d54dbb42e957a2bf63e905e5ebdcb7b52c6fe4613cf
Naucene tezine: 0.049005810855126564 0.15776239933621433 0.032571687110316466 0.072915091344969971 0.069419706829076624 0.10346001631115495 0.026441732973957237 0.044729305703402347 0.070814717372075314 0.043065324495655893 0.034566325921672088 0.055793645404755877 0.10547082511844179 0.069644609079227759 0 0.064338802143952767
MAP pregled: 5005312 / 15380937
MAP pregled: 10002432 / 15380937
MAP pregled: 15007744 / 15380937
Ukupna masa zajednicke raspodele: 1.000000000000


AUDIT istog NEXT: loto loto_plus
Razliciti CSV otisci: True
Razlika marginala naucenih raspodela: 1.401e-02
Isti maksimum je zadrzan. Razlicitost kombinacija nije nametnuta.


Konacni NEXT rezultati
all: 6 x 25 y 32 z 39
Verovatnoca prema modelu: 3.341769518786e-04
loto: 11 x 25 y 32 z 39
Verovatnoca prema modelu: 4.563104122831e-04
loto_plus: 11 x 25 y 32 z 39
Verovatnoca prema modelu: 5.276418848471e-04
"""



"""
Nash Equilibrium

Nashova ravnoteža kroz igru dva modela:
- Kvantni generator uči raspodelu kombinacija.
- Kritičar daje kontinuirani skor razlike između naučene raspodele i podataka.
Ideja ravnoteže suprotstavljenih ciljeva. 
Sličan pristup postoji u kvantnom Wasserstein adversarijalnom učenju.

Našova ravnoteža: stanje u kojem nijedan učesnik ne može poboljšati svoj rezultat jednostranom promenom strategije.

Za Loto 7/39 algoritam sa dva učesnika:
- Kvantni generator uči raspodelu naredne kombinacije uslovljenu prethodnim kolima.
- Kritičar daje brojčanu ocenu odstupanja naučene raspodele od podataka.
Njihovo nadmetanje definisati, meriti približavanje ravnoteži i iz raspodele odrediti jednu NEXT kombinaciju. 

Igra sa konačnim skupom kvantnih strategija. 
Generator uči njihove težine, a kritičar meri odstupanja raspodele i veza sa prethodnim kolom. 
Takva konstrukcija omogućava računanje odstupanja od ravnoteže, bez nasumičnog generisanja kombinacija.
"""
