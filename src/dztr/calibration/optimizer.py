"""A reproducible NumPy NSGA-II prototype for two calibration objectives.

This implements nondominated sorting, crowding, and binary tournaments with
bounded SBX and polynomial mutation. It is not an AMALGAM implementation.
Algorithm outline: https://pymoo.org/algorithms/moo/nsga2.html
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Callable

import numpy as np

from .candidates import _validated_vector


_MAX_DUPLICATE_RETRIES = 256


@dataclass(frozen=True)
class NSGA2Settings:
    """Explicit evaluation budget and variation settings for a seeded pilot."""

    seed: int = 42
    population_size: int = 16
    evaluation_budget: int = 96
    crossover_probability: float = 0.9
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    local_initialization_fraction: float = 0.5
    local_initialization_scale: float = 0.05

    def __post_init__(self) -> None:
        for name in ("seed", "population_size", "evaluation_budget"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer.")
            object.__setattr__(self, name, int(value))
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")
        if self.population_size < 2:
            raise ValueError("population_size must be at least two.")
        if self.evaluation_budget < self.population_size:
            raise ValueError("evaluation_budget must be at least population_size.")
        for name in ("crossover_probability", "crossover_eta", "mutation_eta", "local_initialization_fraction", "local_initialization_scale"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite number.") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("crossover_probability must be in [0, 1].")
        if not 0 <= self.local_initialization_fraction <= 1:
            raise ValueError("local_initialization_fraction must be in [0, 1].")
        if self.local_initialization_scale < 0:
            raise ValueError("local_initialization_scale must be nonnegative.")
        if self.crossover_eta <= 0 or self.mutation_eta <= 0:
            raise ValueError("crossover_eta and mutation_eta must be positive.")


@dataclass(frozen=True)
class EvaluatedCandidate:
    """A detached trial record; daily simulation trajectories are not retained."""

    evaluation_id: int
    generation: int
    vector: np.ndarray
    objectives: tuple[float, float]
    feasible: bool
    reason: str | None


@dataclass(frozen=True)
class GenerationRecord:
    """Cumulative progress after initialization or a complete/partial generation."""

    generation: int
    evaluations: int
    feasible_count: int
    front_count: int


@dataclass(frozen=True)
class OptimizerState:
    """Detached search state at a complete initialization/generation boundary.

    Population order and the entire PCG64 state, including cached integer bits,
    are needed for exact continuation. Partial generations and duplicate-
    exhausted populations are deliberately not resumable states. Stored trial
    costs are reused; state validation does not run reservoir simulations.
    """

    settings: NSGA2Settings
    lower_vector: np.ndarray
    upper_vector: np.ndarray
    baseline_vector: np.ndarray
    evaluated: tuple[EvaluatedCandidate, ...]
    history: tuple[GenerationRecord, ...]
    population_ids: tuple[int, ...]
    rng_state: dict
    generation: int
    boundary_type: str = "generation_complete"
    complete_generation: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.settings, NSGA2Settings):
            raise ValueError("Optimizer state settings must be NSGA2Settings.")
        object.__setattr__(self, "settings", NSGA2Settings(**asdict(self.settings)))
        generation = _state_integer(self.generation, "generation", minimum=0)
        if not isinstance(self.complete_generation, (bool, np.bool_)) or not self.complete_generation:
            raise ValueError("Optimizer state must represent a complete generation boundary.")
        expected_boundary = "initial_population" if generation == 0 else "generation_complete"
        if self.boundary_type != expected_boundary:
            raise ValueError(f"Optimizer state boundary_type must be {expected_boundary!r} for generation {generation}.")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "complete_generation", True)
        lower, upper, baseline = (_validated_vector(getattr(self, name))
                                  for name in ("lower_vector", "upper_vector", "baseline_vector"))
        if (lower > upper).any() or not np.isfinite(upper - lower).all():
            raise ValueError("Optimizer state bounds must have finite nonnegative widths.")
        if not ((baseline >= lower) & (baseline <= upper)).all():
            raise ValueError("Optimizer state baseline must lie inside its bounds.")
        for name, vector in (("lower_vector", lower), ("upper_vector", upper), ("baseline_vector", baseline)):
            vector.setflags(write=False)
            object.__setattr__(self, name, vector)
        try:
            evaluated, history, population_ids = tuple(self.evaluated), tuple(self.history), tuple(self.population_ids)
        except TypeError as exc:
            raise ValueError("Optimizer state evaluated/history/population_ids must be sequences.") from exc
        count = (generation + 1) * self.settings.population_size
        if len(evaluated) != count or count > self.settings.evaluation_budget:
            raise ValueError("Optimizer state evaluation count must equal complete generations and stay within its budget.")
        detached, seen = [], set()
        for index, point in enumerate(evaluated):
            if not isinstance(point, EvaluatedCandidate):
                raise ValueError("Optimizer state trial records must be EvaluatedCandidate values.")
            identifier = _state_integer(point.evaluation_id, "evaluation_id", minimum=0)
            point_generation = _state_integer(point.generation, "candidate generation", minimum=0)
            if identifier != index or point_generation != index // self.settings.population_size:
                raise ValueError("Optimizer state trial IDs/generations must be contiguous and population aligned.")
            vector = _validated_vector(point.vector)
            if not ((vector >= lower) & (vector <= upper)).all():
                raise ValueError("Optimizer state trial vector lies outside its bounds.")
            vector_key = _vector_key(vector)
            if vector_key in seen:
                raise ValueError("Optimizer state cannot contain duplicate trial vectors.")
            seen.add(vector_key)
            costs = _state_costs(point.objectives)
            if not isinstance(point.feasible, (bool, np.bool_)):
                raise ValueError("Optimizer state candidate feasibility must be boolean.")
            feasible = bool(point.feasible)
            if feasible and (not np.isfinite(costs).all() or point.reason is not None):
                raise ValueError("Feasible optimizer state trials require finite costs and no rejection reason.")
            if not feasible and (not np.isposinf(costs).all() or not isinstance(point.reason, str) or not point.reason):
                raise ValueError("Rejected optimizer state trials require two positive-infinite costs and a reason.")
            vector.setflags(write=False)
            detached.append(EvaluatedCandidate(index, point_generation, vector,
                (float(costs[0]), float(costs[1])), feasible, point.reason))
        if not np.array_equal(detached[0].vector, baseline):
            raise ValueError("Optimizer state trial zero must be the saved baseline.")
        if len(history) != generation + 1:
            raise ValueError("Optimizer state history must contain every complete generation exactly once.")
        costs = np.asarray([point.objectives for point in detached])
        feasible_counts = np.cumsum([point.feasible for point in detached])
        detached_history = []
        for index, record in enumerate(history):
            if not isinstance(record, GenerationRecord):
                raise ValueError("Optimizer state history records must be GenerationRecord values.")
            values = [_state_integer(getattr(record, name), name, minimum=0)
                      for name in ("generation", "evaluations", "feasible_count", "front_count")]
            record_generation, record_count, feasible_count, front_count = values
            expected_count = (index + 1) * self.settings.population_size
            if record_generation != index or record_count != expected_count:
                raise ValueError("Optimizer state history generations/counts must be complete and contiguous.")
            if feasible_count != feasible_counts[expected_count - 1] or front_count != len(_pareto_indices(costs[:expected_count])):
                raise ValueError("Optimizer state history feasible/Pareto counts do not match its trial records.")
            detached_history.append(GenerationRecord(*values))
        population_ids = tuple(_state_integer(value, "population ID", minimum=0) for value in population_ids)
        if len(population_ids) != self.settings.population_size or len(set(population_ids)) != len(population_ids) or any(value >= count for value in population_ids):
            raise ValueError("Optimizer state population IDs must identify one distinct full retained population.")
        object.__setattr__(self, "evaluated", tuple(detached))
        object.__setattr__(self, "history", tuple(detached_history))
        object.__setattr__(self, "population_ids", population_ids)
        object.__setattr__(self, "rng_state", _validated_rng_state(self.rng_state))

    @property
    def evaluations(self) -> int:
        return len(self.evaluated)


@dataclass(frozen=True)
class OptimizationResult:
    """All trials, final survivors, and the feasible Pareto archive of every trial."""

    settings: NSGA2Settings
    history: tuple[GenerationRecord, ...]
    evaluated: tuple[EvaluatedCandidate, ...]
    final_population: tuple[EvaluatedCandidate, ...]
    front: tuple[EvaluatedCandidate, ...]
    termination: str

    @property
    def evaluations(self) -> int:
        return len(self.evaluated)


def _state_integer(value, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"Optimizer state {name} must be an integer at least {minimum}.")
    return int(value)


def _vector_key(vector: np.ndarray) -> bytes:
    # Positive and negative zero are the same numerical search point.
    return np.where(vector == 0, 0.0, vector).tobytes()


def _state_costs(costs) -> np.ndarray:
    raw = np.asarray(costs)
    if np.iscomplexobj(raw):
        raise ValueError("Optimizer state costs must be two real numbers.")
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Optimizer state costs must be two real numbers.") from exc
    if values.shape != (2,):
        raise ValueError("Optimizer state costs must contain exactly two values.")
    return values


def _validated_rng_state(state) -> dict:
    """Validate the exact PCG64 integer state, including its cached uint32."""
    if not isinstance(state, dict) or set(state) != {"bit_generator", "state", "has_uint32", "uinteger"} or state.get("bit_generator") != "PCG64":
        raise ValueError("Optimizer state requires a complete PCG64 random-generator state.")
    inner = state["state"]
    if not isinstance(inner, dict) or set(inner) != {"state", "inc"}:
        raise ValueError("Optimizer state PCG64 state/inc fields are missing or invalid.")
    canonical = deepcopy(state)
    for name in ("state", "inc"):
        value = _state_integer(inner[name], f"PCG64 {name}", minimum=0)
        if value >= 2 ** 128 or (name == "inc" and not value % 2):
            raise ValueError("Optimizer state PCG64 state/inc integers are out of range or use an even increment.")
        canonical["state"][name] = value
    for name, limit in (("has_uint32", 1), ("uinteger", 2 ** 32 - 1)):
        value = _state_integer(state[name], f"PCG64 {name}", minimum=0)
        if value > limit:
            raise ValueError(f"Optimizer state PCG64 {name} is out of range.")
        canonical[name] = value
    generator = np.random.PCG64()
    try:
        generator.state = deepcopy(canonical)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Optimizer state random-generator state cannot be restored.") from exc
    return canonical


def _pareto_indices(costs: np.ndarray) -> np.ndarray:
    """Two-cost archive indices, preserving distinct vectors with tied costs."""
    finite = np.flatnonzero(np.isfinite(costs).all(axis=1))
    if not len(finite):
        return finite
    values = costs[finite]
    order = np.lexsort((values[:, 1], values[:, 0]))
    x, y = values[order].T
    running_y = np.minimum.accumulate(y)
    previous_y = np.r_[np.inf, running_y[:-1]]
    improvements = y < previous_y
    best_index = np.maximum.accumulate(np.where(improvements, np.arange(len(y)), -1))
    selected = (y == running_y) & (x == x[best_index])
    return np.sort(finite[order[selected]])


def validate_optimizer_state(state: OptimizerState) -> OptimizerState:
    """Detach and validate trial history, RNG and deterministic survivor lineage.

    This is used before saving/loading/resuming an external snapshot. It only
    recomputes NSGA-II bookkeeping from saved costs, never model evaluations.
    """
    if not isinstance(state, OptimizerState):
        raise ValueError("resume_state must be an OptimizerState.")
    validated = OptimizerState(**{name: getattr(state, name) for name in (
        "settings", "lower_vector", "upper_vector", "baseline_vector", "evaluated", "history",
        "population_ids", "rng_state", "generation", "boundary_type", "complete_generation",
    )})
    count = validated.settings.population_size
    population = list(validated.evaluated[:count])
    for generation in range(1, validated.generation + 1):
        children = list(validated.evaluated[generation * count:(generation + 1) * count])
        population = _survivors(population + children, count)
    if tuple(point.evaluation_id for point in population) != validated.population_ids:
        raise ValueError("Optimizer state population IDs/order do not match deterministic NSGA-II survivors.")
    return validated


def _cost_matrix(costs) -> np.ndarray:
    raw = np.asarray(costs)
    if np.iscomplexobj(raw):
        raise ValueError("Costs must be real numbers.")
    values = np.asarray(raw, dtype=float)
    if values.size == 0 and values.ndim == 1:
        values = values.reshape(0, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Costs must have shape (candidate_count, 2).")
    if np.isnan(values).any() or np.isneginf(values).any():
        raise ValueError("Costs cannot contain NaN or negative infinity.")
    # Infeasibility is a separate state, never a partial objective tradeoff.
    values = values.copy()
    values[~np.isfinite(values).all(axis=1)] = np.inf
    return values


def nondominated_sort(costs) -> list[np.ndarray]:
    """Return all minimization fronts as indices, retaining objective ties.

    Positive-infinite rows are ranked after finite rows. They may form the only
    rank when every point is infeasible; the returned optimization Pareto
    archive separately excludes every such point.
    """
    values = _cost_matrix(costs)
    dominated = []
    domination_count = np.zeros(len(values), dtype=int)
    for row in values:
        indices = np.flatnonzero(np.all(row <= values, axis=1) & np.any(row < values, axis=1))
        dominated.append(indices)
        domination_count[indices] += 1
    fronts = []
    current = np.flatnonzero(domination_count == 0)
    while current.size:
        fronts.append(current)
        following = []
        for index in current:
            for other in dominated[index]:
                domination_count[other] -= 1
                if domination_count[other] == 0:
                    following.append(other)
        current = np.asarray(sorted(following), dtype=int)
    return fronts


def crowding_distance(costs) -> np.ndarray:
    """Normalized Manhattan crowding for a single front, with infinite extremes."""
    values = _cost_matrix(costs)
    distance = np.zeros(len(values))
    if not len(values) or not np.isfinite(values).all():
        return distance
    if len(values) <= 2:
        distance[:] = np.inf
        return distance
    for objective in range(2):
        order = np.argsort(values[:, objective], kind="stable")
        span = values[order[-1], objective] - values[order[0], objective]
        if span <= 0:
            continue
        distance[order[0]] = distance[order[-1]] = np.inf
        distance[order[1:-1]] += (values[order[2:], objective] - values[order[:-2], objective]) / span
    return distance


def _ranks_and_crowding(population: list[EvaluatedCandidate]) -> tuple[np.ndarray, np.ndarray]:
    costs = np.asarray([point.objectives for point in population])
    ranks = np.empty(len(population), dtype=int)
    distances = np.empty(len(population))
    for rank, front in enumerate(nondominated_sort(costs)):
        ranks[front] = rank
        distances[front] = crowding_distance(costs[front])
    return ranks, distances


def _survivors(population: list[EvaluatedCandidate], count: int) -> list[EvaluatedCandidate]:
    costs = np.asarray([point.objectives for point in population])
    selected = []
    for front in nondominated_sort(costs):
        remaining = count - len(selected)
        if len(front) <= remaining:
            selected.extend(population[index] for index in front)
        else:
            distances = crowding_distance(costs[front])
            # Evaluation ID provides a stable tie-break without comparing arrays.
            order = sorted(range(len(front)), key=lambda i: (-distances[i], population[front[i]].evaluation_id))
            selected.extend(population[front[i]] for i in order[:remaining])
        if len(selected) == count:
            break
    return selected


def _tournament(ranks: np.ndarray, distances: np.ndarray, rng: np.random.Generator) -> int:
    if len(ranks) == 1:
        return 0
    first, second = rng.choice(len(ranks), size=2, replace=False)
    if ranks[first] != ranks[second]:
        return int(first if ranks[first] < ranks[second] else second)
    if distances[first] != distances[second]:
        return int(first if distances[first] > distances[second] else second)
    return int(first if rng.random() < 0.5 else second)


def _sbx(first: np.ndarray, second: np.ndarray, active: np.ndarray,
         settings: NSGA2Settings, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Deb's bounded simulated binary crossover on normalized [0, 1] genes."""
    child_one, child_two = first.copy(), second.copy()
    if rng.random() > settings.crossover_probability:
        return child_one, child_two
    exponent = 1.0 / (settings.crossover_eta + 1.0)
    for index in np.flatnonzero(active):
        if rng.random() > 0.5 or abs(first[index] - second[index]) <= 1e-14:
            continue
        lower, upper = sorted((first[index], second[index]))
        draw = rng.random()
        beta = 1.0 + 2.0 * lower / (upper - lower)
        alpha = 2.0 - beta ** (-(settings.crossover_eta + 1.0))
        spread = (draw * alpha) ** exponent if draw <= 1.0 / alpha else (1.0 / (2.0 - draw * alpha)) ** exponent
        left = 0.5 * (lower + upper - spread * (upper - lower))
        beta = 1.0 + 2.0 * (1.0 - upper) / (upper - lower)
        alpha = 2.0 - beta ** (-(settings.crossover_eta + 1.0))
        spread = (draw * alpha) ** exponent if draw <= 1.0 / alpha else (1.0 / (2.0 - draw * alpha)) ** exponent
        right = 0.5 * (lower + upper + spread * (upper - lower))
        left, right = np.clip([left, right], 0.0, 1.0)
        if rng.random() < 0.5:
            left, right = right, left
        child_one[index], child_two[index] = left, right
    return child_one, child_two


def _mutate(candidate: np.ndarray, active: np.ndarray, eta: float, rng: np.random.Generator) -> np.ndarray:
    """Bounded polynomial mutation with probability 1 / free_dimension per gene."""
    child = candidate.copy()
    free_count = int(active.sum())
    if not free_count:
        return child
    exponent = 1.0 / (eta + 1.0)
    for index in np.flatnonzero(active):
        if rng.random() > 1.0 / free_count:
            continue
        value, draw = child[index], rng.random()
        if draw <= 0.5:
            term = 2.0 * draw + (1.0 - 2.0 * draw) * (1.0 - value) ** (eta + 1.0)
            shift = term ** exponent - 1.0
        else:
            term = 2.0 * (1.0 - draw) + 2.0 * (draw - 0.5) * value ** (eta + 1.0)
            shift = 1.0 - term ** exponent
        child[index] = np.clip(value + shift, 0.0, 1.0)
    return child


def _latin_hypercube(count: int, active: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    points = np.zeros((count, len(active)))
    if count:
        for index in np.flatnonzero(active):
            points[:, index] = (rng.permutation(count) + rng.random(count)) / count
    return points


def optimize_nsga2(
    objective,
    baseline_vector,
    settings: NSGA2Settings | None = None,
    *,
    bounds=None,
    progress_callback: Callable[[GenerationRecord], None] | None = None,
    resume_state: OptimizerState | None = None,
    checkpoint_callback: Callable[[OptimizerState], None] | None = None,
    stopping_callback: Callable[[OptimizerState], bool] | None = None,
) -> OptimizationResult:
    """Search calibration objectives only with an exact, counted trial budget.

    ``objective.evaluate(vector)`` must return ``feasible``, two minimization
    ``objectives``, and an optional ``reason``. Bounds default to the objective's
    bounds and expose native ``lower_vector``/``upper_vector``. Both NumPy and
    real reservoir objectives are supported; validation is never called here.

    Baseline is trial zero. Other initial points mix local bounded Gaussian
    perturbations with Latin hypercube exploration. All subsequent variation is
    in normalized coordinates, keeping fixed-width dimensions unchanged.
    Every unique trial, including an infeasible one, consumes one evaluation.
    Budget exhaustion supports a partial final generation; duplicate exhaustion
    stops after bounded retries. Unexpected evaluation exceptions propagate.

    Complete boundaries optionally emit a detached checkpoint before progress
    and stopping callbacks. Partial final generations have no resumable state.
    Resuming restores survivor order, seen trials and RNG without reevaluation;
    only the evaluation budget may grow. A resumed stopping hook is rechecked
    before new trials, protecting a crash after saving but before stopping.
    """
    settings = (resume_state.settings if isinstance(resume_state, OptimizerState)
                else NSGA2Settings()) if settings is None else settings
    if not isinstance(settings, NSGA2Settings):
        raise TypeError("settings must be NSGA2Settings.")
    if not callable(getattr(objective, "evaluate", None)):
        raise TypeError("objective must provide evaluate(vector).")
    for name, callback in (("progress_callback", progress_callback),
                           ("checkpoint_callback", checkpoint_callback), ("stopping_callback", stopping_callback)):
        if callback is not None and not callable(callback):
            raise TypeError(f"{name} must be callable or None.")
    bounds = getattr(objective, "bounds", None) if bounds is None else bounds
    if bounds is None:
        raise ValueError("NSGA-II requires explicit calibration candidate bounds.")
    lower, upper = _validated_vector(bounds.lower_vector), _validated_vector(bounds.upper_vector)
    if (lower > upper).any():
        raise ValueError("Each candidate lower bound must not exceed its upper bound.")
    width = upper - lower
    if not np.isfinite(width).all():
        raise ValueError("Candidate bound widths must be finite.")
    baseline = _validated_vector(baseline_vector)
    if not ((baseline >= lower) & (baseline <= upper)).all():
        raise ValueError("Baseline vector must lie inside the candidate bounds.")
    active = width > 0
    rng = np.random.default_rng(settings.seed)
    evaluated: list[EvaluatedCandidate] = []
    archive: list[EvaluatedCandidate] = []
    history: list[GenerationRecord] = []
    seen: set[bytes] = set()
    feasible_count = 0
    population: list[EvaluatedCandidate] = []
    generation = 0
    termination = "budget_reached"
    if resume_state is not None:
        saved = validate_optimizer_state(resume_state)
        original, requested = asdict(saved.settings), asdict(settings)
        changed = [name for name in original if name != "evaluation_budget" and original[name] != requested[name]]
        if changed:
            raise ValueError(f"Resume settings cannot change {', '.join(changed)}.")
        if settings.evaluation_budget < saved.settings.evaluation_budget:
            raise ValueError("Resume evaluation_budget may extend but cannot shrink the saved budget.")
        if not np.array_equal(lower, saved.lower_vector) or not np.array_equal(upper, saved.upper_vector):
            raise ValueError("Resume candidate bounds must exactly match the saved bounds.")
        if not np.array_equal(baseline, saved.baseline_vector):
            raise ValueError("Resume baseline must exactly match the saved baseline.")
        rng.bit_generator.state = deepcopy(saved.rng_state)
        evaluated = list(saved.evaluated)
        costs = np.asarray([point.objectives for point in evaluated])
        archive = [evaluated[index] for index in _pareto_indices(costs)]
        history = list(saved.history)
        population = [evaluated[index] for index in saved.population_ids]
        seen = {_vector_key(point.vector) for point in evaluated}
        feasible_count = sum(point.feasible for point in evaluated)
        generation = saved.generation

    def normalized(vector):
        return np.divide(vector - lower, width, out=np.zeros(72), where=active)

    def decoded(unit):
        values = lower + np.clip(unit, 0.0, 1.0) * width
        values[~active] = lower[~active]
        return np.clip(values, lower, upper)

    def key(vector):
        return _vector_key(vector)

    def register(vector, generation):
        nonlocal archive, feasible_count
        seen.add(key(vector))
        evaluation = objective.evaluate(vector.copy())
        if not isinstance(evaluation.feasible, (bool, np.bool_)):
            raise ValueError("Objective feasibility must be a boolean.")
        raw_costs = np.asarray(evaluation.objectives)
        if np.iscomplexobj(raw_costs):
            raise ValueError("Objective evaluation must return real minimization costs.")
        costs = np.asarray(raw_costs, dtype=float)
        if costs.shape != (2,):
            raise ValueError("Objective evaluation must return exactly two minimization costs.")
        feasible = bool(evaluation.feasible) and bool(np.isfinite(costs).all())
        reason = getattr(evaluation, "reason", None)
        if not feasible:
            reason = str(reason or "Objective evaluation is infeasible or returned nonfinite costs.")
            costs = np.full(2, np.inf)
        else:
            reason = None
        detached = vector.copy()
        detached.setflags(write=False)
        point = EvaluatedCandidate(len(evaluated), generation, detached,
                                   (float(costs[0]), float(costs[1])), feasible, reason)
        evaluated.append(point)
        if feasible:
            feasible_count += 1
            old_costs = np.asarray([old.objectives for old in archive], dtype=float).reshape(-1, 2)
            dominated_by_old = np.all(old_costs <= costs, axis=1) & np.any(old_costs < costs, axis=1)
            if not dominated_by_old.any():
                dominates_old = np.all(costs <= old_costs, axis=1) & np.any(costs < old_costs, axis=1)
                archive = [old for old, remove in zip(archive, dominates_old) if not remove]
                archive.append(point)
        # Neither evaluation nor its daily SimulationResult enters the archive.
        return point

    def snapshot(generation):
        return OptimizerState(settings, lower, upper, baseline, tuple(evaluated), tuple(history),
            tuple(point.evaluation_id for point in population), deepcopy(rng.bit_generator.state),
            generation, "initial_population" if generation == 0 else "generation_complete", True)

    def should_stop(state):
        if stopping_callback is None:
            return False
        decision = stopping_callback(state)
        if not isinstance(decision, (bool, np.bool_)):
            raise ValueError("stopping_callback must return a boolean.")
        return bool(decision)

    def progress(generation, *, complete_generation=False):
        record = GenerationRecord(generation, len(evaluated), feasible_count, len(archive))
        history.append(record)
        state = snapshot(generation) if complete_generation and (checkpoint_callback is not None or stopping_callback is not None) else None
        if state is not None and checkpoint_callback is not None:
            checkpoint_callback(state)
        if progress_callback is not None:
            progress_callback(record)
        return should_stop(state) if state is not None else False

    def fresh(propose):
        if not active.any():
            return None
        for _ in range(_MAX_DUPLICATE_RETRIES):
            vector = decoded(propose())
            if key(vector) not in seen:
                return vector
        return None

    if resume_state is None:
        unit_baseline = normalized(baseline)
        population = [register(baseline, 0)]
        remaining = settings.population_size - 1
        local_count = int(np.floor(settings.local_initialization_fraction * remaining))
        local = []
        for _ in range(local_count):
            point = unit_baseline.copy()
            point[active] += rng.normal(0.0, settings.local_initialization_scale, int(active.sum()))
            local.append(np.clip(point, 0.0, 1.0))
        global_points = _latin_hypercube(remaining - local_count, active, rng)
        for initial in [*local, *global_points]:
            proposed = decoded(initial)
            if key(proposed) in seen:
                proposed = fresh(lambda: np.where(active, rng.random(72), 0.0))
            if proposed is None:
                termination = "duplicate_exhausted"
                break
            population.append(register(proposed, 0))
        if progress(0, complete_generation=len(population) == settings.population_size):
            termination = "stopping_candidate"
    elif stopping_callback is not None and should_stop(snapshot(generation)):
        termination = "stopping_candidate"
    while len(evaluated) < settings.evaluation_budget and termination not in ("duplicate_exhausted", "stopping_candidate"):
        generation += 1
        ranks, distances = _ranks_and_crowding(population)
        parent_units = [normalized(point.vector) for point in population]
        pending: list[np.ndarray] = []

        def offspring():
            if not pending:
                first = parent_units[_tournament(ranks, distances, rng)]
                second = parent_units[_tournament(ranks, distances, rng)]
                children = _sbx(first, second, active, settings, rng)
                pending.extend(_mutate(child, active, settings.mutation_eta, rng) for child in children)
            return pending.pop(0)

        children = []
        requested = min(settings.population_size, settings.evaluation_budget - len(evaluated))
        for _ in range(requested):
            proposed = fresh(offspring)
            if proposed is None:
                termination = "duplicate_exhausted"
                break
            children.append(register(proposed, generation))
        if children:
            population = _survivors(population + children, settings.population_size)
            complete = len(children) == settings.population_size and termination != "duplicate_exhausted"
            if progress(generation, complete_generation=complete):
                termination = "stopping_candidate"
        else:
            break
    if not archive:
        reasons = Counter(point.reason for point in evaluated)
        detail = "; ".join(f"{count} x {reason}" for reason, count in sorted(reasons.items()))
        raise ValueError(f"No feasible candidate was found in {len(evaluated)} evaluations ({detail}).")
    return OptimizationResult(settings, tuple(history), tuple(evaluated), tuple(population),
                              tuple(archive), termination)
