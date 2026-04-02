from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass
class StrategyExpectation:
    name: str
    expected_return_annual: float = 0.10
    expected_vol_annual: float = 0.15
    expected_sharpe_override: float | None = None
    base_weight: float = 1.0
    min_weight: float = 0.02
    max_weight: float = 0.70
    drawdown_limit: float = 0.10
    hard_stop_drawdown: float = 0.20

    @property
    def expected_sharpe(self) -> float:
        if self.expected_sharpe_override is not None:
            return float(self.expected_sharpe_override)
        return self.expected_return_annual / max(self.expected_vol_annual, 1e-8)


@dataclass
class StrategyState:
    expectation: StrategyExpectation
    observations: int = 0
    ewma_mean_return: float = 0.0
    ewma_var_return: float = 1e-8
    hit_rate: float = 0.5
    cumulative_return: float = 1.0
    peak_equity: float = 1.0
    drawdown: float = 0.0
    capacity_utilization: float = 0.0
    capacity_nav_limit: float = float("inf")
    avg_impact_bps: float = 0.0
    latest_weight: float = 0.0
    performance_gap: float = 0.0
    realized_return_annual: float = 0.0
    realized_sharpe: float = 0.0
    last_realized_return: float = 0.0


class CentralRiskAllocator:

    def __init__(
        self,
        expectations: list[StrategyExpectation],
        annualization: float = 252.0,
        ewma_decay: float = 0.94,
        min_observations: int = 20,
        exploration_floor: float = 0.15,
        capacity_soft_limit: float = 0.70,
        score_temperature: float = 1.0,
    ):
        if not expectations:
            raise ValueError("At least one strategy expectation is required")
        self.annualization = annualization
        self.ewma_decay = ewma_decay
        self.min_observations = min_observations
        self.exploration_floor = exploration_floor
        self.capacity_soft_limit = capacity_soft_limit
        self.score_temperature = score_temperature
        self.states: dict[str, StrategyState] = {
            exp.name: StrategyState(expectation=exp)
            for exp in expectations
        }

    def observe(self, strategy_name: str, realized_return: float) -> None:
        state = self.states[strategy_name]
        lam = self.ewma_decay

        prev_mean = state.ewma_mean_return
        state.ewma_mean_return = lam * state.ewma_mean_return + (1.0 - lam) * realized_return
        deviation = realized_return - prev_mean
        state.ewma_var_return = (
            lam * state.ewma_var_return + (1.0 - lam) * deviation * deviation
        )
        state.hit_rate = lam * state.hit_rate + (1.0 - lam) * (1.0 if realized_return > 0 else 0.0)
        state.cumulative_return *= (1.0 + realized_return)
        state.peak_equity = max(state.peak_equity, state.cumulative_return)
        state.drawdown = (
            1.0 - state.cumulative_return / max(state.peak_equity, 1e-12)
            if state.peak_equity > 0 else 0.0
        )
        state.observations += 1
        state.last_realized_return = realized_return

        state.realized_return_annual = state.ewma_mean_return * self.annualization
        std = math.sqrt(max(state.ewma_var_return, 1e-12))
        state.realized_sharpe = (
            state.ewma_mean_return / std * math.sqrt(self.annualization)
            if std > 1e-12 else 0.0
        )
        state.performance_gap = state.realized_sharpe - state.expectation.expected_sharpe

    def observe_capacity(
        self,
        strategy_name: str,
        utilization: float,
        capacity_nav_limit: float | None = None,
        impact_bps: float | None = None,
    ) -> None:
        state = self.states[strategy_name]
        lam = self.ewma_decay
        if state.capacity_utilization <= 1e-12:
            state.capacity_utilization = max(utilization, 0.0)
        else:
            state.capacity_utilization = (
                lam * state.capacity_utilization + (1.0 - lam) * max(utilization, 0.0)
            )
        if capacity_nav_limit is not None:
            if np.isfinite(state.capacity_nav_limit):
                state.capacity_nav_limit = min(state.capacity_nav_limit, capacity_nav_limit)
            else:
                state.capacity_nav_limit = capacity_nav_limit
        if impact_bps is not None:
            if state.avg_impact_bps <= 1e-12:
                state.avg_impact_bps = max(impact_bps, 0.0)
            else:
                state.avg_impact_bps = lam * state.avg_impact_bps + (1.0 - lam) * max(impact_bps, 0.0)

    def _base_weights(self) -> dict[str, float]:
        total = sum(max(state.expectation.base_weight, 0.0) for state in self.states.values())
        if total <= 0:
            equal = 1.0 / len(self.states)
            return {name: equal for name in self.states}
        return {
            name: max(state.expectation.base_weight, 0.0) / total
            for name, state in self.states.items()
        }

    def target_weights(self) -> dict[str, float]:
        base = self._base_weights()
        raw_scores: dict[str, float] = {}

        for name, state in self.states.items():
            exp = state.expectation
            confidence = min(state.observations / max(self.min_observations, 1), 1.0)
            sharpe_gap = state.realized_sharpe - exp.expected_sharpe
            return_gap = (
                state.realized_return_annual - exp.expected_return_annual
            ) / max(exp.expected_vol_annual, 1e-8)
            hit_bonus = 2.0 * (state.hit_rate - 0.5)
            drawdown_penalty = state.drawdown / max(exp.drawdown_limit, 1e-8)
            capacity_penalty = max(
                0.0,
                (state.capacity_utilization - self.capacity_soft_limit)
                / max(1.0 - self.capacity_soft_limit, 1e-8),
            )

            score = (
                0.60 * sharpe_gap
                + 0.25 * return_gap
                + 0.15 * hit_bonus
                - 0.35 * drawdown_penalty
                - 0.45 * capacity_penalty
            )
            if state.drawdown >= exp.hard_stop_drawdown:
                score = -12.0

            raw_scores[name] = math.log(max(base[name], 1e-12)) + confidence * score

        max_raw = max(raw_scores.values())
        dynamic = {
            name: math.exp((score - max_raw) * self.score_temperature)
            for name, score in raw_scores.items()
        }
        dynamic_total = sum(dynamic.values())
        if dynamic_total <= 0:
            dynamic = base.copy()
        else:
            dynamic = {name: value / dynamic_total for name, value in dynamic.items()}

        blended = {
            name: self.exploration_floor * base[name] + (1.0 - self.exploration_floor) * dynamic[name]
            for name in self.states
        }
        clipped = {
            name: min(
                max(weight, self.states[name].expectation.min_weight),
                self.states[name].expectation.max_weight,
            )
            for name, weight in blended.items()
        }
        total = sum(clipped.values())
        if total <= 0:
            equal = 1.0 / len(clipped)
            clipped = {name: equal for name in clipped}
        else:
            clipped = {name: weight / total for name, weight in clipped.items()}

        for name, weight in clipped.items():
            self.states[name].latest_weight = weight
        return clipped

    def combine_signals(self, strategy_signals: Mapping[str, float]) -> float:
        active = {
            name: float(signal)
            for name, signal in strategy_signals.items()
            if name in self.states and abs(signal) > 1e-12
        }
        if not active:
            return 0.0

        weights = self.target_weights()
        total_weight = sum(weights.get(name, 0.0) for name in active)
        if total_weight <= 1e-12:
            total_weight = len(active)
            weights = {name: 1.0 for name in active}

        score = sum(weights.get(name, 0.0) * signal for name, signal in active.items()) / total_weight
        return float(np.clip(score, -1.0, 1.0))

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "weight": state.latest_weight,
                "realized_return_annual": state.realized_return_annual,
                "realized_sharpe": state.realized_sharpe,
                "expected_sharpe": state.expectation.expected_sharpe,
                "performance_gap": state.performance_gap,
                "drawdown": state.drawdown,
                "capacity_utilization": state.capacity_utilization,
                "avg_impact_bps": state.avg_impact_bps,
                "observations": float(state.observations),
            }
            for name, state in self.states.items()
        }
