"""Boardroom agents that negotiate the next EcoCloud action."""

from __future__ import annotations

from .models import CloudAction, CloudObservation


class ResourceAgent:
    """Focuses on minimising latency."""

    def propose(
        self, obs: CloudObservation, last_obs: CloudObservation | None = None
    ) -> tuple[str, str]:
        """Recommend an action based on latency level and trend."""
        if obs.crisis_just_happened:
            return ("crisis_response", "Response time degrading — deploying horizontal scale-up immediately.")
        if obs.latency > 200 and (last_obs is None or obs.latency >= last_obs.latency):
            return ("scale_up", "Latency high and not improving")
        if obs.latency > 150:
            return ("scale_up", "Latency above 150ms target")
        return ("optimize_energy", "Latency OK")


class CostAgent:
    """Focuses on minimising infrastructure cost."""

    def propose(
        self, obs: CloudObservation, last_obs: CloudObservation | None = None
    ) -> tuple[str, str]:
        """Recommend an action based on cost level and trend."""
        if obs.crisis_just_happened:
            return ("crisis_response", "Cost at $620/hr, target is $400. Switching to reserved capacity first.")
        if obs.cost > 450 and obs.latency > 180:
            return ("optimize_energy", "Cost high, but latency is too fragile for scale-down")
        if obs.cost > 450 and (last_obs is None or obs.cost >= last_obs.cost):
            return ("scale_down", "Cost high and not improving")
        if obs.cost > 400 and obs.latency > 160:
            return ("optimize_energy", "Cost over budget - trim efficiently without hurting latency")
        if obs.cost > 400:
            return ("scale_down", "Cost over $400 budget")
        return ("optimize_energy", "Cost OK")


class SustainabilityAgent:
    """Focuses on minimising carbon emissions."""

    def propose(
        self, obs: CloudObservation, last_obs: CloudObservation | None = None
    ) -> tuple[str, str]:
        """Recommend an action based on carbon level."""
        del last_obs
        if obs.crisis_just_happened:
            return ("crisis_response", "Emissions exceeding target — activating energy optimization protocol.")
        if obs.carbon > 350:
            return ("migrate_region", "Carbon critical - migrating to green data centre")
        if obs.carbon > 280:
            return ("migrate_region", "High carbon - shifting to low-carbon region")
        if obs.carbon > 220:
            return ("optimize_energy", "Carbon above target - optimising local energy")
        if obs.latency > 180:
            return ("optimize_energy", "Carbon OK - preserving latency while trimming cost")
        return ("scale_down", "Carbon OK - reducing cost")


class Boardroom:
    """Runs the multi-agent negotiation and picks the final action."""

    def __init__(self) -> None:
        """Initialise the three boardroom advisors."""
        self.resource_agent = ResourceAgent()
        self.cost_agent = CostAgent()
        self.sustainability_agent = SustainabilityAgent()

    def decide(
        self,
        obs: CloudObservation,
        last_obs: CloudObservation | None = None,
        recent_actions: list[str] | None = None,
        verbose: bool = True,
    ) -> tuple[CloudAction, list[str]]:
        """Collect proposals, block oscillation, and return a coordinated action."""
        recent_actions = recent_actions or []
        proposals = {
            "ResourceAgent": self.resource_agent.propose(obs, last_obs),
            "CostAgent": self.cost_agent.propose(obs, last_obs),
            "SustainabilityAgent": self.sustainability_agent.propose(obs, last_obs),
        }
        if obs.crisis_just_happened:
            return self._crisis_response(proposals, verbose=verbose)
        if len(recent_actions) >= 3:
            last3 = recent_actions[-3:]
            if len(set(last3)) == 3 and "migrate_region" not in last3:
                override_line = "[BOARDROOM] 3-action cycle detected -> migrate_region"
                if verbose:
                    print(override_line)
                return CloudAction(action="migrate_region"), [override_line]
        if len(recent_actions) >= 2 and set(recent_actions[-2:]) == {"scale_up", "scale_down"}:
            override_line = "[BOARDROOM] Oscillation detected -> migrate_region"
            if verbose:
                print(override_line)
            return CloudAction(action="migrate_region"), [override_line]
        votes: dict[str, int] = {}
        log: list[str] = []
        for agent_name, (action, reason) in proposals.items():
            votes[action] = votes.get(action, 0) + 1
            line = f"[BOARDROOM] {agent_name}: {action} - {reason}"
            log.append(line)
            if verbose:
                print(line)
        winning_action = self._select_action(obs, votes)
        decision_line = f"[BOARDROOM] Decision: {winning_action} ({self._decision_reason(obs, votes, winning_action)})"
        log.append(decision_line)
        if verbose:
            print(decision_line)
            print("-" * 50)
        return CloudAction(action=winning_action), log

    def _crisis_response(
        self, proposals: dict[str, tuple[str, str]], verbose: bool = True
    ) -> tuple[CloudAction, list[str]]:
        """Merge crisis advice into one fast, cheap, and green action."""
        log: list[str] = []
        for agent_name, (_, reason) in proposals.items():
            line = f"[BOARDROOM] {agent_name}: {reason}"
            log.append(line)
            if verbose:
                print(line)
        decision = CloudAction(action="crisis_response", server_count=5, region="canada-hydro")
        decision_line = "[BOARDROOM] Decision: crisis_response - add 5 servers in canada-hydro"
        log.append(decision_line)
        if verbose:
            print(decision_line)
            print("-" * 50)
        return decision, log

    def _safety_override(self, obs: CloudObservation, winning_action: str) -> str:
        """Prevent obviously harmful actions when latency is already critical."""
        if obs.latency > 260 and winning_action in {"scale_down", "optimize_energy"}:
            return "scale_up"
        if obs.latency > 220 and winning_action == "scale_down":
            return "scale_up"
        return winning_action

    def _select_action(self, obs: CloudObservation, votes: dict[str, int]) -> str:
        """Choose an action using phase-based recovery with vote-aware guardrails."""
        safe_majority = self._safe_majority(obs, votes)
        if safe_majority is not None:
            return safe_majority
        return self._goal_directed_action(obs)

    def _safe_majority(self, obs: CloudObservation, votes: dict[str, int]) -> str | None:
        """Accept a 2-vote consensus when it does not worsen the current bottleneck."""
        majority_actions = [action for action, count in votes.items() if count >= 2]
        for action in majority_actions:
            if self._is_safe_choice(obs, action):
                return self._safety_override(obs, action)
        return None

    def _is_safe_choice(self, obs: CloudObservation, action: str) -> bool:
        """Reject majority actions that obviously push the state away from recovery."""
        if obs.latency > 170 and action in {"scale_down", "optimize_energy"}:
            return False
        if obs.latency > 160 and action == "migrate_region":
            return False
        if obs.cost > 520 and action == "scale_up":
            return False
        if obs.carbon < 220 and action == "migrate_region":
            return False
        return True

    def _goal_directed_action(self, obs: CloudObservation) -> str:
        """Recover by phase: stabilise latency, then carbon, then compress cost."""
        if obs.latency > 250:
            return "scale_up"
        if obs.latency > 170:
            return "scale_up"
        if obs.carbon > 320 and obs.latency <= 180:
            return "migrate_region"
        if obs.carbon > 260 and obs.latency <= 160:
            return "migrate_region"
        if obs.cost > 520 and obs.latency <= 150:
            return "scale_down"
        if obs.carbon > 220 and obs.latency <= 160:
            return "optimize_energy"
        if obs.cost > 450:
            return "scale_down" if obs.latency <= 150 else "scale_up"
        if obs.latency > 150:
            return "scale_up"
        if obs.cost > 400:
            return "scale_down"
        return "optimize_energy"

    def _decision_reason(
        self, obs: CloudObservation, votes: dict[str, int], winning_action: str
    ) -> str:
        """Explain whether the boardroom followed consensus or a recovery guardrail."""
        if votes.get(winning_action, 0) >= 2:
            return f"{votes[winning_action]} vote(s)"
        if obs.latency > 185 and winning_action == "scale_up":
            return "latency recovery"
        if obs.carbon > 220 and winning_action in {"optimize_energy", "migrate_region"}:
            return "carbon recovery"
        if obs.cost > 400 and winning_action in {"scale_down", "optimize_energy"}:
            return "cost recovery"
        return "goal guardrail"
