"""Shape of the executional sequence the extractor agent must produce.

The extractor reads legacy Java and returns JSON; `Sequence.model_validate_json`
accepts it or raises, so malformed output never reaches the meta-agent.
"""
from pydantic import BaseModel, model_validator


class StateVar(BaseModel):
    """One piece of program state, e.g. `white` (a bitmask of the computer's squares)."""
    name: str
    java_type: str
    meaning: str


class Step(BaseModel):
    """One block of behaviour, described as a card."""
    id: str
    description: str
    source_lines: str  # e.g. "268-280", so a reviewer can check the card against the Java
    reads: list[str] = []
    changes: list[str] = []
    calls: list[str] = []  # other steps this one uses as a helper
    # Flow: unconditional `next`, or a `condition` with two outcomes. None means "stop".
    next: str | None = None
    condition: str | None = None
    next_if_true: str | None = None
    next_if_false: str | None = None
    side_effects: list[str] = []  # things to drop or replace, e.g. "plays a sound"

    @model_validator(mode="after")
    def _flow_is_consistent(self):
        if self.condition is None and (self.next_if_true or self.next_if_false):
            raise ValueError(f"step {self.id}: branch targets given without a condition")
        if self.condition is not None and self.next is not None:
            raise ValueError(f"step {self.id}: use either `next` or `condition`, not both")
        return self


class VerifyResult(BaseModel):
    """What the verifier reports after building and testing the generated app."""
    passed: bool
    output: str  # build/test log; on failure this is what the fixer agent reads


class Sequence(BaseModel):
    """Everything that happens for one trigger (e.g. a mouse release), plus the state it touches."""
    trigger: str
    state: list[StateVar]
    steps: list[Step]

    @model_validator(mode="after")
    def _references_resolve(self):
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step ids")
        known = set(ids)
        for s in self.steps:
            targets = [s.next, s.next_if_true, s.next_if_false, *s.calls]
            for t in targets:
                if t is not None and t not in known:
                    raise ValueError(f"step {s.id} points to unknown step {t!r}")
        return self
