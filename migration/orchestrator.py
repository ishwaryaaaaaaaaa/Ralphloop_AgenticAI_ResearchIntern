"""The meta-agent's state machine (MetaAgent-style FSM) with a Ralph loop on VERIFY <-> FIX.

    EXTRACT -> PLAN -> GENERATE -> VERIFY --pass--> DONE
                                     |  ^
                                   fail |
                                     v  |
                                    FIX +   (at most max_iterations fixes, then FAILED)

The orchestrator knows nothing about how each stage works; the five stages are
passed in as plain functions, so they can be real agents or test fakes.
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from migration.schema import Sequence, VerifyResult


class State(str, Enum):
    EXTRACT = "EXTRACT"
    PLAN = "PLAN"
    GENERATE = "GENERATE"
    VERIFY = "VERIFY"
    FIX = "FIX"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class Context:
    """Everything the pipeline has produced so far; each state reads and writes part of it."""
    java_source: str
    sequence: Sequence | None = None
    plan: Any = None  # the meta-agent's list of subagent specs
    files: dict[str, str] = field(default_factory=dict)  # relative path -> file content
    verify: VerifyResult | None = None
    fixes_used: int = 0


class Orchestrator:
    def __init__(
        self,
        extract: Callable[[str], Sequence],
        plan: Callable[[Sequence], Any],
        generate: Callable[[Sequence, Any], dict[str, str]],
        verify: Callable[[dict[str, str]], VerifyResult],
        fix: Callable[[Sequence, Any, dict[str, str], VerifyResult], dict[str, str]],
        max_iterations: int = 5,
        log_dir: Path | None = Path("runs"),
    ):
        self.extract, self.plan, self.generate = extract, plan, generate
        self.verify, self.fix = verify, fix
        self.max_iterations = max_iterations
        self.log_dir = log_dir

    def run(self, java_source: str) -> Context:
        ctx = Context(java_source=java_source)
        state = State.EXTRACT
        log: list[dict] = []
        try:
            while state not in (State.DONE, State.FAILED):
                started = time.time()
                next_state, note = self._step(state, ctx)
                log.append({
                    "state": state.value, "next": next_state.value, "note": note,
                    "seconds": round(time.time() - started, 2),
                })
                state = next_state
        finally:
            self._write_log(log, state, ctx)
        return ctx

    def _step(self, state: State, ctx: Context) -> tuple[State, str]:
        """Run the work for one state; return (next state, short note for the log)."""
        if state is State.EXTRACT:
            ctx.sequence = self.extract(ctx.java_source)
            return State.PLAN, f"{len(ctx.sequence.steps)} steps extracted"

        if state is State.PLAN:
            assert ctx.sequence is not None
            ctx.plan = self.plan(ctx.sequence)
            return State.GENERATE, "plan ready"

        if state is State.GENERATE:
            assert ctx.sequence is not None
            ctx.files = self.generate(ctx.sequence, ctx.plan)
            return State.VERIFY, f"{len(ctx.files)} files generated"

        if state is State.VERIFY:
            ctx.verify = self.verify(ctx.files)
            if ctx.verify.passed:
                return State.DONE, "build and tests passed"
            if ctx.fixes_used >= self.max_iterations:
                return State.FAILED, f"still failing after {ctx.fixes_used} fixes"
            return State.FIX, "verification failed"

        if state is State.FIX:
            assert ctx.sequence is not None and ctx.verify is not None
            ctx.fixes_used += 1
            ctx.files = self.fix(ctx.sequence, ctx.plan, ctx.files, ctx.verify)
            return State.VERIFY, f"fix attempt {ctx.fixes_used}"

        raise ValueError(f"no handler for state {state}")

    def _write_log(self, log: list[dict], final: State, ctx: Context) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
        record = {
            "final_state": final.value,
            "fixes_used": ctx.fixes_used,
            "transitions": log,
            "last_verify_output": ctx.verify.output if ctx.verify else None,
        }
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
