"""Extractor agent: legacy Java source in, validated executional `Sequence` out."""
from migration import llm
from migration.schema import Sequence

SYSTEM = """\
You are a reverse-engineering agent. You read legacy Java GUI code (an applet) and describe what it \
does as an ordered, structured sequence of steps, so that another agent can rebuild it in a modern \
framework without seeing the Java.

Rules:
- Describe observable behaviour, not Java syntax. Give each distinct behaviour block a short, unique \
snake_case `id` and a concise description of what it does.
- Set `trigger` to the name of the user-driven event handler that actually contains the behavior. \
Ignore empty interface/listener stubs; identify the real handler instead.
- For the trigger's execution path, make a step for each behavior block in execution order. Link \
unconditional flow with `next`. For a decision, set `condition` and both \
`next_if_true` and `next_if_false`; use null for an outcome that stops or returns. Do not set `next` \
on a conditional step. Represent early returns as branch outcomes that stop.
- Make a separate step for a helper method when it is called from multiple places, and reference that \
step from each caller's `calls`. Every flow target and call target must be an existing step `id`.
- Include constructors, initialization (`init` and static initializers), and drawing/paint behavior as \
steps too. Keep these standalone steps out of the trigger's `next` chain unless the Java execution \
actually flows into them; they do not need a `next` link otherwise.
- In `state`, list every field whose value persists between events, with its exact Java type and a \
meaningful description (including what any bitmask or encoded value represents). `reads` and `changes` \
must contain only names from `state`; use them for persistent fields, not method-local variables.
- Preserve exact behavior and rules, including priorities, ordering, constants, tie-breaking, turn or \
game alternation, and randomness. Do not infer a rule that the source does not establish.
- Record non-portable effects (such as sound, image loading, or deprecated API use) in the relevant \
step's `side_effects`; do not create steps whose only purpose is to describe such an effect.
- Set `source_lines` to the inclusive Java source line range for that step, using the line numbers \
shown in the input. Use a single line number when the behavior comes from one line.
"""


def _number_lines(source: str) -> str:
    return "\n".join(f"{i:4d}  {line}" for i, line in enumerate(source.splitlines(), start=1))


def extract(java_source: str) -> Sequence:
    """Run the extractor agent. Raises if the model cannot produce a valid Sequence."""
    user = f"Java source (line numbers on the left):\n\n{_number_lines(java_source)}"
    return llm.ask_json(SYSTEM, user, Sequence)
