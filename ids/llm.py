"""LLM analyst for Sentinel IDS, powered by the Claude API (Anthropic SDK).

Three capabilities, all through one client:
  1. describe_alert(frame, context) -> vision: looks at the alert snapshot plus
     the sensor/detection data and writes a scene description + threat assessment.
  2. narrate(context) -> text-only natural-language situation report.
  3. ask(question, incident_log) -> Q&A over the logged incidents.

Needs ANTHROPIC_API_KEY in the environment / .env. If the key is missing or the
anthropic package isn't installed, `available` is False and the app silently
falls back to the rule-based text (no LLM).
"""
import base64
import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)            # .env wins over stale/empty shell vars
except ImportError:                       # pragma: no cover
    pass

try:
    import anthropic
    _HAS_SDK = True
except ImportError:                       # pragma: no cover
    _HAS_SDK = False

MODEL = "claude-opus-4-8"

_SYSTEM_ANALYST = (
    "You are the analyst for Sentinel IDS, a home security system. You receive "
    "structured sensor and computer-vision detections (and sometimes a camera "
    "snapshot). Write a SHORT situation report (2-4 sentences) for the property "
    "owner: describe what is happening (people, what they are holding or doing, "
    "distance), state the threat level and why, and if it is an emergency advise "
    "contacting authorities. Be factual and specific; do not speculate beyond the "
    "evidence. Output only the report text - no preamble, headings, or meta-commentary."
)

_SYSTEM_QA = (
    "You are the analyst for Sentinel IDS, a home security system. Answer the "
    "owner's question using ONLY the incident log provided. Be concise and "
    "factual. If the log does not contain the answer, say so plainly. Output only "
    "the answer - no preamble."
)


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def format_context(context: dict) -> str:
    """Turn the engine's structured data into a compact prompt block."""
    lines = []
    r = context.get("reading")
    if r is not None:
        lines.append(
            f"Sensors: distance={r.distance_cm:.0f}cm, motion={'yes' if r.motion else 'no'}, "
            f"smoke={r.smoke_ppm:.0f}ppm, temperature={r.temperature_c:.0f}C")
    lines.append(f"People detected: {context.get('people', 0)}")
    weapons = context.get("weapons") or []
    if weapons:
        lines.append("Weapons detected: " + ", ".join(
            f"{w['label']} ({w['conf']:.2f})" for w in weapons))
    faces = context.get("faces") or []
    if faces:
        lines.append("Faces: " + ", ".join(
            (f["name"] if f["known"] else "UNKNOWN") for f in faces))
    if context.get("fire"):
        lines.append("Fire detected in frame: yes")
    events = context.get("events") or []
    if events:
        lines.append("Events: " + "; ".join(
            f"{e.type}/{e.severity} - {e.message}" for e in events))
    if context.get("activity"):
        lines.append(f"Activity so far: {context['activity']}")
    return "\n".join(lines) or "No notable detections."


class ThreatNarrator:
    def __init__(self, model: str = MODEL):
        self.model = model
        self.available = _HAS_SDK and bool(os.environ.get("ANTHROPIC_API_KEY"))
        self.client = anthropic.Anthropic() if self.available else None

    # ------------------------------------------------------------ vision
    def describe_alert(self, frame_jpeg: bytes | None, context: dict) -> str:
        """Vision (or text-only if frame is None): describe + assess the scene."""
        if not self.client:
            return ""
        content = []
        if frame_jpeg:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(frame_jpeg).decode(),
                },
            })
        content.append({"type": "text", "text": format_context(context)})
        return self._call(_SYSTEM_ANALYST, content)

    # ------------------------------------------------------------ text
    def narrate(self, context: dict) -> str:
        if not self.client:
            return ""
        return self._call(_SYSTEM_ANALYST,
                          [{"type": "text", "text": format_context(context)}])

    # ------------------------------------------------------------ Q&A
    def ask(self, question: str, incident_log: list[str]) -> str:
        if not self.client:
            return "AI is not configured (set ANTHROPIC_API_KEY)."
        log = "\n".join(incident_log) or "(no incidents recorded yet)"
        prompt = f"Incident log:\n{log}\n\nQuestion: {question}"
        return self._call(_SYSTEM_QA, [{"type": "text", "text": prompt}])

    # ------------------------------------------------------------ internal
    def _call(self, system: str, content: list) -> str:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                thinking={"type": "disabled"},   # fast; alerts are latency-sensitive
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
            )
            return _text(resp)
        except anthropic.APIError as e:           # pragma: no cover
            return f"(AI unavailable: {getattr(e, 'message', str(e))})"
        except Exception as e:                    # pragma: no cover
            return f"(AI error: {e})"
