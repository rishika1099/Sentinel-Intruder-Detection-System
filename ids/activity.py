"""Incident tracker: summarise what an intruder was *doing*.

Rather than logging one identical line per frame, this groups a continuous
stretch of activity into a single "incident" and describes the behaviour:
how long it lasted, whether the person approached or moved away, which way
they moved across the view, whether they were armed and with what, whether
they were known or unidentified, and whether fire was involved.

Usage per frame:
    tracker.update(events, persons, faces, weapons, reading, frame_w)
    closed = tracker.maybe_close()      # returns a summary string when an
                                        # incident ends, else None
    live = tracker.live_summary()       # short description of the ongoing one
"""
import time

# event types that count as ongoing activity worth narrating
_ACTIVITY = {"INTRUDER", "ARMED_INTRUDER", "WEAPON", "FIRE", "HAZARD"}


class IncidentTracker:
    def __init__(self, idle_end_s: float = 4.0, loiter_s: float = 6.0):
        self.idle_end_s = idle_end_s
        self.loiter_s = loiter_s
        self._reset()

    def _reset(self):
        self.open = False
        self.start = self.last = 0.0
        self.dist_start = self.dist_min = self.dist_last = None
        self.x_start = self.x_last = None      # normalised person center-x
        self.weapons: set[str] = set()
        self.known_names: set[str] = set()
        self.saw_unknown = False
        self.max_people = 0
        self.fire = False
        self.hazard = False

    def update(self, events, persons, faces, weapons, reading, frame_w,
               now: float | None = None):
        now = now or time.time()
        active = bool(persons) or bool(weapons) or any(
            e.type in _ACTIVITY for e in events)
        if not active:
            return

        if not self.open:
            self.open = True
            self.start = now
            self.dist_start = reading.distance_cm
            self.dist_min = reading.distance_cm
        self.last = now

        self.dist_last = reading.distance_cm
        self.dist_min = min(self.dist_min, reading.distance_cm)

        if persons:
            self.max_people = max(self.max_people, len(persons))
            box = max(persons, key=lambda p: (p["box"][2] - p["box"][0]))["box"]
            cx = (box[0] + box[2]) / 2 / max(1, frame_w)
            if self.x_start is None:
                self.x_start = cx
            self.x_last = cx

        for w in weapons:
            self.weapons.add(w["label"])
        for f in faces:
            if f["known"]:
                self.known_names.add(f["name"])
            else:
                self.saw_unknown = True
        for e in events:
            if e.type == "FIRE":
                self.fire = True
            elif e.type == "HAZARD":
                self.hazard = True

    def maybe_close(self, now: float | None = None):
        now = now or time.time()
        if self.open and (now - self.last) >= self.idle_end_s:
            summary = self._summary()
            self._reset()
            return summary
        return None

    def force_close(self):
        """Close an open incident immediately (e.g. when monitoring stops)."""
        if self.open:
            summary = self._summary()
            self._reset()
            return summary
        return None

    # ---------------------------------------------------------------- narrative
    def live_summary(self):
        if not self.open:
            return None
        return self._describe(ongoing=True)

    def _summary(self):
        return self._describe(ongoing=False)

    def _subject(self):
        if self.weapons:
            who = "armed person"
            if self.known_names and not self.saw_unknown:
                who = f"armed known person ({', '.join(sorted(self.known_names))})"
            return who
        if self.max_people:
            if self.known_names and not self.saw_unknown:
                return f"known person ({', '.join(sorted(self.known_names))})"
            return "unidentified person"
        if self.fire:
            return "fire"
        return "activity"

    def _movement(self):
        parts = []
        if self.dist_start is not None and self.dist_last is not None:
            delta = self.dist_last - self.dist_start
            if delta < -40:
                parts.append(f"approached ({self.dist_start:.0f}->{self.dist_min:.0f}cm)")
            elif delta > 40:
                parts.append(f"moved away ({self.dist_start:.0f}->{self.dist_last:.0f}cm)")
            else:
                parts.append(f"stayed ~{self.dist_last:.0f}cm")
        dur = max(0.0, self.last - self.start)
        if dur >= self.loiter_s and (self.dist_start is None or
                                     abs((self.dist_last or 0) - (self.dist_start or 0)) < 40):
            parts.append("loitered")
        if self.x_start is not None and self.x_last is not None:
            dx = self.x_last - self.x_start
            if dx > 0.2:
                parts.append("crossed view L->R")
            elif dx < -0.2:
                parts.append("crossed view R->L")
        return parts

    def _describe(self, ongoing: bool):
        dur = max(0.0, self.last - self.start)
        bits = [self._subject()]
        bits += self._movement()
        if self.weapons:
            bits.append("carrying " + ", ".join(sorted(self.weapons)))
        if self.fire:
            bits.append("fire present")
        if self.hazard and not self.fire:
            bits.append("smoke/heat hazard")
        body = "; ".join(bits)
        if ongoing:
            return f"[{dur:.0f}s] {body}"
        t0 = time.strftime("%H:%M:%S", time.localtime(self.start))
        t1 = time.strftime("%H:%M:%S", time.localtime(self.last))
        return f"{t0}-{t1} ({dur:.0f}s): {body}"
