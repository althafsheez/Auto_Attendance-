# =====================================
# duration_tracker.py
# Keeps track of how long each person is visible
# =====================================
import time

class DurationTracker:
    def __init__(self):
        # Store time data: {name: {"first_seen": t1, "last_seen": t2}}
        self.times = {}

    def update(self, name):
        """Update the timestamps when a person is detected."""
        now = time.time()

        if name not in self.times:
            # First time this person is seen
            self.times[name] = {"first_seen": now, "last_seen": now}
        else:
            # Update last seen time
            self.times[name]["last_seen"] = now

    def get_durations(self):
        """Return total duration (seconds) each person was visible."""
        durations = {}
        for name, data in self.times.items():
            duration = data["last_seen"] - data["first_seen"]
            durations[name] = round(duration, 2)
        return durations
