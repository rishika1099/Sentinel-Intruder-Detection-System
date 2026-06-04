"""Email alerting with per-event-type cooldown and snapshot attachment."""
import smtplib
import ssl
import time
from email.message import EmailMessage

import cv2


class AlertManager:
    def __init__(self, settings):
        self.s = settings
        self._last_sent: dict[str, float] = {}

    def _cooldown_ok(self, key: str) -> bool:
        now = time.time()
        if now - self._last_sent.get(key, 0.0) >= self.s.alert_cooldown_s:
            self._last_sent[key] = now
            return True
        return False

    def configured(self) -> bool:
        return bool(self.s.smtp_user and self.s.smtp_password and self.s.alert_to)

    def send(self, subject: str, body: str, frame=None, key: str = "generic",
             force: bool = False):
        """Send an alert email. Returns (sent: bool, info: str)."""
        if not force and not self._cooldown_ok(key):
            return False, "cooldown"
        if not self.configured():
            return False, "email not configured"

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.s.alert_from or self.s.smtp_user
        msg["To"] = self.s.alert_to
        msg.set_content(body)

        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                msg.add_attachment(buf.tobytes(), maintype="image",
                                   subtype="jpeg", filename="snapshot.jpg")

        try:
            ctx = ssl.create_default_context()
            with smtplib.SMTP(self.s.smtp_host, self.s.smtp_port, timeout=15) as srv:
                srv.starttls(context=ctx)
                srv.login(self.s.smtp_user, self.s.smtp_password)
                srv.send_message(msg)
            return True, "sent"
        except Exception as e:                       # noqa: BLE001 - report to UI
            return False, str(e)
