"""
modules/alerts.py
Alert digests + optional email delivery.

- build_retail_digest(): reorder alerts for Small Retailer products
- build_enterprise_digest(): exception + audit digest for Enterprise mode
- send_email(): SMTP delivery when SMTP_* env vars are configured
  (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM);
  degrades gracefully — the digest is always downloadable in-app.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional


def _env(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val:
        return val
    for path in [".env", "logistics-ai-dashboard/.env"]:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(name + "="):
                        return line.split("=", 1)[1].strip()
    return None


def smtp_configured() -> bool:
    return bool(_env("SMTP_HOST") and _env("SMTP_FROM"))


def send_email(to_addr: str, subject: str, body: str) -> tuple[bool, str]:
    """Send a plain-text email via configured SMTP. Returns (ok, message)."""
    host = _env("SMTP_HOST")
    sender = _env("SMTP_FROM")
    if not host or not sender:
        return False, "SMTP not configured — set SMTP_HOST and SMTP_FROM in .env"
    port = int(_env("SMTP_PORT") or 587)
    user, password = _env("SMTP_USER"), _env("SMTP_PASS")

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to_addr, subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls(context=ssl.create_default_context())
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        return True, f"Sent to {to_addr}"
    except Exception as e:
        return False, f"Email send failed: {e}"


# ── Digest builders ────────────────────────────────────────────────────────────

def build_retail_digest(products: list[dict], tracker_rows: list[dict]) -> tuple[str, int]:
    """
    Build the Small Retailer reorder digest.
    tracker_rows come from retail.tracker_row() (Status / Current stock /
    Reorder when / Order qty columns). Returns (digest_text, n_alerts).
    """
    urgent, soon, ok = [], [], []
    for row in tracker_rows:
        status = str(row.get("Status", ""))
        line = (f"{row.get('Product', '?')}: {row.get('Current stock', 0):.0f} in stock, "
                f"reorder at {row.get('Reorder when (units left)', 0)}, "
                f"order qty {row.get('Order qty', 0)}")
        if "ORDER NOW" in status.upper():
            urgent.append(line)
        elif "SOON" in status.upper():
            soon.append(line)
        else:
            ok.append(line)

    lines = ["SUPCHAINMATE REORDER DIGEST", "=" * 28, ""]
    if urgent:
        lines.append(f"🔴 ORDER NOW ({len(urgent)}):")
        lines += [f"  - {l}" for l in urgent]
        lines.append("")
    if soon:
        lines.append(f"🟡 ORDER SOON ({len(soon)}):")
        lines += [f"  - {l}" for l in soon]
        lines.append("")
    lines.append(f"🟢 OK: {len(ok)} product(s)")
    if not urgent and not soon:
        lines.append("")
        lines.append("Nothing needs ordering right now.")
    return "\n".join(lines), len(urgent) + len(soon)


def build_enterprise_digest(exception_text: str, audit_text: Optional[str] = None,
                            health_text: Optional[str] = None) -> str:
    """Combine the enterprise digests into one email body."""
    parts = [exception_text]
    if audit_text:
        parts += ["", "", audit_text]
    if health_text:
        parts += ["", "", health_text]
    return "\n".join(parts)
