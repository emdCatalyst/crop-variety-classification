"""Lightweight email delivery: Gmail SMTP (or any STARTTLS host) with a
console fallback when no credentials are configured. Plain-text bodies only.

Failures never raise — auth flows must succeed even if SMTP is misbehaving;
the user can request a resend.
"""
from __future__ import annotations

from email.message import EmailMessage

from loguru import logger

from ..core.config import get_settings
from ..models import User


def _subject_and_body(kind: str, code: str, lang: str, ttl_min: int) -> tuple[str, str]:
    lang = lang if lang in ("en", "fr", "ar") else "en"
    if kind == "verify":
        if lang == "fr":
            return (
                "Votre code de vérification Agro-Vision",
                f"Bienvenue sur Agro-Vision !\n\nVotre code de vérification est : {code}\n"
                f"Ce code expire dans {ttl_min} minutes.\n\nSi vous n'avez pas créé ce compte, ignorez ce message.",
            )
        if lang == "ar":
            return (
                "رمز التحقق الخاص بك في أجرو-فيجن",
                f"أهلاً بك في أجرو-فيجن!\n\nرمز التحقق الخاص بك هو: {code}\n"
                f"تنتهي صلاحية هذا الرمز خلال {ttl_min} دقيقة.\n\nإذا لم تُنشئ هذا الحساب، يمكنك تجاهل هذه الرسالة.",
            )
        return (
            "Your Agro-Vision verification code",
            f"Welcome to Agro-Vision!\n\nYour verification code is: {code}\n"
            f"This code expires in {ttl_min} minutes.\n\nIf you didn't create this account, you can ignore this email.",
        )
    # reset
    if lang == "fr":
        return (
            "Réinitialisation de votre mot de passe Agro-Vision",
            f"Vous avez demandé à réinitialiser votre mot de passe.\n\nVotre code est : {code}\n"
            f"Ce code expire dans {ttl_min} minutes.\n\nSi vous n'êtes pas à l'origine de cette demande, ignorez ce message.",
        )
    if lang == "ar":
        return (
            "إعادة تعيين كلمة مرور أجرو-فيجن",
            f"لقد طلبت إعادة تعيين كلمة المرور.\n\nرمزك هو: {code}\n"
            f"تنتهي صلاحية هذا الرمز خلال {ttl_min} دقيقة.\n\nإذا لم تطلب ذلك، فتجاهل هذه الرسالة.",
        )
    return (
        "Reset your Agro-Vision password",
        f"You requested a password reset.\n\nYour code is: {code}\n"
        f"This code expires in {ttl_min} minutes.\n\nIf you didn't request this, you can ignore this email.",
    )


async def send_email(to: str, subject: str, body: str) -> None:
    s = get_settings()
    if not s.smtp_user or not s.smtp_password:
        logger.info("[EMAIL fallback] to={} subject={!r}\n{}", to, subject, body)
        return

    msg = EmailMessage()
    from_email = s.smtp_from_email or s.smtp_user
    msg["From"] = f"{s.smtp_from_name} <{from_email}>"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        # Imported lazily so the dependency is only required when SMTP is on.
        import aiosmtplib

        await aiosmtplib.send(
            msg,
            hostname=s.smtp_host,
            port=s.smtp_port,
            start_tls=True,
            username=s.smtp_user,
            password=s.smtp_password,
        )
        logger.info("Sent email to={} subject={!r}", to, subject)
    except Exception as exc:  # noqa: BLE001 — never break auth on email failure
        logger.warning("SMTP send failed to={} subject={!r}: {}", to, subject, exc)


async def send_verification_email(user: User, code: str, lang: str) -> None:
    ttl = get_settings().email_code_ttl_minutes
    subject, body = _subject_and_body("verify", code, lang, ttl)
    await send_email(user.email, subject, body)


async def send_password_reset_email(user: User, code: str, lang: str) -> None:
    ttl = get_settings().email_code_ttl_minutes
    subject, body = _subject_and_body("reset", code, lang, ttl)
    await send_email(user.email, subject, body)
