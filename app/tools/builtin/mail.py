"""
邮件工具：send_email（QQ 邮箱 SMTP）。

配置（.env）：
    SMTP_HOST=smtp.qq.com
    SMTP_PORT=465
    SMTP_USER=<发件人QQ邮箱，如 123456@qq.com>
    SMTP_PASSWORD=<SMTP 授权码，QQ 邮箱设置里生成>

安全：
    - 凭据全部走环境变量，代码零硬编码；
    - 收件人/主题/正文由模型生成，长度限制防滥用；
    - 发送失败返回结构化错误（Gateway 兜底）。
"""
import smtplib
import ssl
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr

from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import ToolExecutionError

# 正文 / 主题长度限制
MAX_SUBJECT = 200
MAX_BODY = 8000


class SendEmailArgs(BaseModel):
    """发送邮件参数。"""

    to: str = Field(description="收件人邮箱地址，如 user@example.com")
    subject: str = Field(description="邮件主题（限 200 字符）")
    body: str = Field(description="邮件正文（限 8000 字符）")


def _smtp_settings() -> tuple[str, int, str, str]:
    """读取 SMTP 配置；缺任一关键项即报错（提示先配置 .env）。"""
    s = Settings()
    host = s.smtp_host
    port = s.smtp_port
    user = s.smtp_user
    password = s.smtp_password
    missing = [name for name, v in
               (("SMTP_HOST", host), ("SMTP_USER", user), ("SMTP_PASSWORD", password))
               if not v]
    if missing:
        raise ToolExecutionError(
            f"SMTP 配置缺失: {', '.join(missing)}。请在 .env 配置 SMTP_HOST / SMTP_USER / SMTP_PASSWORD"
        )
    return host, port, user, password


def send_email_handler(to: str, subject: str, body: str) -> str:
    """发送一封文本邮件，返回发送结果摘要。"""
    to = to.strip()
    subject = subject.strip()
    if "@" not in to or "." not in to.split("@")[-1]:
        raise ToolExecutionError(f"收件人地址非法: {to}")
    if len(subject) > MAX_SUBJECT:
        raise ToolExecutionError(f"主题超过 {MAX_SUBJECT} 字符")
    if len(body) > MAX_BODY:
        raise ToolExecutionError(f"正文超过 {MAX_BODY} 字符")

    host, port, user, password = _smtp_settings()

    # 中文邮件头：主题需要 Header 编码，发件人显示名用 formataddr
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = formataddr((str(Header("Agent Runtime", "utf-8")), user))
    msg["To"] = to

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, timeout=15, context=ctx) as server:
            server.login(user, password)
            server.sendmail(user, [to], msg.as_string())
    except smtplib.SMTPAuthenticationError as exc:
        raise ToolExecutionError(f"SMTP 认证失败（授权码错误？）: {exc.smtp_code}") from exc
    except smtplib.SMTPException as exc:
        raise ToolExecutionError(f"邮件发送失败: {exc}") from exc
    except OSError as exc:
        raise ToolExecutionError(f"SMTP 连接失败: {exc}", transient=True) from exc

    return f"邮件已发送至 {to}（主题：{subject[:50]}）"
