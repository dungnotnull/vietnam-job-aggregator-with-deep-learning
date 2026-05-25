from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel

console = Console()

AUTH_MATRIX: dict[str, bool] = {
    "LinkedIn": False,
    "VietnamWorks": False,
    "ITViec": False,
    "TopDev": False,
}

LOGIN_URLS: dict[str, str] = {
    "LinkedIn": "https://www.linkedin.com/login",
}


def prompt_user_login(platform: str, browser: Any) -> None:
    if not AUTH_MATRIX.get(platform, False):
        return

    login_url = LOGIN_URLS.get(platform)
    if not login_url:
        return

    page = browser.new_page()
    page.goto(login_url, wait_until="domcontentloaded")

    console.print(
        Panel.fit(
            f"[bold yellow]Authentication Required[/bold yellow]\n\n"
            f"Platform: [cyan]{platform}[/cyan]\n"
            f"A browser window has been opened to [underline]{login_url}[/underline]\n\n"
            f"Please log in manually in the browser, then press [green]Enter[/green] here to continue.",
            title="Auth Prompt",
        )
    )
    input()

    console.print(f"[green]✓[/green] Authentication confirmed for {platform}")
