from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from agent.filters import LEVELS, build_query
from agent.output_writer import write_report
from agent.scrapers.base_scraper import Job, ScraperError
from agent.scrapers.itviec_scraper import ITViecScraper
from agent.scrapers.linkedin_scraper import LinkedInScraper
from agent.scrapers.topdev_scraper import TopDevScraper
from agent.scrapers.vietnamworks_scraper import VietnamWorksScraper
from agent.security import run_security_audit

console = Console()

SCRAPERS: list[tuple[str, object]] = [
    ("LinkedIn", LinkedInScraper()),
    ("VietnamWorks", VietnamWorksScraper()),
    ("ITViec", ITViecScraper()),
    ("TopDev", TopDevScraper()),
]

COUNT_OPTIONS = {
    "1": 10,
    "2": 20,
    "3": 50,
    "4": 100,
    "5": 999,
}


def print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Vietnam Job Aggregator[/bold cyan]\n\n"
            "[dim]Search job listings across LinkedIn, VietnamWorks, ITViec, and TopDev[/dim]\n"
            "[dim]Get AI-powered skill intelligence reports.[/dim]",
            title="Job Search Agent",
            border_style="cyan",
        )
    )


def ask_field() -> str:
    console.print()
    field = Prompt.ask("  [bold]1.[/bold] Field / Industry", default="Data Engineer")
    return field.strip()


def ask_level() -> str:
    console.print()
    table = Table(title="Level Options", show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("Level", width=15)
    for i, lvl in enumerate(LEVELS, 1):
        table.add_row(str(i), lvl)
    console.print(table)

    choice = Prompt.ask("  [bold]2.[/bold] Level (enter number or name)", default="Senior")

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(LEVELS):
            return LEVELS[idx]

    for lvl in LEVELS:
        if lvl.lower() == choice.lower():
            return lvl

    console.print(f"[yellow]Unknown level '{choice}', defaulting to Senior[/yellow]")
    return "Senior"


def ask_count() -> int:
    console.print()
    console.print("  [bold]3.[/bold] Results per platform:")
    for key, val in COUNT_OPTIONS.items():
        label = "All" if val == 999 else str(val)
        console.print(f"     [[{key}]] {label}")

    choice = Prompt.ask("  Select", default="2")
    count = COUNT_OPTIONS.get(choice, 20)
    if count == 999:
        count = 100
    return count


def ask_skill_analysis() -> bool:
    console.print()
    console.print("  [bold]4.[/bold] Generate Skill Intelligence Report?")
    console.print("     Uses ML model (if trained) or rule-based fallback")
    choice = Prompt.ask("  Run skill analysis?", choices=["y", "n"], default="y")
    return choice.lower() == "y"


def run_scrapers_concurrent(
    field: str, level: str, count: int
) -> dict[str, list[Job]]:
    all_jobs: dict[str, list[Job]] = {}
    platform_names = [p for p, _ in SCRAPERS]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        tasks: dict[str, int] = {}
        for platform in platform_names:
            tasks[platform] = progress.add_task(
                f"Searching {platform}...", total=None
            )

        with ThreadPoolExecutor(max_workers=len(SCRAPERS)) as executor:
            future_map = {
                executor.submit(scraper.search, field, level, count): platform
                for platform, scraper in SCRAPERS
            }

            for future in as_completed(future_map):
                platform = future_map[future]
                try:
                    jobs = future.result()
                    all_jobs[platform] = jobs
                    progress.update(
                        tasks[platform],
                        description=f"[green]✓[/green] {platform}: {len(jobs)} jobs found",
                        completed=True,
                    )
                except ScraperError as e:
                    all_jobs[platform] = []
                    progress.update(
                        tasks[platform],
                        description=f"[red]✗[/red] {platform}: {e}",
                        completed=True,
                    )
                except NotImplementedError:
                    all_jobs[platform] = []
                    progress.update(
                        tasks[platform],
                        description=f"[yellow]⊘[/yellow] {platform}: not yet supported",
                        completed=True,
                    )
                except Exception as e:
                    all_jobs[platform] = []
                    progress.update(
                        tasks[platform],
                        description=f"[red]✗[/red] {platform}: {e}",
                        completed=True,
                    )

    return all_jobs


def print_summary(all_jobs: dict[str, list[Job]]) -> None:
    total = sum(len(jobs) for jobs in all_jobs.values())
    console.print()
    table = Table(title="Search Summary", header_style="bold")
    table.add_column("Platform", style="cyan")
    table.add_column("Results", justify="right")
    for platform, jobs in all_jobs.items():
        table.add_row(platform, str(len(jobs)))
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
    console.print(table)


def main() -> None:
    if not run_security_audit():
        console.print("[red]Security audit failed. Exiting.[/red]")
        sys.exit(1)

    console.print("[green]✓[/green] Security audit passed")
    print_banner()

    field = ask_field()
    level = ask_level()
    count = ask_count()
    run_skill = ask_skill_analysis()

    build_query(field, level)

    console.print()
    console.print(
        f"[bold]Searching:[/bold] {field} | {level} | {count} results per platform"
    )
    console.print()

    all_jobs = run_scrapers_concurrent(field, level, count)
    print_summary(all_jobs)

    filepath = write_report(field, level, all_jobs, include_skill_analysis=run_skill)
    console.print()
    console.print(f"[bold green]✓[/bold green] Report saved to [cyan]{filepath}[/cyan]")


if __name__ == "__main__":
    main()
