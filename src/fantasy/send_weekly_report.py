
import argparse
import html
import json
import os
import smtplib
import urllib.request

from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ESPN_SEASON_ID = 2027
TIMEZONE = ZoneInfo("America/Montreal")

RANKINGS_FILE = (
    PROJECT_ROOT / "data" / "gold"
    / f"available_player_rankings_{ESPN_SEASON_ID}.json"
)

REPORT_DIR = PROJECT_ROOT / "reports"

NHL_SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule"

TOP_N = 10
DETAIL_N = 3

WEIGHTS = {
    "season": 0.50,
    "l10": 0.30,
    "l5": 0.20,
}


def load_rankings():
    if not RANKINGS_FILE.exists():
        raise FileNotFoundError(
            f"Rankings file not found: {RANKINGS_FILE}"
        )

    with open(RANKINGS_FILE, "r", encoding="utf-8") as file:
        rankings = json.load(file)

    if not isinstance(rankings, list):
        raise ValueError("Expected a JSON list of ranked players.")

    if not rankings:
        raise ValueError("Ranking file contains no players.")

    return rankings[:TOP_N]


def fetch_nhl_schedule(start_date, end_date):
    url = f"{NHL_SCHEDULE_URL}/{start_date.isoformat()}"

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NHL-Fantasy-Data-Platform/1.0"}
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    game_days = payload.get("gameWeek", [])

    if not game_days:
        raise ValueError("NHL API returned no schedule data.")

    games = []

    for day in game_days:
        game_date = datetime.strptime(
            day["date"], "%Y-%m-%d"
        ).date()

        if not start_date <= game_date < end_date:
            continue

        for game in day.get("games", []):
            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})

            games.append({
                "date": game_date,
                "time_utc": game.get("startTimeUTC"),
                "home": home.get("abbrev"),
                "away": away.get("abbrev"),
                "game_id": game.get("id"),
            })

    return games


def get_player_schedule(player, games):
    team = player.get("team")

    if not team:
        return []

    return [
        game for game in games
        if game["home"] == team or game["away"] == team
    ]


def format_number(value, decimals=2):
    try:
        return f"{float(value or 0):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def format_game_time(game):
    if not game.get("time_utc"):
        return "Time TBA"

    try:
        utc_time = datetime.fromisoformat(
            game["time_utc"].replace("Z", "+00:00")
        )

        local_time = utc_time.astimezone(TIMEZONE)

        return local_time.strftime("%a %b %d, %I:%M %p")
    except ValueError:
        return "Time TBA"


def build_player_rationale(player):
    season = float(player.get("fantasy_points_per_game") or 0)
    l10 = float(player.get("fantasy_points_l10") or 0)
    l5 = float(player.get("fantasy_points_l5") or 0)

    contributions = {
        "season": season * WEIGHTS["season"],
        "l10": l10 * WEIGHTS["l10"],
        "l5": l5 * WEIGHTS["l5"],
    }

    strongest_factor = max(
        contributions,
        key=contributions.get
    )

    factor_names = {
        "season": "season production",
        "l10": "last 10 games",
        "l5": "last 5 games",
    }

    recent_difference = l5 - season

    if recent_difference > 0.05:
        trend = "Recent production is above the season average."
    elif recent_difference < -0.05:
        trend = "Recent production is below the season average."
    else:
        trend = "Recent production is close to the season average."

    return (
        f"The strongest contribution to this player's ranking "
        f"comes from {factor_names[strongest_factor]}. "
        f"{trend}"
    )


def build_schedule_html(player, games):
    player_games = get_player_schedule(player, games)

    if not player_games:
        return "<p>No games scheduled during this period.</p>"

    rows = []

    for game in sorted(player_games, key=lambda item: item["date"]):
        opponent = (
            game["away"]
            if game["home"] == player["team"]
            else game["home"]
        )

        location = "Home" if game["home"] == player["team"] else "Away"

        rows.append(
            "<tr>"
            f"<td>{html.escape(format_game_time(game))}</td>"
            f"<td>{html.escape(opponent or 'TBD')}</td>"
            f"<td>{location}</td>"
            "</tr>"
        )

    return (
        "<table>"
        "<tr><th>Date</th><th>Opponent</th><th>Location</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def build_report_html(rankings, games, start_date, end_date):
    top_players = rankings[:DETAIL_N]

    ranking_rows = []

    for player in rankings:
        ranking_rows.append(
            "<tr>"
            f"<td>{player['rank']}</td>"
            f"<td>{html.escape(player.get('player_name') or '')}</td>"
            f"<td>{html.escape(player.get('team') or '')}</td>"
            f"<td>{html.escape(player.get('position') or '')}</td>"
            f"<td>{player.get('games_played', 0)}</td>"
            f"<td>{format_number(player.get('fantasy_points_per_game'))}</td>"
            f"<td>{format_number(player.get('fantasy_points_l10'))}</td>"
            f"<td>{format_number(player.get('fantasy_points_l5'))}</td>"
            f"<td>{format_number(player.get('ranking_score'), 3)}</td>"
            "</tr>"
        )

    detail_sections = []

    for player in top_players:
        name = html.escape(player.get("player_name") or "")
        team = html.escape(player.get("team") or "")

        rationale = html.escape(build_player_rationale(player))

        detail_sections.append(
            f"""
            <h3>#{player['rank']} - {name} ({team})</h3>

            <p>
                Games played: {player.get('games_played', 0)}<br>
                Season average:
                {format_number(player.get('fantasy_points_per_game'))}<br>
                Last 10 games:
                {format_number(player.get('fantasy_points_l10'))}<br>
                Last 5 games:
                {format_number(player.get('fantasy_points_l5'))}<br>
                Ranking score:
                {format_number(player.get('ranking_score'), 3)}
            </p>

            <p><strong>Why this player ranks here:</strong>
            {rationale}</p>

            <strong>Upcoming schedule</strong>
            {build_schedule_html(player, games)}
            """
        )

    remaining_players = rankings[DETAIL_N:]

    remaining_schedules = []

    for player in remaining_players:
        name = html.escape(player.get("player_name") or "")
        team = html.escape(player.get("team") or "")

        schedule = build_schedule_html(player, games)

        remaining_schedules.append(
            f"<h4>{name} ({team})</h4>{schedule}"
        )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                color: #222;
                line-height: 1.5;
                max-width: 900px;
                margin: auto;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0 25px;
            }}

            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}

            th {{
                background-color: #f0f0f0;
            }}

            h2 {{
                border-bottom: 2px solid #ddd;
                padding-bottom: 6px;
                margin-top: 30px;
            }}

            .note {{
                color: #666;
                font-size: 13px;
            }}
        </style>
    </head>

    <body>
        <h1>NHL Fantasy - Weekly Waiver Report</h1>

        <p>
            <strong>Reporting period:</strong>
            {start_date} to {end_date - timedelta(days=1)}
        </p>

        <p>
            Top available players in your ESPN Fantasy league,
            ranked using current-season and recent fantasy production.
        </p>

        <h2>Top 10 Available Players</h2>

        <table>
            <tr>
                <th>Rank</th>
                <th>Player</th>
                <th>Team</th>
                <th>Pos</th>
                <th>GP</th>
                <th>Season</th>
                <th>L10</th>
                <th>L5</th>
                <th>Score</th>
            </tr>

            {''.join(ranking_rows)}
        </table>

        <p class="note">
            Ranking formula:
            50% season average + 30% last 10 games
            + 20% last 5 games.
        </p>

        <h2>Top 3 Player Analysis</h2>

        {''.join(detail_sections)}

        <h2>Upcoming Schedule - Remaining Top 10</h2>

        {''.join(remaining_schedules)}

        <p class="note">
            Data source: NHL API and ESPN Fantasy available-player data.
            All game times are displayed in Montréal local time.
        </p>
    </body>
    </html>
    """


def save_report(report_html, start_date):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / (
        f"weekly_fantasy_report_{start_date.isoformat()}.html"
    )

    output_file.write_text(report_html, encoding="utf-8")

    return output_file


def send_email(report_html, start_date):
    username = os.getenv("MAIL_USERNAME")
    password = os.getenv("MAIL_PASSWORD")
    recipient = os.getenv("MAIL_TO")

    if not username or not password or not recipient:
        raise EnvironmentError(
            "Missing MAIL_USERNAME, MAIL_PASSWORD or MAIL_TO."
        )

    message = EmailMessage()

    message["Subject"] = (
        f"NHL Fantasy Weekly Waiver Report - {start_date}"
    )
    message["From"] = username
    message["To"] = recipient

    message.set_content(
        "Your weekly NHL Fantasy waiver report is available "
        "in the HTML version of this email."
    )

    message.add_alternative(report_html, subtype="html")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(username, password)
        smtp.send_message(message)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--send",
        action="store_true",
        help="Send the report by email. Without this flag, preview only."
    )

    args = parser.parse_args()

    now = datetime.now(TIMEZONE)
    start_date = now.date()
    end_date = start_date + timedelta(days=7)

    print("Loading available-player rankings...")

    rankings = load_rankings()

    print(f"Players loaded: {len(rankings)}")

    print("Fetching NHL schedule...")

    games = fetch_nhl_schedule(start_date, end_date)

    print(f"Games found in reporting period: {len(games)}")

    report_html = build_report_html(
        rankings,
        games,
        start_date,
        end_date
    )

    output_file = save_report(report_html, start_date)

    print("\nREPORT GENERATED")
    print(output_file)

    if args.send:
        print("\nSending email...")

        send_email(report_html, start_date)

        print("EMAIL SENT SUCCESSFULLY!")
    else:
        print("\nPREVIEW MODE - Email was not sent.")
        print("Use --send to send the email.")


if __name__ == "__main__":
    main()
