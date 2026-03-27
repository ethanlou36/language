import requests
from bs4 import BeautifulSoup


API_URL = "https://codeforces.com/api/problemset.problems"
PROBLEM_URL_PATTERNS = (
    "https://mirror.codeforces.com/problemset/problem/{contest_id}/{index}",
    "https://mirror.codeforces.com/contest/{contest_id}/problem/{index}",
    "https://mirror.codeforces.com/gym/{contest_id}/problem/{index}",
)


def _create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())


def _normalize_preformatted_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\xa0", " ").splitlines()]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def _extract_statement_text(statement_node) -> str:
    statement_copy = BeautifulSoup(str(statement_node), "html.parser")
    statement_root = statement_copy.select_one("div.problem-statement")

    for sample_tests_node in statement_root.select("div.sample-tests"):
        sample_tests_node.decompose()

    sections = []
    for child in statement_root.find_all(recursive=False):
        text = _normalize_text(child.get_text(" ", strip=True))
        if text:
            sections.append(text)

    return "\n\n".join(sections)


def _build_problem_urls(problem: dict) -> list[str]:
    contest_id = problem.get("contestId")
    index = problem.get("index")

    if contest_id is None or index is None:
        return []

    return [
        pattern.format(contest_id=contest_id, index=index)
        for pattern in PROBLEM_URL_PATTERNS
    ]


def _fetch_problem_details(problem: dict, session: requests.Session) -> dict:
    details = {"statement": None, "sample_tests": [], "problem_url": None}

    for url in _build_problem_urls(problem):
        response = session.get(url, timeout=30)
        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        statement_node = soup.select_one("div.problem-statement")
        if statement_node is None:
            continue

        details["statement"] = _extract_statement_text(statement_node)
        details["sample_tests"] = [
            {
                "input": _normalize_preformatted_text(
                    sample_test.select_one("div.input pre").get_text("\n", strip=False)
                ),
                "output": _normalize_preformatted_text(
                    sample_test.select_one("div.output pre").get_text("\n", strip=False)
                ),
            }
            for sample_test in statement_node.select("div.sample-test")
            if sample_test.select_one("div.input pre") is not None
            and sample_test.select_one("div.output pre") is not None
        ]
        details["problem_url"] = url
        return details

    return details


def fetch_problems(
    tags: list[str], min_rating: int, max_rating: int, limit: int = 100
) -> list[dict]:
    params = {"tags": ";".join(tags)} if tags else {}
    session = _create_session()
    response = session.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()["result"]["problems"]

    filtered_problems = [
        problem
        for problem in data
        if min_rating <= problem.get("rating", 0) <= max_rating
    ][:limit]

    enriched_problems = []
    for problem in filtered_problems:
        enriched_problem = problem.copy()
        enriched_problem.update(_fetch_problem_details(problem, session))
        enriched_problems.append(enriched_problem)

    return enriched_problems


if __name__ == "__main__":
    problems = fetch_problems(tags=["math"], min_rating=800, max_rating=800, limit=1)
    print(problems[0]["statement"])
    print(problems[0]["sample_tests"])
