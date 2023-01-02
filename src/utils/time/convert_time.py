from datetime import datetime


def get_date_from_string(s_date):
    date_patterns = [
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
    ]

    try:
        for pattern in date_patterns:
            return datetime.strptime(s_date, pattern)
    except ValueError:
        print(f"Date {s_date} not in expected format: %s")
