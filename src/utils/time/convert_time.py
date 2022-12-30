from datetime import datetime


def get_date_from_string(s_date):
    date_patterns = [
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
    ]

    for pattern in date_patterns:
        try:
            return datetime.strptime(s_date, pattern)
        except ValueError:
            print("Date is not in expected format: %s" % (s_date))
