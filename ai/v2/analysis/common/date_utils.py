import calendar


def get_last_date(dt):
    # calendar.monthrange()関数で、月の初日の曜日（月曜が0、日曜が6）と、月の日数のタプルが取得できる。
    # replaceは、dayを置き換えて、datetimeクラスのインスタンスを作成
    return dt.replace(day=calendar.monthrange(dt.year, dt.month)[1])
