#!/usr/bin/env python3
"""Write data/visitors.json from the GoatCounter API for the HUD's visitors cell.

Runs in CI before `hugo`. Needs GOATCOUNTER_TOKEN (an API key with "read
statistics" permission) and GOATCOUNTER_CODE (the site code, e.g. "akurniawan").
Without a token, or if the API fails, it writes nothing and the cell is hidden;
it never fails the build.
"""
import datetime as dt
import json
import os
import sys
import urllib.parse
import urllib.request

WINDOW_H = 48
SINCE = "2018-01-01T00:00:00Z"  # before the first post; GoatCounter clamps to the site's start


def get_total(code, token, start, end):
    q = urllib.parse.urlencode({"start": start, "end": end})
    req = urllib.request.Request(
        f"https://{code}.goatcounter.com/api/v0/stats/total?{q}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def main():
    token = os.environ.get("GOATCOUNTER_TOKEN", "").strip()
    code = os.environ.get("GOATCOUNTER_CODE", "").strip()
    if not token or not code:
        print("visitors: no GOATCOUNTER_TOKEN/GOATCOUNTER_CODE, skipping")
        return

    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    end = (now + dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent_start = (now - dt.timedelta(hours=WINDOW_H + 24)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        all_time = get_total(code, token, SINCE, end)
        recent = get_total(code, token, recent_start, end)
    except Exception as e:  # network, auth, rate limit: keep the build going
        print(f"visitors: API request failed ({e}), skipping")
        return

    # Flatten day/hourly buckets into one series, oldest first, ending at the current hour.
    series = []
    for day in sorted(recent.get("stats") or [], key=lambda d: d["day"]):
        for h, n in enumerate(day.get("hourly") or [0] * 24):
            t = dt.datetime.fromisoformat(day["day"]).replace(hour=h, tzinfo=dt.timezone.utc)
            if t <= now:
                series.append(int(n))
    hourly = series[-WINDOW_H:]

    out = {
        "total": int(all_time.get("total") or 0),
        "window_h": len(hourly),
        "window_total": sum(hourly),
        "hourly": hourly,
        "updated": now.strftime("%Y-%m-%dT%H:%MZ"),
    }
    os.makedirs("data", exist_ok=True)
    with open("data/visitors.json", "w") as f:
        json.dump(out, f)
    print(f"visitors: total={out['total']} last {out['window_h']}h={out['window_total']}")


if __name__ == "__main__":
    main()
    sys.exit(0)
