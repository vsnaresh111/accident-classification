"""Live road-safety system (GPS -> weather -> Google routes -> risk -> SMS).

Examples
    python run_system.py --destination "51.5033,-0.1196" --origin "51.5155,-0.0922"
    python run_system.py --destination "London Bridge, London" --live-sms
    python run_system.py --destination "51.50,-0.12" --track 5 --interval 30
    python run_system.py --test-alert --live-sms        # labelled TEST SMS, measures delivery
    python run_system.py --tables                        # Tables 11-13 from the log
    python run_system.py --benchmark --destination ...   # Table 12 (concurrent users)

Credentials come from environment variables (or Colab secrets exported to the
environment): GOOGLE_MAPS_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
TWILIO_PHONE_NUMBER, EMERGENCY_PHONE_NUMBER. Without --live-sms no SMS is sent
(dry run); the decision is still computed and logged.
"""
import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from rsr import config as C
from rsr import live
from rsr.routing import google_routes


def parse_point(s):
    if s is None:
        return None
    try:
        a, b = s.split(",")
        return float(a), float(b)
    except ValueError:
        return s  # address string (Routes API geocodes it)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(C.MODEL_DIR / "system_bundle.joblib"))
    ap.add_argument("--destination")
    ap.add_argument("--origin", help="lat,lon; omit to use GPS (browser in Colab, else IP)")
    ap.add_argument("--gps-mode", default="auto", choices=["auto", "browser", "ip", "manual"])
    ap.add_argument("--profile", default="{}", help='JSON, e.g. {"Age_of_Driver": 24, "Vehicle_Type": "9"}')
    ap.add_argument("--live-sms", action="store_true", help="actually send SMS via Twilio")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=C.ROUTE_ALPHA)
    ap.add_argument("--beta", type=float, default=C.ROUTE_BETA)
    ap.add_argument("--track", type=int, default=1, help="number of GPS updates")
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--map", default=str(C.OUTPUT_DIR / "route_map.html"))
    ap.add_argument("--log", default=str(C.LOG_DIR / "system_logs.csv"))
    ap.add_argument("--test-alert", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    a = ap.parse_args()
    C.ensure_dirs()

    origin_pt = parse_point(a.origin)
    if isinstance(origin_pt, str):
        ap.error("--origin must be numeric 'lat,lon' (an address cannot be used as the "
                 "vehicle position); omit --origin to use GPS/IP location")

    if a.tables:
        try:
            L = pd.read_csv(a.log)
        except FileNotFoundError:
            ap.error(f"log file {a.log} not found; run trips first (run_system.py --destination ...)")
        print(f"Runs in log: {len(L)}")
        t11, t13 = live.table11(L), live.table13(L)
        print("\nTABLE 11  Real-time performance (measured)\n", t11.to_string(index=False))
        print("\nTABLE 13  Reliability (measured)\n", t13.to_string(index=False))
        t11.to_csv(C.TABLE_DIR / "table11_realtime_performance.csv", index=False)
        t13.to_csv(C.TABLE_DIR / "table13_reliability.csv", index=False)
        return

    system = live.RiskSystem(a.bundle)
    profile = {**system.meta.get("default_profile", {}), **json.loads(a.profile)}
    print(f"Model: {system.meta.get('model')} | alert score: {system.meta.get('alert_score')} "
          f"| tau = {system.tau}")

    if a.test_alert:
        loc = live.get_location(a.gps_mode, manual=origin_pt)
        if loc["status"] != "OK":
            ap.error(f"no location for the test alert: {loc['status']}")
        body = live.alert_body(loc["lat"] or 0, loc["lon"] or 0, 1.0, "system test", test=True)
        sms = live.send_sms(body, dry_run=not a.live_sms)
        print("TEST ALERT:", sms)
        row = {"run_id": "test-" + datetime.now().strftime("%H%M%S"),
               "timestamp": datetime.now(timezone.utc).isoformat(), "test_alert": True,
               "gps_source": loc["source"], "gps_success": loc["status"] == "OK",
               "gps_seconds": loc["seconds"],
               "sms_attempted": a.live_sms, "sms_status": sms["status"],
               "sms_delivery_seconds": sms["delivery_seconds"]}
        try:
            L = pd.concat([pd.read_csv(a.log), pd.DataFrame([row])], ignore_index=True)
        except FileNotFoundError:
            L = pd.DataFrame([row])
        L.to_csv(a.log, index=False)
        return

    if not a.destination:
        ap.error("--destination is required")
    dest = parse_point(a.destination)
    origin = origin_pt
    kw = dict(origin=origin, gps_mode="manual" if origin else a.gps_mode, profile=profile,
              dry_run=not a.live_sms, alpha=a.alpha, beta=a.beta, tau=a.tau,
              log_path=a.log, map_path=a.map)
    results = []
    for i in range(a.track):
        results.append(live.run_trip(system, dest, **kw))
        if i < a.track - 1:
            time.sleep(a.interval)
    import os
    print(f"\nLog: {a.log}" + (f"; map: {a.map}" if os.path.exists(a.map) else ""))

    if a.benchmark:
        last = next((r for r in reversed(results) if r["routes"]), None)
        if not last:
            print("Benchmark needs at least one successful route request.")
            return
        t12 = live.table12(system, last["routes"], last["conditions"])
        t12.to_csv(C.TABLE_DIR / "table12_scalability.csv", index=False)
        print("\nTABLE 12  Concurrent users (measured)\n", t12.to_string(index=False))


if __name__ == "__main__":
    main()
