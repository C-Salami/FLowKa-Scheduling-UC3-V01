import os
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Re-generate toy data with order_id like "Order 1", "Order 2", ...
random.seed(42)
np.random.seed(42)

WHEEL_TYPES = ["Urban-200", "Offroad-250", "Racing-180", "HeavyDuty-300", "Eco-160"]
MACHINES = ["Lathe A", "Lathe B", "Grinder A", "Grinder B", "Polisher A", "Polisher B"]

N_ORDERS = 30
OPS_PER_ORDER = 3  # e.g., cut -> grind -> polish

start_base = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
data_sched = []
data_orders = []

for i in range(1, N_ORDERS + 1):
    order_id = f"Order {i}"
    wheel = random.choice(WHEEL_TYPES)
    due = start_base + timedelta(days=random.randint(3, 14))

    # simple linear operations per order
    t = start_base + timedelta(hours=random.randint(0, 72))
    for seq in range(1, OPS_PER_ORDER + 1):
        op = ["Cutting", "Grinding", "Polishing"][seq - 1]
        dur_h = random.choice([2, 3, 4])
        machine = random.choice(MACHINES if seq != 2 else [m for m in MACHINES if "Grinder" in m])

        start = t
        end = start + timedelta(hours=dur_h)
        t = end + timedelta(hours=1)  # gap to next op

        data_sched.append({
            "order_id": order_id,
            "operation": op,
            "sequence": seq,
            "machine": machine,
            "start": start,
            "end": end,
            "due_date": due,
            "wheel_type": wheel,
        })

    data_orders.append({
        "order_id": order_id,
        "wheel_type": wheel,
        "due_date": due.date(),
        "priority": random.choice(["Low", "Medium", "High"]),
    })

os.makedirs("data", exist_ok=True)
pd.DataFrame(data_orders).to_csv("data/scooter_orders.csv", index=False)
pd.DataFrame(data_sched).to_csv("data/scooter_schedule.csv", index=False)
print("Data regenerated: data/scooter_orders.csv, data/scooter_schedule.csv")
