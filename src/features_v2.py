"""Final (v2) ozellik seti."""

TARGET = "Late_delivery_risk"

FEATURES = [
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Category Name",
    "Order Region",
    "Customer Segment",
    "Department Name",
    "Market",
    "order_month",
    "order_weekday",
    "is_weekend",
    "scheduled_x_first",
    "scheduled_x_second",
]

LEAKY = [
    "Days for shipping (real)",
    "Delivery Status",
]

FINAL_MODEL = "BaggingClassifier (tuned)"
