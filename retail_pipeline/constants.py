"""Shared dimensions for the generated retail star schema.

The generator and the write-up both use these labels. Region, segment, and
loyalty are assigned independently so a later group-by is not just echoing
one modulo of customer_id.
"""

N_CUSTOMERS = 500_000
N_PRODUCTS = 20_000
N_STORES = 400
DATE_ORIGIN = "2022-01-01"
DATE_SPAN_DAYS = 1095  # 2022-01-01 through 2024-12-30

REGIONS = [
    "Northeast",
    "Southeast",
    "Midwest",
    "Southwest",
    "West",
    "Pacific",
    "Mountain",
    "Mid-Atlantic",
]
CHANNELS = ["web", "store", "app", "marketplace"]
STORE_CHANNEL_INDEX = 1
MARKETPLACE_CHANNEL_INDEX = 3
SEGMENTS = ["consumer", "premium", "business", "student"]
LOYALTY = ["bronze", "silver", "gold", "platinum"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
CATEGORIES = [
    "Electronics",
    "Home",
    "Apparel",
    "Grocery",
    "Sports",
    "Beauty",
    "Toys",
    "Automotive",
]
SUBCATEGORIES = ["Basics", "Premium", "Seasonal", "Essentials"]
BRANDS = [
    "Northwind",
    "Contoso",
    "Fabrikam",
    "Adventure",
    "Wide World",
    "Litware",
    "Proseware",
    "ADatum",
]
DEVICES = ["pos", "ios", "android", "web", "kiosk"]
PAYMENTS = ["card", "cash", "wallet", "bnpl"]

# Columns kept in the fact file only so the physical plan can show that the
# scan does not read them. receipt_token and payment_ref are high-entropy and
# are what push the Parquet dataset over 1 GB.
UNUSED_FACT_COLUMNS = [
    "store_id",
    "payment_method",
    "device_type",
    "promo_code",
    "receipt_token",
    "payment_ref",
]

PLANTED_SIGNALS = [
    "In-store purchases are assigned the customer's home region. Web, app, and marketplace purchases use an independent fulfillment region, so cross-region sales concentrate outside the store channel.",
    "A row is more likely to be cancelled when the channel is marketplace, the discount is 20% or 60%, or the quantity is 4 or 5.",
    "Roughly 2% of fact rows carry a 60% discount. The revenue pipeline drops those rows with a native filter.",
    "Sale dates cover 2022-01 through 2024-12. Revenue queries keep sale_month >= 2023-01, so the 2022 partition directories should not be read.",
]
