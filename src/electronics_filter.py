import pandas as pd

INCLUDE_KEYWORDS = [
    r"\btv\b", r"\bled\b", r"smart tv", r"television", r"\bmobile\b", r"\bgb ram\b", r"\bram\b",
    r"power bank", r"\bmah\b", r"charger", r"\bcable\b", r"earphone", r"headphone", r"earbud",
    r"neckband", r"bluetooth", r"speaker", r"soundbar", r"smartwatch", r"smart watch",
    r"laptop", r"\bmouse\b", r"keyboard", r"router", r"memory card", r"microsdxc", r"pen drive",
    r"hard disk", r"\bssd\b", r"webcam", r"camera", r"printer", r"monitor", r"tablet",
    r"trimmer", r"shaver", r"epilator", r"hair dryer", r"straightener", r"kettle",
    r"\bcooler\b", r"\bfan\b", r"heater", r"geyser", r"\biron\b", r"mixer", r"grinder",
    r"induction", r"microwave", r"\boven\b", r"refrigerator", r"washing machine",
    r"air conditioner", r"\bac\b", r"sewing machine", r"air purifier", r"vacuum cleaner",
    r"water purifier", r"extension board", r"\bled bulb\b", r"\bwatt\b",
    r"dishwasher", r"streaming device", r"projector", r"\bhdmi\b",
    r"cctv", r"stabilizer", r"home theatre", r"chromecast",
]

EXCLUDE_KEYWORDS = [
    r"\bstand\b", r"\bcabinets?\b", r"entertainment unit", r"dinner set", r"opalware",
    r"holder", r"sleeve", r"\bcover\b", r"\bcase\b", r"\bpouch\b", r"\bbag\b",
]

_INCLUDE = "|".join(INCLUDE_KEYWORDS)
_EXCLUDE = "|".join(EXCLUDE_KEYWORDS)


def is_electronics(product_names):
    """Return a True/False Series: is each product name an electronics item?"""
    has_include = product_names.str.contains(_INCLUDE, case=False, regex=True)
    has_exclude = product_names.str.contains(_EXCLUDE, case=False, regex=True)
    return has_include & ~has_exclude