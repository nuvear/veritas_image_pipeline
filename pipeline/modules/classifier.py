"""
classifier.py — Stage 1: Food type classification and query/slot description generation.

Given a food record dict, returns:
  - food_type: cooked_dish | raw_ingredient | packaged_product | beverage
  - queries: list of search query strings per slot
  - slot_descriptions: dict of hero/macro/in_the_wild descriptions for vision scoring
"""

import re

# ---------------------------------------------------------------------------
# Food type classification
# ---------------------------------------------------------------------------

BEVERAGE_KEYWORDS = [
    "tea", "coffee", "juice", "drink", "milk", "beer", "wine", "spirit",
    "water", "soda", "smoothie", "shake", "latte", "kopi", "teh", "milo",
    "kombucha", "kefir", "lassi", "toddy", "whisky", "vodka", "rum",
    "brandy", "gin", "sake", "soju", "cider", "ale", "stout", "lager",
    "beverage", "syrup", "cordial", "barley", "chrysanthemum", "bandung",
    "ribena", "yakult", "pocari", "100plus", "isotonic", "energy drink",
    "bubble tea", "boba", "pearl milk", "sugarcane", "coconut water",
    "longan drink", "grass jelly drink",
]

PACKAGED_KEYWORDS = [
    "packaged", "instant", "canned", "bottled", "frozen", "processed",
    "snack", "chip", "biscuit", "cracker", "cereal", "bread", "loaf",
    "sauce", "paste", "condiment", "spread", "jam", "butter", "margarine",
    "oil", "vinegar", "soy sauce", "oyster sauce", "fish sauce",
    "mayonnaise", "ketchup", "mustard", "pickle", "preserved",
    "dried", "powder", "flour", "sugar", "salt",
]

RAW_INGREDIENT_KEYWORDS = [
    "raw", "fresh", "uncooked", "ingredient", "vegetable", "fruit",
    "meat", "fish", "seafood", "egg", "grain", "legume", "bean",
    "lentil", "nut", "seed", "herb", "spice", "mushroom", "tofu",
    "tempeh", "noodle", "rice", "pasta", "leaf", "root", "tuber",
]


def classify_food_type(record: dict) -> str:
    """Classify food into one of four types based on record fields."""
    # Explicit field takes priority
    if record.get("food_type"):
        ft = record["food_type"].lower().strip()
        if ft in ("cooked_dish", "raw_ingredient", "packaged_product", "beverage"):
            return ft

    # Use food_group if available
    food_group = (record.get("food_group") or "").lower()
    if any(k in food_group for k in ["beverage", "drink", "juice", "tea", "coffee", "alcohol"]):
        return "beverage"
    if any(k in food_group for k in ["packaged", "processed", "snack", "condiment"]):
        return "packaged_product"
    if any(k in food_group for k in ["raw", "ingredient", "produce", "meat", "seafood", "grain"]):
        return "raw_ingredient"

    # Fall back to name matching
    name = (record.get("food_name") or "").lower()
    if any(k in name for k in BEVERAGE_KEYWORDS):
        return "beverage"
    if any(k in name for k in PACKAGED_KEYWORDS):
        return "packaged_product"
    if any(k in name for k in RAW_INGREDIENT_KEYWORDS):
        return "raw_ingredient"

    return "cooked_dish"


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def _get_alt_names(record: dict) -> list:
    """Extract alternative names from aliases and local_names."""
    alts = []
    aliases = record.get("aliases") or []
    if isinstance(aliases, list):
        alts.extend([a for a in aliases if isinstance(a, str) and len(a) < 60])
    local = record.get("local_names") or {}
    if isinstance(local, dict):
        for v in local.values():
            if isinstance(v, str) and len(v) < 60 and v not in alts:
                alts.append(v)
    return alts[:4]


def build_queries(record: dict, food_type: str) -> dict:
    """
    Build search queries for each slot.
    Returns dict: {hero: [q1,q2,...], macro: [...], in_the_wild: [...]}
    """
    name = record.get("food_name", "food")
    cuisine = record.get("cuisine") or record.get("region_of_origin") or "Asian"
    alts = _get_alt_names(record)
    alt_str = alts[0] if alts else ""

    if food_type == "beverage":
        hero = [
            f"{name} drink photo",
            f"{name} glass cup photo",
            f"{alt_str} beverage" if alt_str else f"{cuisine} {name} drink",
        ]
        macro = [
            f"{name} close up texture",
            f"{name} detail shot",
            f"{name} macro photography",
        ]
        in_the_wild = [
            f"{name} cafe restaurant",
            f"{name} hawker stall served",
            f"{cuisine} {name} street food",
        ]

    elif food_type == "raw_ingredient":
        hero = [
            f"{name} ingredient photo",
            f"fresh {name} food photography",
            f"{alt_str} ingredient" if alt_str else f"{name} raw fresh",
        ]
        macro = [
            f"{name} close up texture detail",
            f"{name} macro food photo",
            f"fresh {name} detail",
        ]
        in_the_wild = [
            f"{name} market grocery",
            f"{name} cooking preparation",
            f"{name} kitchen home",
        ]

    elif food_type == "packaged_product":
        hero = [
            f"{name} product photo",
            f"{name} package front",
            f"{alt_str} product" if alt_str else f"{name} packaged food",
        ]
        macro = [
            f"{name} close up detail",
            f"{name} texture macro",
            f"{name} food detail shot",
        ]
        in_the_wild = [
            f"{name} supermarket shelf",
            f"{name} pantry kitchen",
            f"{name} in use cooking",
        ]

    else:  # cooked_dish
        hero = [
            f"{name} food photography",
            f"{name} dish plated",
            f"{alt_str} food photo" if alt_str else f"{cuisine} {name} dish",
        ]
        macro = [
            f"{name} close up texture",
            f"{name} macro food photo",
            f"{name} detail shot food",
        ]
        in_the_wild = [
            f"{name} hawker stall restaurant",
            f"{name} street food served",
            f"{cuisine} {name} food stall",
        ]

    # Clean up empty strings
    def clean(lst):
        return [q.strip() for q in lst if q.strip() and len(q.strip()) > 3]

    return {
        "hero": clean(hero),
        "macro": clean(macro),
        "in_the_wild": clean(in_the_wild),
    }


# ---------------------------------------------------------------------------
# Slot descriptions for vision scoring
# ---------------------------------------------------------------------------

def build_slot_descriptions(record: dict, food_type: str) -> dict:
    """
    Build natural-language descriptions for each slot used in vision scoring prompts.
    """
    name = record.get("food_name", "food")

    if food_type == "beverage":
        return {
            "hero": f"A clear, well-lit photo of {name} in a glass, cup, or bottle on a clean background. Single serving, no clutter.",
            "macro": f"A close-up macro shot of {name} showing liquid texture, colour, bubbles, or garnish detail.",
            "in_the_wild": f"A photo of {name} being served or consumed at a cafe, hawker stall, restaurant, or home setting.",
        }
    elif food_type == "raw_ingredient":
        return {
            "hero": f"A clean, well-lit photo of {name} as a raw ingredient on a neutral background. Shows the whole item clearly.",
            "macro": f"A close-up macro shot of {name} showing surface texture, colour, and detail.",
            "in_the_wild": f"A photo of {name} in a market, grocery store, kitchen, or being prepared for cooking.",
        }
    elif food_type == "packaged_product":
        return {
            "hero": f"A clear photo of {name} showing the product packaging, label, or front of pack on a clean background.",
            "macro": f"A close-up of {name} showing texture, colour, or detail of the food itself (not just the packaging).",
            "in_the_wild": f"A photo of {name} on a supermarket shelf, in a pantry, or being used in a kitchen context.",
        }
    else:  # cooked_dish
        return {
            "hero": f"A well-lit food photo of {name} plated as a single serving, overhead or 45-degree angle, clean background or minimal props.",
            "macro": f"A close-up macro shot of {name} showing the texture, colour, and detail of the food.",
            "in_the_wild": f"A photo of {name} at a hawker centre, restaurant, street food stall, or home dining table in its natural serving context.",
        }
