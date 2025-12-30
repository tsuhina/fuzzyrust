"""
Real-world test data for FuzzyRust testing.

Contains realistic examples of:
- Person names with typos and variations
- Addresses with formatting differences
- Product names with abbreviations
- Company names with variations
"""

# Person names - common names with variations and typos
PERSON_NAMES = [
    "John Smith",
    "Jon Smith",
    "John Smyth",
    "Jonathan Smith",
    "J. Smith",
    "Jane Doe",
    "Jan Doe",
    "Janet Doe",
    "Michael Johnson",
    "Mike Johnson",
    "Micheal Johnson",  # Common typo
    "Robert Williams",
    "Bob Williams",
    "Rob Williams",
    "Roberto Williams",
    "William Davis",
    "Bill Davis",
    "Will Davis",
    "Wm. Davis",
    "James Brown",
    "Jim Brown",
    "Jimmy Brown",
    "Elizabeth Taylor",
    "Liz Taylor",
    "Beth Taylor",
    "Elisabeth Taylor",  # Variant spelling
    "Christopher Wilson",
    "Chris Wilson",
    "Christoper Wilson",  # Typo
    "Kristopher Wilson",  # Variant
]

# Pairs of names that should match with high similarity
PERSON_NAME_PAIRS = [
    ("John Smith", "Jon Smith", 0.85),  # (name1, name2, expected_min_similarity)
    ("John Smith", "John Smyth", 0.85),
    ("Michael Johnson", "Micheal Johnson", 0.9),
    ("Robert Williams", "Bob Williams", 0.6),
    ("Elizabeth Taylor", "Liz Taylor", 0.6),
    ("Christopher Wilson", "Chris Wilson", 0.6),
    ("James Brown", "Jim Brown", 0.5),
]

# Addresses with variations
ADDRESSES = [
    "123 Main Street, New York, NY 10001",
    "123 Main St., New York, NY 10001",
    "123 Main St, New York, New York 10001",
    "456 Oak Avenue, Los Angeles, CA 90001",
    "456 Oak Ave., Los Angeles, CA 90001",
    "456 Oak Ave, L.A., California 90001",
    "789 Elm Boulevard, Chicago, IL 60601",
    "789 Elm Blvd., Chicago, IL 60601",
    "789 Elm Blvd, Chicago, Illinois 60601",
    "1000 Pine Road, Houston, TX 77001",
    "1000 Pine Rd., Houston, TX 77001",
    "1000 Pine Rd, Houston, Texas 77001",
    "2500 Maple Lane, Phoenix, AZ 85001",
    "2500 Maple Ln., Phoenix, AZ 85001",
    "2500 Maple Ln, Phoenix, Arizona 85001",
]

# Address pairs that should match
ADDRESS_PAIRS = [
    ("123 Main Street, New York, NY 10001", "123 Main St., New York, NY 10001", 0.85),
    ("456 Oak Avenue, Los Angeles, CA 90001", "456 Oak Ave., Los Angeles, CA 90001", 0.85),
    ("789 Elm Boulevard, Chicago, IL 60601", "789 Elm Blvd., Chicago, IL 60601", 0.85),
]

# Product names with variations
PRODUCT_NAMES = [
    "Apple iPhone 14 Pro Max 256GB",
    "Apple iPhone14 Pro Max 256 GB",
    "iPhone 14 Pro Max 256GB Apple",
    "Samsung Galaxy S23 Ultra 512GB",
    "Samsung Galaxy S23Ultra 512 GB",
    "Galaxy S23 Ultra Samsung 512GB",
    "Sony WH-1000XM5 Headphones",
    "Sony WH1000XM5 Headphones",
    "WH-1000XM5 Sony Headphones",
    "Nike Air Max 90 Running Shoes",
    "Nike AirMax 90 Running Shoes",
    "Air Max 90 Nike Running Shoes",
    "Dell XPS 15 Laptop 16GB RAM",
    "Dell XPS15 Laptop 16 GB RAM",
    "XPS 15 Dell Laptop 16GB RAM",
]

# Product name pairs that should match
PRODUCT_NAME_PAIRS = [
    ("Apple iPhone 14 Pro Max 256GB", "Apple iPhone14 Pro Max 256 GB", 0.9),
    ("Samsung Galaxy S23 Ultra 512GB", "Samsung Galaxy S23Ultra 512 GB", 0.9),
    ("Sony WH-1000XM5 Headphones", "Sony WH1000XM5 Headphones", 0.9),
]

# Company names with variations
COMPANY_NAMES = [
    "Apple Inc.",
    "Apple Incorporated",
    "Apple Inc",
    "Apple",
    "Microsoft Corporation",
    "Microsoft Corp.",
    "Microsoft Corp",
    "Microsoft",
    "Amazon.com Inc.",
    "Amazon.com, Inc.",
    "Amazon Inc.",
    "Amazon",
    "Alphabet Inc.",
    "Alphabet Incorporated",
    "Google LLC",
    "Google",
    "Meta Platforms Inc.",
    "Meta Platforms, Inc.",
    "Facebook Inc.",  # Old name
    "Meta",
    "Tesla Inc.",
    "Tesla, Inc.",
    "Tesla Motors",
    "Tesla",
    "Johnson & Johnson",
    "Johnson and Johnson",
    "J&J",
    "Procter & Gamble",
    "Procter and Gamble",
    "P&G",
]

# Company name pairs that should match
COMPANY_NAME_PAIRS = [
    ("Apple Inc.", "Apple Incorporated", 0.7),
    ("Microsoft Corporation", "Microsoft Corp.", 0.8),
    ("Amazon.com Inc.", "Amazon Inc.", 0.8),
    ("Johnson & Johnson", "Johnson and Johnson", 0.9),
    ("Procter & Gamble", "Procter and Gamble", 0.9),
]

# Email addresses with typos
EMAIL_ADDRESSES = [
    "john.smith@gmail.com",
    "john.smtih@gmail.com",  # Typo
    "johnsmith@gmail.com",
    "j.smith@gmail.com",
    "jane.doe@yahoo.com",
    "jane_doe@yahoo.com",
    "janedoe@yahoo.com",
    "michael.johnson@outlook.com",
    "m.johnson@outlook.com",
    "mjohnson@outlook.com",
]

# Phone numbers in various formats
PHONE_NUMBERS = [
    "555-123-4567",
    "(555) 123-4567",
    "5551234567",
    "555.123.4567",
    "+1 555-123-4567",
    "+1 (555) 123-4567",
    "1-555-123-4567",
]

# Misspelled words with corrections
MISSPELLINGS = [
    ("accomodate", "accommodate"),
    ("occurence", "occurrence"),
    ("recieve", "receive"),
    ("seperate", "separate"),
    ("definately", "definitely"),
    ("occured", "occurred"),
    ("refered", "referred"),
    ("untill", "until"),
    ("wierd", "weird"),
    ("thier", "their"),
    ("beleive", "believe"),
    ("concensus", "consensus"),
    ("enterpreneur", "entrepreneur"),
    ("goverment", "government"),
    ("independant", "independent"),
]

# Medical terms with variations
MEDICAL_TERMS = [
    ("acetaminophen", "paracetamol"),  # Same drug, different names
    ("ibuprofen", "ibuprofin"),  # Misspelling
    ("amoxicillin", "amoxicilin"),  # Misspelling
    ("penicillin", "penicillin"),
    ("aspirin", "acetylsalicylic acid"),  # Common vs scientific
]

# Names in different scripts (for Unicode testing)
INTERNATIONAL_NAMES = [
    "田中太郎",  # Japanese
    "김철수",  # Korean
    "Müller",  # German
    "François",  # French
    "José",  # Spanish
    "Владимир",  # Russian
    "محمد",  # Arabic
    "Αλέξανδρος",  # Greek
    "Süreyya",  # Turkish
    "Søren",  # Danish
]
