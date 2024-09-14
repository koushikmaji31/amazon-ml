import re

# List of possible units
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

# Flattened set of all units from the entity_unit_map
all_units = {unit for units in entity_unit_map.values() for unit in units}

# Mapping for unit abbreviations to standard units
unit_mapping = {
    'g': 'gram',
    'kg': 'kilogram',
    'mg': 'milligram',
    'mcg': 'microgram',
    'oz': 'ounce',
    'lb': 'pound',
    't': 'ton',
    'ml': 'millilitre',
    'l': 'litre',
    'cl': 'centilitre',
    'dl': 'decilitre',
    'fl oz': 'fluid ounce',
    'ft': 'foot',
    'in': 'inch',
    'cm': 'centimetre',
    'mm': 'millimetre',
    'yd': 'yard',
    'v': 'volt',
    'mv': 'millivolt',
    'kv': 'kilovolt',
    'w': 'watt',
    'kw': 'kilowatt',
    'gal': 'gallon',
    'pt': 'pint',
    'qt': 'quart',
    'cu ft': 'cubic foot',
    'cu in': 'cubic inch',
    'imp gal': 'imperial gallon'
}

def normalize_unit(unit):
    # Normalize the input unit to match the standard list
    unit = unit.lower().strip()

    # Handle plural units by removing 's' and checking if the singular form exists
    if unit.endswith('s') and unit[:-1] in all_units:
        unit = unit[:-1]

    if unit in unit_mapping:
        return unit_mapping[unit]

    return unit

def extract_value_unit(input_string):
    # Regular expression to extract numeric value and unit
    pattern = r"(\d+\.?\d*)\s*([a-zA-Z]+(?:\s*[a-zA-Z]*)?)"
    matches = re.findall(pattern, input_string)

    # Return the first valid match
    for match in matches:
        value = match[0]  # Extract numeric value
        unit = normalize_unit(match[1])  # Extract and normalize unit

        if unit in all_units:  # Check if unit is in the list of allowed units
            return f"{value} {unit}"
    
    return None  # If no valid match is found

# Test cases with complex input
test_strings = [
    '11 oz',
    "weight is 10 gram",
    "10g",
    "11 ounces",  # Should output "11 ounce"
    "11oz",
    "15 kg",
    "4 fluid ounce",
    "7ft",
    "3 cubic inch",
    'The wattage mentioned in the text is "30W," which stands for 30 watts. The unit provided is watt.',
    'The device weighs 12g and the height is 15cm.',
    'The fluid ounce is marked as 20fl oz in the instructions.',
    'He mentioned the voltage as being around 5kv or 5000 volts.'
]

# Running the test cases
for test in test_strings:
    result = extract_value_unit(test)
    if result:
        print(f"{result}\n")
    else:
        print("\n")
