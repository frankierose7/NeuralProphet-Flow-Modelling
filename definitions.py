locations = [
    'Bacup', 
    'Blackstone-Edge-No-2',
    'Bury', 
    'Cowm',
    #'Heaton-Park', # just outside catchment
    'Holden-Wood', 
    'Kitcliffe',
    'Loveclough',
    'Ringley',
    'Sweetloves'
]

quality_codes = {
    'Good': 4,
    'Estimated': 3,
    'Unchecked': 2,
    'Suspect': 1,
    'Missing': 0
}

quality_codes_inv = {value:key for key, value in quality_codes.items()}

completeness_codes = {
    'Complete': 1,
    'Incomplete': 0
}


