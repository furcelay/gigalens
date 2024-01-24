def merge_dicts(d1, d2):
    """
    Merge two nested dictionaries into a new one without modifying the originals.
    Raises ValueError in case of conflicts.
    """
    merged = {}
    for key in d1.keys() | d2.keys():  # Union of keys from both dictionaries
        if key in d1 and key in d2:  # Key is in both dictionaries
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):  # Both values are dictionaries
                merged[key] = merge_dicts(d1[key], d2[key])  # Recursively merge them
            else:
                raise ValueError(f"Conflict: {key} parameter is in both dictionaries, cannot safely merge them\nleft:{d1[key]}right:{d2[key]}")
        elif key in d1:  # Key is only in d1
            merged[key] = d1[key]
        else:  # Key is only in d2
            merged[key] = d2[key]
    return merged
