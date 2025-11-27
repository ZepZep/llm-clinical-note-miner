import regex
from typing import List, Dict, Optional, Any

def find_matches(
    text: str, 
    query: str, 
    fuzzy_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, int]]:
    """
    Finds matches of query in text using fuzzy regex.
    
    fuzzy_config can contain:
    - errors: int or str (default 2 for queries > 10 chars, else 0)
    """
    if not query:
        return []
        
    fuzzy_config = fuzzy_config or {}
    
    # Default fuzzy logic
    if 'errors' in fuzzy_config:
        errors = fuzzy_config['errors']
    else:
        # Simple heuristic: allow errors for longer strings
        errors = 2 if len(query) > 10 else 0
        
    # Escape the query to treat it as literal text, then apply fuzzy settings
    # regex.escape escapes special characters. 
    # We wrap it in (...) and append {e<=N} for fuzzy matching.
    escaped_query = regex.escape(query)
    
    if errors:
        pattern = f"({escaped_query}){{e<={errors}}}"
    else:
        pattern = escaped_query
        
    # best_match=True tries to find the best match
    # However, regex.findall doesn't give positions directly easily with overlaps/best match logic combined simply.
    # regex.finditer is better.
    
    matches = []
    # POSIX matching might be better for "best" match but it's slower. 
    # We'll use standard finditer.
    
    for match in regex.finditer(pattern, text, flags=regex.IGNORECASE):
        matches.append({
            "start": match.start(),
            "end": match.end(),
            "match_text": match.group()
        })
        
    # If we want strictly the "best" match (fewest errors), we might need to inspect the fuzzy counts.
    # regex module provides fuzzy_counts in the match object if we use the right syntax, 
    # but standard {e<=N} just matches. 
    # For now, returning all matches is safer, or we could filter.
    # The user asked to "find the best match".
    # If there are multiple matches, usually the first one or the one with fewest errors is preferred.
    # Let's try to sort by match quality if possible, but regex doesn't easily expose error count directly in the match object without complex groups.
    # A simple approximation is to pick the one with length closest to query length (since errors usually mean insertions/deletions/substitutions).
    
    if matches:
        # Sort by how close the match length is to the query length
        matches.sort(key=lambda m: abs(len(m['match_text']) - len(query)))
        
    return matches
