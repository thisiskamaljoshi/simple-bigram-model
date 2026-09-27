import string
 
# Standard ASCII punctuation plus common typographic Unicode punctuation
PUNCTUATION = set(string.punctuation) | {"“", "”", "‘", "’", "—", "–", "…"}

# Attach directly to the previous token (no preceding space)
NO_SPACE_BEFORE = {
    ".", ",", "!", "?", ":", ";",
    ")", "]", "}", "”", "’"
}

# Attach directly to the next token (no trailing space)
NO_SPACE_AFTER = {
    "(", "[", "{", "“", "‘"
}

# Attach to both previous and next tokens
NO_SPACE_AROUND = {
    "'"
}