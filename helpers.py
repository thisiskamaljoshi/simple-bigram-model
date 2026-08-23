import string

# All standard ASCII punctuation
PUNCTUATION = set(string.punctuation)

# Attach directly to the previous token
NO_SPACE_BEFORE = {
    ".", ",", "!", "?", ":", ";",
    ")", "]", "}"
}

# Attach directly to the next token
NO_SPACE_AFTER = {
    "(", "[", "{"
}

# Attach to both previous and next tokens
NO_SPACE_AROUND = {
    "'"
}