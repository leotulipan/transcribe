#!/usr/bin/env bash
# Usage: source load_env.sh <path-to-.env> [optional-key]

ENV_FILE="$1"
TARGET_KEY="$2"

if [ -z "$ENV_FILE" ] || [ ! -f "$ENV_FILE" ]; then
    echo "Usage: source load_env.sh <path-to-.env> [optional-key]"
    return 1 2>/dev/null || exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
    if [[ -z "$line" ]] || [[ "$line" == \#* ]]; then continue; fi
    
    # Strip trailing CR (Windows CRLF) before any field splitting
    line="${line%$'\r'}"
    key="${line%%=*}"
    value="${line#*=}"

    # Strip quotes from value, trailing whitespace from value, whitespace from key
    value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
    # shellcheck disable=SC2001  # trailing whitespace removal needs sed
    value="$(echo -n "$value" | sed 's/[[:space:]]*$//')"
    key=$(echo "$key" | xargs)
    
    if [ -z "$TARGET_KEY" ] || [ "$key" == "$TARGET_KEY" ]; then
        export "$key"="$value"
        [ -n "$TARGET_KEY" ] && { echo "Loaded $key"; return 0 2>/dev/null || exit 0; }
    fi
done < "$ENV_FILE"