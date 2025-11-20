# /// script
# dependencies = [
#   "matplotlib",
# ]
# ///
import sys
import json
import matplotlib.pyplot as plt
from collections import Counter

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <json_file>")
    sys.exit(1)

filename = sys.argv[1]
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

words = data.get('words', [])
words = [w for w in words if w.get('type', 'word') == 'word']
if not words or len(words) < 2:
    print("No or too few words found.")
    sys.exit(1)

def to_ms(val):
    """Convert S.ms to ms if float or string with '.', else return as int ms."""
    if isinstance(val, (float, int)):
        return int(val * 1000) if not isinstance(val, int) else val
    if isinstance(val, str) and '.' in val:
        return int(float(val) * 1000)
    return int(val)

# Compute pause lengths
pauses = []
for prev, curr in zip(words[:-1], words[1:]):
    prev_end = to_ms(prev['end'])
    curr_start = to_ms(curr['start'])
    pause = curr_start - prev_end
    # Only consider positive pauses
    if pause > 0:
        # Round to nearest 50 ms
        if pause > 1000:
            pause_rounded = 1000
        else:
            pause_rounded = round(pause / 50) * 50
        pauses.append(pause_rounded)

if not pauses:
    print("No positive pauses found.")
    sys.exit(1)

# Count occurrences
pause_counts = Counter(pauses)
# Sort by pause length
sorted_pauses = sorted(pause_counts.items())

# Prepare for plotting
x = [p for p, _ in sorted_pauses]
y = [c for _, c in sorted_pauses]

plt.figure(figsize=(10, 6))
plt.bar(x, y, width=40)
plt.xlabel('Pause length (ms)')
plt.ylabel('Occurrences')
plt.title('Pause Lengths (rounded to 50ms, >1000ms bucketed)')
plt.xticks(x, rotation=90)
plt.tight_layout()
plt.show() 