import os
import re

data_folder = "data"

# Pattern 1: <catchphrase "id=c0" ...>
pattern1 = re.compile(r'<catchphrase\s+["\']id=([^"\'>\s]+)["\'](?P<rest>[^>]*>)', re.IGNORECASE)
# Pattern 2: <catchphrase id=c0 ...>
pattern2 = re.compile(r'<catchphrase\s+id=([^"\'>\s]+)(?P<rest>[^>]*>)', re.IGNORECASE)

fixed_any = False

for filename in sorted(os.listdir(data_folder)):
    if not filename.lower().endswith(".xml"):
        continue

    file_path = os.path.join(data_folder, filename)

    # --- Safe open (try utf-8, else fallback to latin-1) ---
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()

    # Apply regex fixes
    fixed = pattern1.sub(r'<catchphrase id="\1"\g<rest>', content)
    fixed = pattern2.sub(r'<catchphrase id="\1"\g<rest>', fixed)

    # Save back only if modified
    if fixed != content:
        with open(file_path, "w", encoding="utf-8") as f:  # always normalize to utf-8
            f.write(fixed)
        print(f"Fixed: {filename}")
        fixed_any = True
    else:
        print(f"No issues: {filename}")

if not fixed_any:
    print("No files needed fixing.")
else:
    print("✅ Finished fixing malformed <catchphrase> tags (files now UTF-8).")
