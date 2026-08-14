import requests

# Ask Zenodo for the record's metadata. No data files are downloaded here.
r = requests.get("https://zenodo.org/api/records/14017092")
r.raise_for_status()
rec = r.json()

# Print each file's size in gigabytes next to its name, smallest first.
files = sorted(rec["files"], key=lambda f: f["size"])
for f in files:
    print(f"{f['size']/1e9:8.2f} GB  {f['key']}")

# Print the total so you know what a full download would cost.
total = sum(f["size"] for f in files)
print(f"\n{len(files)} files, {total/1e9:.1f} GB total")
