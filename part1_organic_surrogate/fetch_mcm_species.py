"""Download the MCM v3.3.1 species list (all 143 primary VOCs, "species TSV" export) to mcm_v331_species.tsv.

The notebooks that use the MCM pool expect this file in the working directory. Please cite the MCM website and
the MCM protocol papers (Jenkin et al., 1997; Saunders et al., 2003) when using it.
"""
import re
import urllib.parse
import urllib.request

BASE = "https://mcm.york.ac.uk/MCM"
html = urllib.request.urlopen(f"{BASE}/browse", timeout=60).read().decode()
vocs = sorted(set(re.findall(r'href="?/MCM/species/([A-Za-z0-9_]+)', html)))
print(f"{len(vocs)} primary VOCs found (expected 143)")
query = urllib.parse.urlencode([("selected[]", v) for v in vocs] + [("format", "species_tsv")])
data = urllib.request.urlopen(f"{BASE}/export/download?{query}", timeout=300).read()
open("mcm_v331_species.tsv", "wb").write(data)
print(f"wrote mcm_v331_species.tsv ({len(data)/1e3:.0f} kB)")
