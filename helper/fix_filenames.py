#!/usr/bin/env python3
import os, re

pattern = re.compile(r'^(data_)0*(\d+)(\.h5)$')
for fname in os.listdir('.'):
    m = pattern.match(fname)
    if m:
        prefix, num_str, suffix = m.groups()
        num = int(num_str)
        newname = f"{prefix}{num:06d}{suffix}"
        if fname != newname:
            os.rename(fname, newname)
            print(f"Renamed {fname} → {newname}")
