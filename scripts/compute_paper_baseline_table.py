#!/usr/bin/env python3
import re
import urllib.request
import numpy as np

html = urllib.request.urlopen("https://arxiv.org/html/2303.13845v2").read().decode("utf-8")
idx = html.find('id="A2.T11"')
end_idx = html.find("</table>", idx)
table_str = html[idx:end_idx]

trs = re.findall(r"<tr[^>]*>(.*?)</tr>", table_str, re.DOTALL)
methods = ["Deep-SVDD", "f-AnoGAN", "KDAD", "RD4AD", "Augmix", "Mixstyle", "EFDM", "Jigsaw", "GNL (Ours)"]
corruptions = ["ID", "Brightness", "Contrast", "Blur", "Noise"]

results = {m: {c: [] for c in corruptions} for m in methods}

for tr in trs:
    tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
    clean_tds = [re.sub(r"<[^>]+>", "", td).strip() for td in tds]
    if not clean_tds:
        continue
    m_name = clean_tds[0]
    for m in methods:
        if m == m_name or (m.startswith("GNL") and m_name.startswith("GNL")):
            vals = [float(x) for x in clean_tds[1:] if x.replace(".", "", 1).isdigit()]
            for i in range(len(vals)):
                corr_idx = i % 5
                results[m][corruptions[corr_idx]].append(vals[i])

print(f"{'Method (ICCV 2023 Paper)':<22} | {'Clean (ID)':>10} | {'Brightness':>10} | {'Contrast':>10} | {'Blur':>10} | {'Noise':>10} | {'Avg OOD':>10}")
print("-" * 90)
for m in methods:
    row = f"{m:<22} | "
    ood_vals = []
    for c in corruptions:
        mean_v = np.mean(results[m][c]) if results[m][c] else 0.0
        row += f"{mean_v:>10.2f} | "
        if c != "ID":
            ood_vals.append(mean_v)
    row += f"{np.mean(ood_vals):>10.2f}"
    print(row)
