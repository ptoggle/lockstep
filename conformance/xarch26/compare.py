# Compare the kit's digests across machines: python compare.py digests_h100.json digests_ada.json digests_blackwell.json
import json, sys
files = sys.argv[1:]
D = {f: json.load(open(f)) for f in files}
names = [D[f]["info"]["gpu"] + " (sm_" + D[f]["info"]["capability"].replace(".", "") + ", torch " + D[f]["info"]["torch"] + ")" for f in files]
for f, n in zip(files, names):
    d = D[f]; print("%-60s local gate %-12s kernel digest %s  reference %s  inputs %s" % (n, d["local_gate"], d["all_kernel_outputs"][:16], d["all_reference_outputs"][:16], d["info"]["inputs_sha"]))
base = D[files[0]]
cases = sorted(base["cases"]); n_out = 0; n_diff = 0; diffs = []
for c in cases:
    for k, v in base["cases"][c]["outputs"].items():
        n_out += 1
        for f in files[1:]:
            w = D[f]["cases"].get(c, {}).get("outputs", {}).get(k)
            if w != v: n_diff += 1; diffs.append((f, c, k))
print("cases %d, output digests %d per machine, differing across machines: %d" % (len(cases), n_out, n_diff))
for f, c, k in diffs[:20]: print("  DIFF", f, c, k)
fails = [(f, c) for f in files for c in D[f]["cases"] if not D[f]["cases"][c]["local_check"].startswith("PASS")]
print("local-check failures:", len(fails)); [print("  FAIL", f, c, D[f]["cases"][c]["local_check"]) for f, c in fails[:20]]
print("VERDICT:", "IDENTICAL on all machines and every local check passes" if (n_diff == 0 and not fails) else "NOT identical or a local check failed")
