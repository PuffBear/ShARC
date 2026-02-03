import argparse
import numpy as np

def summarize_array(name, arr):
    print(f"{name}:")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    if arr.size == 0:
        return
    print(f"  min: {arr.min():.6g}")
    print(f"  max: {arr.max():.6g}")
    print(f"  mean: {arr.mean():.6g}")
    print(f"  std: {arr.std():.6g}")

def main():
    parser = argparse.ArgumentParser(description="Simple EDA for .npz instance files")
    parser.add_argument("path", help="Path to .npz instance file")
    parser.add_argument("--head", type=int, default=5, help="Number of rows to show")
    args = parser.parse_args()

    es = np.load(args.path)
    print("keys:", es.files)

    req = es["req"]
    nonreq = es["nonreq"]

    print(f"P: {es['P']}")
    print(f"M: {es['M']}")
    print(f"C: {es['C']}")

    summarize_array("req", req)
    summarize_array("nonreq", nonreq)

    print("\nreq head:")
    print(req[: args.head])
    print("\nnonreq head:")
    print(nonreq[: args.head])

    # Column-wise summaries: [u, v, demand, class, service_time, distance]
    cols = ["u", "v", "demand", "class", "service_time", "distance"]
    print("\nreq column summaries:")
    for i, name in enumerate(cols):
        col = req[:, i]
        print(f"  {name}: min={col.min():.6g} max={col.max():.6g} mean={col.mean():.6g}")

    print("\nnonreq column summaries:")
    for i, name in enumerate(cols):
        col = nonreq[:, i]
        print(f"  {name}: min={col.min():.6g} max={col.max():.6g} mean={col.mean():.6g}")

if __name__ == "__main__":
    main()
