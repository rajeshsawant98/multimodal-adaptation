import numpy as np, faiss
d = 64
xb = np.random.rand(1000, d).astype("float32")
xq = np.random.rand(3, d).astype("float32")
faiss.normalize_L2(xb); faiss.normalize_L2(xq)
index = faiss.IndexFlatIP(d)
index.add(xb)
D, I = index.search(xq, 5)
print("✅ FAISS top-5 for first query:", I[0].tolist(), "| scores:", [round(float(s),3) for s in D[0]])