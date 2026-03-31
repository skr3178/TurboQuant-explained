Algorithm 1 TurboQuantmse: optimized for MSE
1: input: dimension d and bit-width b
// Global Parameters for Setting up TurboQuantmse
2: Generate a random rotation matrix Π ∈ Rd×d
3: Construct codebook by finding centroids c1, c2, . . . c2b ∈ [−1, 1] that minimize MSE cost in
Eq. (4)
4: Procedure Quantmse(x)
5: y ← Π· x
6: idxj ← arg mink∈[2b] |yj − ck| for every j ∈ [d] {idxj’s are b-bit integers}
7: output: idx
8: Procedure DeQuantmse(idx)
9: ˜yj ← cidxj for every j ∈ [d]
10: ˜x ← Π⊤ · ˜y
11: output: ˜x



Algorithm 2 TurboQuantprod: optimized for inner product
1: input: dimension d and bit-width b
// Global Parameters for Setting up TurboQuantprod
2: Instantiate a TurboQuantmse with bit-width b − 1 as per Algorithm 1
3: Generate a random projection matrix S ∈ Rd×d with i.i.d. entries Si,j ∼ N(0, 1)
4: Procedure Quantprod(x)
5: idx ← Quantmse(x)
6: r ← x − DeQuantmse(idx) {residual vector}
7: qjl ← sign (S · r) {QJL on residual vector}
8: output: (idx, qjl, ∥r∥2)
9: Procedure DeQuantprod(idx, qjl, γ)
10: ˜xmse ← DeQuantmse(idx)
11: ˜xqjl ←
√π/2
d · γ · S⊤ · qjl
12: output: ˜xmse + ˜xqjl