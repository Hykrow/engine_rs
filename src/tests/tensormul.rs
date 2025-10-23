#[cfg(test)]
    use crate::{ops::tensor_mul, tensor::Tensor};

    fn t_seq(shape: &[usize]) -> Tensor {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (1..=size).map(|x| x as f32).collect();
        Tensor::from_vec(&data, shape).unwrap()
    }

    fn pad_shape(s: &[usize], rank: usize) -> Vec<usize> {
        let mut v = vec![1; rank - s.len()];
        v.extend_from_slice(s);
        v
    }

    fn out_shape_batched(a: &[usize], b: &[usize]) -> Vec<usize> {
        let rank = a.len().max(b.len());
        let ap = pad_shape(a, rank);
        let bp = pad_shape(b, rank);
        assert_eq!(ap[rank - 1], bp[rank - 2]);
        let mut o = vec![0; rank];
        for i in 0..rank - 2 {
            let (da, db) = (ap[i], bp[i]);
            assert!(da == db || da == 1 || db == 1);
            o[i] = da.max(db);
        }
        o[rank - 2] = ap[rank - 2];
        o[rank - 1] = bp[rank - 1];
        o
    }

    fn assert_batched_matmul_ok(a: &Tensor, b: &Tensor) {
        let c = tensor_mul(a, b);
        let oshape = out_shape_batched(&a.shape, &b.shape);
        assert_eq!(c.shape, oshape);

        let rank = oshape.len();
        let br = rank - 2;
        let m = oshape[rank - 2];
        let n = oshape[rank - 1];
        let kdim = pad_shape(&a.shape, rank)[rank - 1];

        let mut batch_idx = vec![0usize; br];

        loop {
            for i in 0..m {
                for j in 0..n {
                    let mut expected = 0.0f32;
                    for k in 0..kdim {
                        let la = a.shape.len();
                        let pa = rank - la;
                        let mut ia = vec![0usize; la];
                        for p in 0..la {
                            if p < la - 2 {
                                let dim = a.shape[p];
                                let src = batch_idx[pa + p];
                                ia[p] = if dim == 1 { 0 } else { src };
                            } else if p == la - 2 {
                                ia[p] = i;
                            } else {
                                ia[p] = k;
                            }
                        }

                        let lb = b.shape.len();
                        let pb = rank - lb;
                        let mut ib = vec![0usize; lb];
                        for p in 0..lb {
                            if p < lb - 2 {
                                let dim = b.shape[p];
                                let src = batch_idx[pb + p];
                                ib[p] = if dim == 1 { 0 } else { src };
                            } else if p == lb - 2 {
                                ib[p] = k;
                            } else {
                                ib[p] = j;
                            }
                        }

                        expected += a.get(&ia) * b.get(&ib);
                    }

                    let mut ic = vec![0usize; rank];
                    for d in 0..br {
                        ic[d] = batch_idx[d];
                    }
                    ic[rank - 2] = i;
                    ic[rank - 1] = j;

                    let got = c.get(&ic);
                    assert!((got - expected).abs() < 1e-5, "mismatch at {:?}: got {}, expected {}", ic, got, expected);
                }
            }

            if br == 0 {
                break;
            }
            let mut carry = true;
            for d in (0..br).rev() {
                if carry {
                    batch_idx[d] += 1;
                    if batch_idx[d] >= oshape[d] {
                        batch_idx[d] = 0;
                        carry = true;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }
    }

    #[test]
    fn batch_same_rank_bmk_kn_bn() {
        let a = t_seq(&[2, 2, 3]);
        let b = t_seq(&[2, 3, 4]);
        assert_batched_matmul_ok(&a, &b);
    }

    #[test]
    fn batch_broadcast_rhs() {
        let a = t_seq(&[2, 2, 3]);
        let b = t_seq(&[3, 4]);
        assert_batched_matmul_ok(&a, &b);
    }

    #[test]
    fn batch_broadcast_lhs() {
        let a = t_seq(&[1, 2, 3]);
        let b = t_seq(&[5, 3, 4]);
        assert_batched_matmul_ok(&a, &b);
    }

    #[test]
    fn batch_two_dims_with_partial_broadcast() {
        let a = t_seq(&[2, 3, 2, 3]);
        let b = t_seq(&[2, 1, 3, 2]);
        assert_batched_matmul_ok(&a, &b);
    }

#[cfg(test)]



    // Copie du helper de vérification : même logique que tes tests existants




#[test]
fn big_batch_small_mnk() {
    // Beaucoup de batch, petites matrices (3x2) x (2x4) -> (3x4)
    let a = t_seq(&[7, 6, 1, 3, 2]);
    let b = t_seq(&[1, 6, 1, 2, 4]);
    assert_batched_matmul_ok(&a, &b);
}

#[test]
fn two_batch_dims_moderately_large_mnk() {
    // (10x11) x (11x12) avec deux dims de batch
    let a = t_seq(&[8, 9, 10, 11]);
    let b = t_seq(&[8, 9, 11, 12]);
    assert_batched_matmul_ok(&a, &b);
}

#[test]
fn broadcast_on_lhs_many_dims() {
    // LHS broadcasté sur plusieurs dims -> sortie [2,3,8,5]
    let a = t_seq(&[1, 1, 8, 16]);
    let b = t_seq(&[2, 3, 16, 5]);
    assert_batched_matmul_ok(&a, &b);
}

#[test]
fn broadcast_on_rhs_many_dims() {
    // RHS broadcasté -> sortie [2,3,32,7]
    let a = t_seq(&[2, 3, 32, 64]);
    let b = t_seq(&[1, 1, 64, 7]);
    assert_batched_matmul_ok(&a, &b);
}

#[test]
fn deep_rank_mixed_broadcast() {
    // Rang 6 vs rang 4 (après padding), mix de dims {=,1} compatibles
    // A[..., 5, 7] x B[..., 7, 8] -> out[..., 5, 8]
    let a = t_seq(&[2, 3, 1, 4, 5, 7]);
    let b = t_seq(&[3, 1, 7, 8]); // pad -> [1,1,3,1,7,8]
    assert_batched_matmul_ok(&a, &b);
}

#[test]
fn k_equals_one_path() {
    // Chemin "k=1" (utile pour tester les strides zéro internes)
    // (5x1) x (1x9) -> (5x9) avec batch [2,3,4]
    let a = t_seq(&[2, 3, 4, 5, 1]);
    let b = t_seq(&[1, 1, 1, 1, 9]);
    assert_batched_matmul_ok(&a, &b);
}

#[test]
fn multi_batch_with_partial_broadcast() {
    // Mélange d'égalité et de 1 dans le broadcast
    // (2x9) x (9x13) -> (2x13), batch [5,4,7] vs [1,4,1] => out [5,4,7]
    let a = t_seq(&[5, 4, 7, 2, 9]);
    let b = t_seq(&[1, 4, 1, 9, 13]);
    assert_batched_matmul_ok(&a, &b);
}

#[test]
fn medium_dense_mnk_three_batch_dims() {
    // (6x8) x (8x6) -> (6x6), batch [3,2,4]
    let a = t_seq(&[3, 2, 4, 6, 8]);
    let b = t_seq(&[3, 2, 4, 8, 6]);
    assert_batched_matmul_ok(&a, &b);
}

// ---- Cas d'erreurs attendues ----

#[test]
#[should_panic]
fn panic_on_mismatched_k() {
    // k incompatible: A[..., m, **3**] vs B[..., **4**, n]
    let a = t_seq(&[2, 5, 3]);
    let b = t_seq(&[2, 4, 4]);
    let _ = tensor_mul(&a, &b);
}

#[test]
#[should_panic]
fn panic_on_incompatible_broadcast() {
    // Broadcast impossible sur batch: [2,3] vs [4,4]
    // (4x5) x (5x6) ok sur les 2 dernières dims, mais batch échoue
    let a = t_seq(&[2, 3, 4, 5]);
    let b = t_seq(&[4, 4, 5, 6]);
    let _ = tensor_mul(&a, &b);
}
