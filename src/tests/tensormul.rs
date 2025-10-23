#[cfg(test)]
mod tests {
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
}
