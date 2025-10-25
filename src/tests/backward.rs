#[cfg(test)]
    use std::iter;
    use crate::{tensor::Tensor, ops::{tensor_mul, Op}};

    // ----------------- Helpers génériques -----------------

    fn t_seq(shape: &[usize]) -> Tensor {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (1..=size).map(|x| x as f32).collect();
        Tensor::from_owned(data, shape).unwrap()
    }

    fn ones_like_shape(shape: &[usize]) -> Tensor {
        let size: usize = shape.iter().product();
        Tensor::from_owned(vec![1.0; size], shape).unwrap()
    }

    fn zeros_like_shape(shape: &[usize]) -> Tensor {
        let size: usize = shape.iter().product();
        Tensor::from_owned(vec![0.0; size], shape).unwrap()
    }

    fn for_each_index(shape: &[usize], mut f: impl FnMut(Vec<usize>)) {
        let total: usize = shape.iter().product();
        for lin in 0..total {
            let idx = Tensor::idx_from_lin(shape, lin);
            f(idx);
        }
    }

    fn assert_tensor_close(a: &Tensor, b: &Tensor, tol: f32) {
        &assert_eq!(a.shape, b.shape, "Shape mismatch: {:?} vs {:?}", a.shape, b.shape);
        for_each_index(&a.shape, |idx| {
            let da = a.get(&idx);
            let db = b.get(&idx);
            assert!(
                (da - db).abs() <= tol,
                "Diff at {:?}: got {}, expected {}, tol={}",
                idx, da, db, tol
            );
        });
    }

    /// Réduit `g_out` (shape sortie) vers `target_shape` en sommant sur les axes broadcastés.
    fn sum_reduce_to_shape(g_out: &Tensor, target_shape: &[usize]) -> Tensor {
        // Aligner les rangs par left-padding de 1
        let rank_out = g_out.shape.len();
        let rank_tgt = target_shape.len();
        let rank = rank_out.max(rank_tgt);

        let pad = |s: &[usize], r: usize| {
            let mut v = vec![1; r - s.len()];
            v.extend_from_slice(s);
            v
        };
        let outp = pad(&g_out.shape, rank);
        let tgtp = pad(target_shape, rank);

        // Accumulateur
        let mut acc = vec![0.0f32; target_shape.iter().product()];

        // Pour convertir un index "plein" (rank) vers index pour target, on clamp les axes où tgt dim = 1 à 0.
        for_each_index(&outp, |full_idx| {
            let mut t_idx = vec![0usize; rank];
            for d in 0..rank {
                t_idx[d] = if tgtp[d] == 1 { 0 } else { full_idx[d] };
            }
            // Retirer le padding pour calculer la position dans acc
            let t_idx_trim = t_idx[(rank - rank_tgt)..].to_vec();

            let val = g_out.get(&full_idx[(rank - rank_out)..]); // map full_idx -> g_out index (skip left pad)
            // position linéaire dans acc (contigu)
            let mut lin = 0usize;
            let mut stride = 1usize;
            for d in (0..rank_tgt).rev() {
                lin += t_idx_trim[d] * stride;
                stride *= target_shape[d];
            }
            acc[lin] += val;
        });

        Tensor::from_owned(acc, target_shape).unwrap()
    }

    // Elementwise f(a,b) sur la même shape (respecte shape/strides via get)
    fn map2_same_shape(a: &Tensor, b: &Tensor, mut f: impl FnMut(f32, f32) -> f32) -> Tensor {
        assert_eq!(a.shape, b.shape);
        let mut out = Vec::with_capacity(a.shape.iter().product());
        for lin in 0..a.shape.iter().product::<usize>() {
            let idx = Tensor::idx_from_lin(&a.shape, lin);
            out.push(f(a.get(&idx), b.get(&idx)));
        }
        Tensor::from_owned(out, &a.shape).unwrap()
    }

    // ----------------- Tests ADD (avec broadcast) -----------------
    
    #[test]
    fn backward_add_no_broadcast() {
        let a = t_seq(&[2, 3, 4]);
        let b = t_seq(&[2, 3, 4]);
        let y = Op::Add.forward(&[&a, &b]);

        // gradient amont arbitraire
        let g_out = t_seq(&y.shape);

        // backward (sous test)
        let grads = Op::Add.backward(&[&a, &b], &y, &g_out);
        let (ga, gb) = (&grads[0], &grads[1]);

        // attendu : pass-through vers chaque parent
        let exp_ga = g_out.clone();
        let exp_gb = g_out.clone();

        // shapes doivent matcher celles des PARENTS
        assert_eq!(ga.shape, a.shape);
        assert_eq!(gb.shape, b.shape);

        assert_tensor_close(ga, &exp_ga, 1e-6);
        assert_tensor_close(gb, &exp_gb, 1e-6);
    }

    #[test]
    fn backward_add_broadcast_rhs() {
        // a: [2,3,4], b: [1,3,4] -> y: [2,3,4]
        let a = t_seq(&[2, 3, 4]);
        let b = t_seq(&[1, 3, 4]);
        let y = Op::Add.forward(&[&a, &b]);

        let g_out = t_seq(&y.shape);

        let grads = Op::Add.backward(&[&a, &b], &y, &g_out);
        let (ga, gb) = (&grads[0], &grads[1]);

        // attendu :
        // dL/da = g_out
        // dL/db = sum_{batch axis broadcastés} g_out  (ici somme sur axe 0)
        let exp_ga = g_out.clone();
        let exp_gb = sum_reduce_to_shape(&g_out, &b.shape);

        assert_eq!(ga.shape, a.shape);
        assert_eq!(gb.shape, b.shape);

        assert_tensor_close(ga, &exp_ga, 1e-6);
        assert_tensor_close(gb, &exp_gb, 1e-6);
    }

    #[test]
    fn backward_add_broadcast_lhs_many_dims() {
        // a: [1,2,1,4], b: [3,2,5,4] -> y: [3,2,5,4]
        let a = t_seq(&[1, 2, 1, 4]);
        let b = t_seq(&[3, 2, 5, 4]);
        let y = Op::Add.forward(&[&a, &b]);

        let g_out = t_seq(&y.shape);

        let grads = Op::Add.backward(&[&a, &b], &y, &g_out);
        let (ga, gb) = (&grads[0], &grads[1]);

        let exp_ga = sum_reduce_to_shape(&g_out, &a.shape);
        let exp_gb = g_out.clone();

        assert_tensor_close(ga, &exp_ga, 1e-6);
        assert_tensor_close(gb, &exp_gb, 1e-6);
    }

    // ----------------- Tests TANH -----------------

    #[test]
    fn backward_tanh_elementwise() {
        let x = t_seq(&[2, 3, 4]);
        let y = Op::Tanh.forward(&[&x]); // y = tanh(x)

        // gradient amont quelconque
        let g_out = t_seq(&y.shape);

        let grads = Op::Tanh.backward(&[&x], &y, &g_out);
        let gx = &grads[0];

        // attendu : dL/dx = g_out * (1 - y^2)
        let one_minus_y2 = map2_same_shape(&y, &y, |u, v| 1.0 - u * v);
        let exp_gx = map2_same_shape(&g_out, &one_minus_y2, |a, b| a * b);

        assert_eq!(gx.shape, x.shape);
        assert_tensor_close(gx, &exp_gx, 1e-6);
    }

    // ----------------- Tests ReLU -----------------

    #[test]
    fn backward_relu_elementwise() {
        // Construire un x qui contient négatifs, zéros et positifs
        let x = Tensor::from_owned(
            vec![
                -2.0, -0.5, 0.0, 0.5,
                1.0,  2.0, -1.0, 3.0,
            ],
            &[2, 4],
        ).unwrap();

        let y = Op::ReLU.forward(&[&x]); // y = max(x, 0)

        // gradient amont
        let g_out = t_seq(&y.shape); // 1..=8

        let grads = Op::ReLU.backward(&[&x], &y, &g_out);
        let gx = &grads[0];

        // attendu : dL/dx = g_out si x>0, sinon 0 (convention à 0 en x=0)
        let exp_gx = {
            let mut v = Vec::with_capacity(x.shape.iter().product());
            for lin in 0..x.shape.iter().product::<usize>() {
                let idx = Tensor::idx_from_lin(&x.shape, lin);
                let xi = x.get(&idx);
                let gi = g_out.get(&idx);
                v.push(if xi > 0.0 { gi } else { 0.0 });
            }
            Tensor::from_owned(v, &x.shape).unwrap()
        };

        assert_tensor_close(gx, &exp_gx, 1e-6);
    }

    // ----------------- Tests MATMUL (2D + batch + broadcast) -----------------

    #[test]
    fn backward_matmul_2d() {
        // A:[3,5], B:[5,4] => Y:[3,4]
        let a = t_seq(&[3, 5]);
        let b = t_seq(&[5, 4]);
        let y = Op::MatMul.forward(&[&a, &b]);

        let g_out = t_seq(&y.shape);

        let grads = Op::MatMul.backward(&[&a, &b], &y, &g_out);
        let (ga, gb) = (&grads[0], &grads[1]);

        // attendu :
        // dA = g_out @ B^T
        // dB = A^T @ g_out
        let exp_ga = tensor_mul(&g_out, &b.mat_transpose());
        let exp_gb = tensor_mul(&a.mat_transpose(), &g_out);

        assert_tensor_close(ga, &exp_ga, 1e-6);
        assert_tensor_close(gb, &exp_gb, 1e-6);
    }
    
    #[test]
    fn backward_matmul_batched_no_broadcast() {
        // A:[2,3,5], B:[2,5,4] => Y:[2,3,4]
        let a = t_seq(&[2, 3, 5]);
        let b = t_seq(&[2, 5, 4]);
        let y = Op::MatMul.forward(&[&a, &b]);

        let g_out = t_seq(&y.shape);

        let grads = Op::MatMul.backward(&[&a, &b], &y, &g_out);
        let (ga, gb) = (&grads[0], &grads[1]);

        let exp_ga = tensor_mul(&g_out, &b.mat_transpose());   // [2,3,5]
        let exp_gb = tensor_mul(&a.mat_transpose(), &g_out);   // [2,5,4]

        assert_tensor_close(ga, &exp_ga, 1e-6);
        assert_tensor_close(gb, &exp_gb, 1e-6);
    }

    #[test]
    fn backward_matmul_broadcast_rhs() {
        // B broadcasté sur batch
        // A:[2,3,5], B:[1,5,4] => Y:[2,3,4]
        let a = t_seq(&[2, 3, 5]);
        let b = t_seq(&[1, 5, 4]);
        let y = Op::MatMul.forward(&[&a, &b]);

        let g_out = t_seq(&y.shape);

        let grads = Op::MatMul.backward(&[&a, &b], &y, &g_out);
        let (ga, gb) = (&grads[0], &grads[1]);

        // dA = g_out @ B^T  (shape broadcastée [2,3,5], pas de réduction)
        let exp_ga = tensor_mul(&g_out, &b.mat_transpose());

        // dB_full = A^T @ g_out  (shape broadcastée [2,5,4])
        // puis réduire sur batch vers la shape de B: [1,5,4] -> somme sur axe batch
        let dB_full = tensor_mul(&a.mat_transpose(), &g_out);
        let exp_gb = sum_reduce_to_shape(&dB_full, &b.shape);

        assert_tensor_close(ga, &exp_ga, 1e-6);
        assert_tensor_close(gb, &exp_gb, 1e-6);
    }

    #[test]
    fn backward_matmul_broadcast_lhs() {
        // A broadcasté sur batch
        // A:[1,3,5], B:[2,5,4] => Y:[2,3,4]
        let a = t_seq(&[1, 3, 5]);
        let b = t_seq(&[2, 5, 4]);

        print!("MAT A : {}, MAT B : {}", a, b);
        let y = Op::MatMul.forward(&[&a, &b]);
        print!("MAT Y {}", y);

        let g_out = t_seq(&y.shape);

        let grads = Op::MatMul.backward(&[&a, &b], &y, &g_out);
        let (ga, gb) = (&grads[0], &grads[1]);

        // dA_full = g_out @ B^T  (shape [2,3,5]), puis réduire vers [1,3,5]
        let dA_full = tensor_mul(&g_out, &b.mat_transpose());
        let exp_ga = sum_reduce_to_shape(&dA_full, &a.shape);

        // dB = A^T @ g_out  (pas de réduction côté B)
        let exp_gb = tensor_mul(&a.mat_transpose(), &g_out);

        assert_tensor_close(ga, &exp_ga, 1e-6);
        assert_tensor_close(gb, &exp_gb, 1e-6);
    }
