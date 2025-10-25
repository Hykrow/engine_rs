#[cfg(test)]
    use crate::tensor::Tensor;

    // ---- Helpers (recyclés de ton style de tests matmul) -------------------

    fn t_seq(shape: &[usize]) -> Tensor {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (1..=size).map(|x| x as f32).collect();
        Tensor::from_vec(&data, shape).unwrap()
    }

    fn t_seq_scaled(shape: &[usize], scale: f32) -> Tensor {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (1..=size).map(|x| (x as f32) * scale).collect();
        Tensor::from_vec(&data, shape).unwrap()
    }

    fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
        if shape.is_empty() {
            f(&[]);
            return;
        }
        let mut idx = vec![0usize; shape.len()];
        loop {
            f(&idx);

            // Incrément multi-dims (style "carry" comme dans tes tests matmul)
            let mut carry = true;
            for d in (0..shape.len()).rev() {
                if carry {
                    idx[d] += 1;
                    if idx[d] >= shape[d] {
                        idx[d] = 0;
                        carry = true;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break; // on a débordé la dimension la plus à gauche
            }
        }
    }

    fn assert_tensor_eq_tol(a: &Tensor, b: &Tensor, tol: f32) {
        assert_eq!(a.shape, b.shape, "shapes différentes: {:?} vs {:?}", a.shape, b.shape);
        for_each_index(&a.shape, |i| {
            let da = a.get(i);
            let db = b.get(i);
            assert!(
                (da - db).abs() <= tol,
                "diff à {:?}: got {}, expected {}, tol={}",
                i, da, db, tol
            );
        });
    }

    fn assert_add_ok(a: &Tensor, b: &Tensor) {
        assert_eq!(a.shape, b.shape, "L’addition actuelle requiert des shapes identiques");
        let c = a + b;
        assert_eq!(c.shape, a.shape);
        for_each_index(&a.shape, |i| {
            let expected = a.get(i) + b.get(i);
            let got = c.get(i);
            assert!(
                (got - expected).abs() < 1e-6,
                "mismatch à {:?}: got {}, expected {}",
                i, got, expected
            );
        });
    }

    // ---- Cas OK ------------------------------------------------------------

    #[test]
    fn add_same_shape_1d() {
        let a = t_seq(&[7]);
        let b = t_seq(&[7]);
        assert_add_ok(&a, &b);
    }

    #[test]
    fn add_same_shape_2d() {
        let a = t_seq(&[3, 4]);
        let b = t_seq_scaled(&[3, 4], -1.0);
        assert_add_ok(&a, &b);
    }

    #[test]
    fn add_same_shape_deep_rank() {
        let a = t_seq(&[2, 3, 4, 5]);
        let b = t_seq(&[2, 3, 4, 5]);
        assert_add_ok(&a, &b);
    }

    #[test]
    fn add_commutative() {
        let a = t_seq(&[2, 3, 4]);
        let b = t_seq_scaled(&[2, 3, 4], 2.0);
        let c1 = &a + &b;
        let c2 = &b + &a;
        assert_tensor_eq_tol(&c1, &c2, 0.0);
    }

    #[test]
    fn add_associative() {
        let a = t_seq(&[2, 3, 4]);
        let b = t_seq(&[2, 3, 4]);
        let c = t_seq_scaled(&[2, 3, 4], 0.5);

        let left  = &(&a + &b) + &c; // (a + b) + c
        let right = &a + &(&b + &c); // a + (b + c)
        assert_tensor_eq_tol(&left, &right, 1e-6);
    }

    #[test]
    fn add_with_zeros_identity() {
        let a = t_seq(&[2, 3, 4]);
        let z = t_seq_scaled(&[2, 3, 4], 0.0);
        let c = &a + &z;
        assert_tensor_eq_tol(&c, &a, 0.0);
    }

    #[test]
    fn add_self_doubles_values() {
        let a = t_seq(&[2, 2, 3]);
        let c = &a + &a;
        for_each_index(&a.shape, |i| {
            assert!((c.get(i) - 2.0 * a.get(i)).abs() < 1e-6);
        });
    }

    // ---- Cas d’erreurs attendues (pas de broadcast) ------------------------

    #[test]
    #[should_panic]
    fn add_mismatched_last_dim() {
        // shapes différentes sur la dernière dim
        let a = t_seq(&[3, 5]);
        let b = t_seq(&[3, 4]);
        let _ = &a + &b;
    }

    #[test]
    #[should_panic]
    fn add_mismatched_rank() {
        // rangs différents
        let a = t_seq(&[6]);
        let b = t_seq(&[2, 3]);
        let _ = &a + &b;
    }

    #[test]
    #[should_panic]
    fn add_same_numel_but_different_shape() {
        // même nombre d’éléments mais shapes différentes -> doit paniquer
        let a = t_seq(&[2, 6]); // 12
        let b = t_seq(&[3, 4]); // 12
        let _ = &a + &b;
    }

