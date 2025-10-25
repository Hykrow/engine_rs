#[cfg(test)]
    use crate::tensor::*;
    

    fn t_seq(shape: &[usize]) -> Tensor {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (1..=size).map(|x| x as f32).collect();
        Tensor::from_vec(&data, shape).unwrap()
    }

    fn assert_tensor_eq_eps(t: &Tensor, expected_data: &[f32], expected_shape: &[usize], eps: f32) {
        assert_eq!(
            t.shape, expected_shape,
            "shape mismatch: got {:?}, expected {:?}",
            t.shape, expected_shape
        );
        let got: Vec<f32> = (0..expected_data.len())
            .map(|i| t.get_from_lin(i))
            .collect();
        for (i, (g, e)) in got.iter().zip(expected_data.iter()).enumerate() {
            assert!(
                (g - e).abs() <= eps,
                "value mismatch at {}: got {}, expected {} (eps={})",
                i, g, e, eps
            );
        }
    }

    #[test]
    fn reduce_row_broadcast_simple() {
        // Parent P = [1,3], gradient G over S = [2,3]
        // G =
        // [[ 1,  2,  3],
        //  [10, 20, 30]]
        // Expected reduction over axis 0 => [11, 22, 33]
        let origin = [1, 3];
        let g = Tensor::from_vec(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 3]).unwrap();

        let red = g.sum_over_broadcasted_batches(&origin);
        let expected = [11.0, 22.0, 33.0];

        assert_tensor_eq_eps(&red, &expected, &origin, 1e-6);
    }

    #[test]
    fn reduce_with_left_padding_and_parent_dim_one() {
        // Parent P = [1,4], Self S = [2,3,4] (padding à gauche + parent_dim==1)
        // Data = 1..=24 in row-major over [2,3,4]
        // Reduction axes: {0 (padding), 1 (P[0]==1)} -> result shape [1,4]
        //
        // Somme analytique par k (dernière dim):
        // sum_{i=0..1} sum_{j=0..2} (i*12 + j*4 + k + 1) = 66 + 6*k
        // => [66, 72, 78, 84]
        let origin = [1, 4];
        let g = t_seq(&[2, 3, 4]); // 1..=24

        let red = g.sum_over_broadcasted_batches(&origin);
        let expected = [66.0, 72.0, 78.0, 84.0];

        assert_tensor_eq_eps(&red, &expected, &origin, 1e-6);
    }

    #[test]
    fn reduce_noop_when_shapes_equal() {
        // Aucun axe à réduire: origin == self.shape
        let origin = [2, 3];
        let g = t_seq(&origin); // 1..=6

        let red = g.sum_over_broadcasted_batches(&origin);
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        assert_tensor_eq_eps(&red, &expected, &origin, 1e-6);
    }

    #[test]
    fn reduce_multiple_axes_parent_2_1_1() {
        // Parent P = [2,1,1], Self S = [2,3,4]
        // Réduction axes: dims où P==1 -> {1,2}
        // Data = 1..=24 in row-major over [2,3,4]
        //
        // Pour i fixé:
        // sum_{j,k} (i*12 + j*4 + k + 1)
        // = i*12*(3*4) + (sum_j j*4)*4 + (sum_k k)*3 + 1*(3*4)
        // = i*144 + 48 + 18 + 12 = i*144 + 78
        // => i=0 -> 78, i=1 -> 222
        let origin = [2, 1, 1];
        let g = t_seq(&[2, 3, 4]); // 1..=24

        let red = g.sum_over_broadcasted_batches(&origin);
        let expected = [78.0, 222.0];

        assert_tensor_eq_eps(&red, &expected, &origin, 1e-6);
    }

    #[test]
    fn reduce_row_broadcast_non_uniform() {
        // Comme le premier test mais mettons des valeurs qui ne sont pas
        // une simple copie pour montrer que ce n'est pas "x nombre de copies".
        // P = [1,3], S = [3,3]
        // G =
        // [[1,  2,  3],
        //  [4,  5,  6],
        //  [7,  8,  9]]
        // sum over axis 0 -> [12, 15, 18]
        let origin = [1, 3];
        let g = Tensor::from_vec(
            &[
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0
            ],
            &[3, 3]
        ).unwrap();

        let red = g.sum_over_broadcasted_batches(&origin);
        let expected = [12.0, 15.0, 18.0];

        assert_tensor_eq_eps(&red, &expected, &origin, 1e-6);
    }

