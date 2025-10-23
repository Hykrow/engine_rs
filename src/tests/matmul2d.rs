// ecrit avec ChatGPT

#[cfg(test)]
use crate::{tensor::*, ops::*};

fn t(data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_vec(data, shape).unwrap()
}

#[test]
fn matmul_2d_2d_basic() {
    // A: 2x3
    let a = t(&[1.0, 2.0, 3.0,
                4.0, 5.0, 6.0], &[2, 3]);
    // B: 3x2
    let b = t(&[7.0, 8.0,
                9.0, 10.0,
                11.0, 12.0], &[3, 2]);

    let c = tensor_mul(&a, &b);
    assert_eq!(c.shape, vec![2, 2]);
    assert_eq!(c.get(&[0, 0]), 58.0); // 1*7 + 2*9 + 3*11
    assert_eq!(c.get(&[0, 1]), 64.0); // 1*8 + 2*10 + 3*12
    assert_eq!(c.get(&[1, 0]), 139.0); // 4*7 + 5*9 + 6*11
    assert_eq!(c.get(&[1, 1]), 154.0); // 4*8 + 5*10 + 6*12
}

#[test]
fn matmul_1d_1d_dot_gives_1x1() {
    // a: [3], b: [3] -> (1x1)
    let a = t(&[1.0, 2.0, 3.0], &[3]);
    let b = t(&[4.0, 5.0, 6.0], &[3]);

    let c = tensor_mul(&a, &b);
    assert_eq!(c.shape, vec![1, 1]);
    assert_eq!(c.get(&[0, 0]), 32.0); // 1*4 + 2*5 + 3*6
}

#[test]
fn matmul_2d_1d_matrix_times_vector() {
    // A: 2x3, v: [3] -> (2x1)
    let a = t(&[1.0, 2.0, 3.0,
                4.0, 5.0, 6.0], &[2, 3]);
    let v = t(&[7.0, 9.0, 11.0], &[3]);

    let c = tensor_mul(&a, &v);
    assert_eq!(c.shape, vec![2, 1]);
    assert_eq!(c.get(&[0, 0]), 58.0);  // 1*7 + 2*9 + 3*11
    assert_eq!(c.get(&[1, 0]), 139.0); // 4*7 + 5*9 + 6*11
}

#[test]
fn matmul_1d_2d_vector_times_matrix() {
    // u: [2], B: 2x3 -> (1x3)
    let u = t(&[2.0, 3.0], &[2]);
    let b = t(&[1.0, 4.0, 7.0,
                2.0, 5.0, 8.0], &[2, 3]); // rows: [1,4,7]; [2,5,8]

    let c = tensor_mul(&u, &b);
    assert_eq!(c.shape, vec![1, 3]);
    assert_eq!(c.get(&[0, 0]), 2.0*1.0 + 3.0*2.0); // 8
    assert_eq!(c.get(&[0, 1]), 2.0*4.0 + 3.0*5.0); // 23
    assert_eq!(c.get(&[0, 2]), 2.0*7.0 + 3.0*8.0); // 38
}

#[test]
fn matmul_2d_2d_non_square() {
    // A: 3x2, B: 2x4 -> C: 3x4
    let a = t(&[1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0], &[3, 2]);
    let b = t(&[7.0, 8.0, 9.0, 10.0,
                11.0, 12.0, 13.0, 14.0], &[2, 4]);

    let c = tensor_mul(&a, &b);
    assert_eq!(c.shape, vec![3, 4]);

    assert_eq!(c.get(&[0, 0]), 1.0*7.0  + 2.0*11.0);
    assert_eq!(c.get(&[0, 1]), 1.0*8.0  + 2.0*12.0);
    assert_eq!(c.get(&[0, 2]), 1.0*9.0  + 2.0*13.0);
    assert_eq!(c.get(&[0, 3]), 1.0*10.0 + 2.0*14.0);

    assert_eq!(c.get(&[1, 0]), 3.0*7.0  + 4.0*11.0);
    assert_eq!(c.get(&[1, 1]), 3.0*8.0  + 4.0*12.0);
    assert_eq!(c.get(&[1, 2]), 3.0*9.0  + 4.0*13.0);
    assert_eq!(c.get(&[1, 3]), 3.0*10.0 + 4.0*14.0);

    assert_eq!(c.get(&[2, 0]), 5.0*7.0  + 6.0*11.0);
    assert_eq!(c.get(&[2, 1]), 5.0*8.0  + 6.0*12.0);
    assert_eq!(c.get(&[2, 2]), 5.0*9.0  + 6.0*13.0);
    assert_eq!(c.get(&[2, 3]), 5.0*10.0 + 6.0*14.0);
}

